"""
Tender Intelligence — Streamlit Frontend
Interactive UI for uploading tenders, running analysis, and asking questions.
"""

import json
import requests
import streamlit as st
from pathlib import Path

API_URL = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Tender Intelligence | HBT",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🏗️ Tender Intelligence")
    st.markdown("*AI-powered tender analysis for tunnelling projects*")
    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API online")
        st.metric("Documents loaded", health["documents_loaded"])
        st.metric("Index ready", "✓" if health["index_exists"] else "✗")
    except Exception:
        st.error("API offline — is the FastAPI server running?")
        st.code("uvicorn app.api:app --reload", language="bash")

    st.divider()

    # Load sample documents
    st.markdown("**Quick start**")
    if st.button("Load sample tenders", use_container_width=True):
        with st.spinner("Ingesting sample documents..."):
            try:
                r = requests.post(f"{API_URL}/ingest/sample", timeout=60)
                if r.ok:
                    st.success(f"Loaded {len(r.json()['documents'])} sample tenders")
                    st.rerun()
                else:
                    st.error(r.json().get("detail", "Error"))
            except Exception as e:
                st.error(str(e))

    if st.button("Clear index", use_container_width=True, type="secondary"):
        requests.delete(f"{API_URL}/index", timeout=10)
        st.success("Index cleared")
        st.rerun()

    st.divider()
    st.caption("Built by Girish Kundal · BCU MSc AI 2025")
    st.caption("Stack: FastAPI · LangChain · FAISS · GPT-3.5 · Docker")


# ── Main tabs ────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Upload & Ingest",
    "📊 Structured Analysis",
    "💬 Ask a Question",
    "🏗️ Architecture",
])


# ── Tab 1: Upload ─────────────────────────────────────────────────────────────

with tab1:
    st.header("Upload Tender Documents")
    st.markdown(
        "Upload a tender document (.txt or .pdf) to ingest it into the RAG pipeline. "
        "Multiple documents can be loaded — they share one FAISS vector index."
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload tender document",
            type=["txt", "pdf"],
            help="Supports plain text and PDF tender documents",
        )

        if uploaded:
            if st.button("Ingest document", type="primary"):
                with st.spinner(f"Ingesting {uploaded.name}..."):
                    try:
                        r = requests.post(
                            f"{API_URL}/ingest",
                            files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                            timeout=120,
                        )
                        if r.ok:
                            st.success(f"✓ Ingested: {r.json()['filename']}")
                            st.json(r.json())
                        else:
                            st.error(r.json().get("detail", "Ingestion failed"))
                    except Exception as e:
                        st.error(str(e))

    with col2:
        st.markdown("**Ingested documents**")
        try:
            docs = requests.get(f"{API_URL}/documents", timeout=5).json()
            if docs["count"] == 0:
                st.info("No documents loaded yet.")
            for doc in docs["documents"]:
                st.markdown(f"📄 `{doc['filename']}`")
                st.caption(f"Ingested: {doc['ingested_at'][:19].replace('T', ' ')}")
        except Exception:
            st.warning("Could not fetch document list.")

    st.divider()
    st.markdown("**What happens during ingestion?**")
    steps = {
        "1. Load": "Document is read from disk (TextLoader for .txt, PyPDFLoader for .pdf)",
        "2. Chunk": f"Split into 512-token segments with 64-token overlap using RecursiveCharacterTextSplitter",
        "3. Embed": "Each chunk embedded using OpenAI text-embedding-3-small (1,536 dimensions)",
        "4. Index": "Embeddings stored in FAISS vector store for sub-millisecond similarity search",
    }
    cols = st.columns(4)
    for col, (step, desc) in zip(cols, steps.items()):
        with col:
            st.markdown(f"**{step}**")
            st.caption(desc)


# ── Tab 2: Structured Analysis ────────────────────────────────────────────────

with tab2:
    st.header("Structured Tender Analysis")
    st.markdown(
        "Runs 14 targeted extraction queries against the ingested documents. "
        "Each field is retrieved via semantic search then answered by GPT-3.5."
    )

    if st.button("Run full analysis", type="primary", use_container_width=True):
        with st.spinner("Extracting tender intelligence... (14 queries, ~30 seconds)"):
            try:
                r = requests.post(f"{API_URL}/analyse", timeout=120)
                if r.ok:
                    data = r.json()

                    # Summary banner
                    st.success(data["summary"])

                    extraction = data["extraction"]

                    # Key metrics row
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Contract value", extraction.get("contract_value", {}).get("value", "—")[:40])
                    col2.metric("Duration", extraction.get("contract_duration", {}).get("value", "—")[:20])
                    col3.metric("Start date", extraction.get("start_date", {}).get("value", "—")[:20])
                    col4.metric("Deadline", extraction.get("tender_deadline", {}).get("value", "—")[:20])

                    st.divider()

                    # Main extraction table
                    left, right = st.columns(2)

                    left_fields = [
                        "project_name", "client", "contract_reference",
                        "location", "tunnel_type", "boq_total", "scope_summary",
                    ]
                    right_fields = [
                        "key_requirements", "evaluation_criteria",
                        "key_risks",
                    ]

                    with left:
                        st.markdown("**Project details**")
                        for field in left_fields:
                            info = extraction.get(field, {})
                            label = field.replace("_", " ").title()
                            st.markdown(f"**{label}**")
                            st.markdown(info.get("value", "—"))
                            st.divider()

                    with right:
                        st.markdown("**Commercial intelligence**")
                        for field in right_fields:
                            info = extraction.get(field, {})
                            label = field.replace("_", " ").title()
                            st.markdown(f"**{label}**")
                            st.markdown(info.get("value", "—"))
                            st.divider()

                    # Raw JSON expander
                    with st.expander("View raw JSON output"):
                        st.json(extraction)

                else:
                    st.error(r.json().get("detail", "Analysis failed"))
            except Exception as e:
                st.error(str(e))


# ── Tab 3: Free-form Q&A ──────────────────────────────────────────────────────

with tab3:
    st.header("Ask a Question")
    st.markdown(
        "Ask anything about the ingested tender documents. "
        "The RAG pipeline retrieves the most relevant chunks and passes them to GPT-3.5."
    )

    # Suggested questions
    st.markdown("**Suggested questions**")
    suggestions = [
        "What are the ground conditions for this project?",
        "What is the programme for tunnel construction?",
        "What insurance or certification requirements are specified?",
        "Summarise the key risks identified.",
        "What are the evaluation criteria and their weightings?",
        "What monitoring and instrumentation is required?",
    ]

    cols = st.columns(3)
    selected = None
    for i, suggestion in enumerate(suggestions):
        if cols[i % 3].button(suggestion, use_container_width=True):
            selected = suggestion

    st.divider()

    question = st.text_input(
        "Your question",
        value=selected or "",
        placeholder="e.g. What is the contract value and duration?",
    )

    if st.button("Ask", type="primary") and question:
        with st.spinner("Retrieving relevant chunks and querying GPT-3.5..."):
            try:
                r = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question},
                    timeout=60,
                )
                if r.ok:
                    result = r.json()

                    st.markdown("### Answer")
                    st.info(result["answer"])

                    with st.expander(f"Source chunks ({len(result['source_chunks'])} retrieved)"):
                        for i, chunk in enumerate(result["source_chunks"], 1):
                            st.markdown(f"**Chunk {i}** — `{chunk['file']}`")
                            st.text(chunk["excerpt"])
                            st.divider()
                else:
                    st.error(r.json().get("detail", "Query failed"))
            except Exception as e:
                st.error(str(e))


# ── Tab 4: Architecture ───────────────────────────────────────────────────────

with tab4:
    st.header("System Architecture")

    st.markdown("""
    ### RAG Pipeline — How it works

    This system implements a **Retrieval-Augmented Generation (RAG)** architecture
    for tender document intelligence. The key insight: instead of fine-tuning an LLM
    on tender documents (expensive, slow), we retrieve the relevant context at query
    time and pass it to a general-purpose LLM.

    ---

    **Stage 1 — Ingestion**
    ```
    Tender PDF/TXT
         │
         ▼
    Document Loader (LangChain)
         │
         ▼
    RecursiveCharacterTextSplitter
    chunk_size=512, overlap=64
         │
         ▼
    OpenAI text-embedding-3-small
    1,536-dimension vectors
         │
         ▼
    FAISS Vector Index (on disk)
    ```

    **Stage 2 — Retrieval & Generation**
    ```
    User Question
         │
         ▼
    Embed question → query vector
         │
         ▼
    FAISS similarity search (k=6)
    → top 6 most relevant chunks
         │
         ▼
    Prompt construction:
    [System] + [Context chunks] + [Question]
         │
         ▼
    GPT-3.5-turbo (temperature=0)
         │
         ▼
    Structured answer + source citations
    ```

    ---

    ### Design decisions

    | Decision | Choice | Reason |
    |----------|--------|--------|
    | Chunk size | 512 tokens | Balances context preservation vs retrieval precision |
    | Chunk overlap | 64 tokens | Prevents information loss at boundaries |
    | k (retrieved chunks) | 6 | Covers multi-part questions without exceeding context window |
    | Temperature | 0 | Deterministic outputs for commercial use — no hallucination variance |
    | Embedding model | text-embedding-3-small | Cost-efficient, strong semantic search performance |

    ---

    ### Stack
    - **LangChain** — document loading, chunking, chain orchestration
    - **FAISS** — Facebook AI Similarity Search — sub-millisecond vector retrieval
    - **OpenAI** — embeddings (text-embedding-3-small) and LLM (GPT-3.5-turbo)
    - **FastAPI** — REST API backend with async endpoints
    - **Streamlit** — interactive frontend UI
    - **Docker** — containerised deployment

    ---

    ### Relevance to HBT KTP
    The core technical challenge of the KTP — *take unstructured tender documents,
    extract what matters, and produce structured outputs a commercial team can act on*
    — is exactly what this system demonstrates in a working, deployed form.
    """)
