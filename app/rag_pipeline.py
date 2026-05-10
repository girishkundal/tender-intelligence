"""
Tender Intelligence RAG Pipeline
Core document ingestion, embedding, retrieval and LLM reasoning engine.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader, PyPDFLoader


# ── Configuration ────────────────────────────────────────────────────────────

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"
FAISS_INDEX_PATH = "data/faiss_index"


# ── Document Ingestion ────────────────────────────────────────────────────────

def load_document(file_path: str) -> list:
    """
    Load a document from disk. Supports .txt and .pdf.
    Returns a list of LangChain Document objects.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        loader = TextLoader(str(path), encoding="utf-8")

    documents = loader.load()

    # Tag each chunk with source metadata
    for doc in documents:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_hash"] = _file_hash(path)

    return documents


def chunk_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks for embedding.
    512 tokens with 64-token overlap preserves context across boundaries.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"  Chunked into {len(chunks)} segments (chunk_size={CHUNK_SIZE})")
    return chunks


def _file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()[:8]


# ── Vector Store ──────────────────────────────────────────────────────────────

def build_vector_store(chunks: list, save_path: str = FAISS_INDEX_PATH) -> FAISS:
    """
    Embed document chunks and store in FAISS vector index.
    Saves index to disk for reuse.
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"  Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")

    vector_store = FAISS.from_documents(chunks, embeddings)

    Path(save_path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"  FAISS index saved to {save_path}/")

    return vector_store


def load_vector_store(index_path: str = FAISS_INDEX_PATH) -> Optional[FAISS]:
    """Load an existing FAISS index from disk."""
    path = Path(index_path)
    if not path.exists():
        return None
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


def add_to_vector_store(
    new_chunks: list, index_path: str = FAISS_INDEX_PATH
) -> FAISS:
    """Add new document chunks to an existing FAISS index."""
    existing = load_vector_store(index_path)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if existing:
        existing.add_documents(new_chunks)
        existing.save_local(index_path)
        return existing
    else:
        return build_vector_store(new_chunks, index_path)


# ── LLM Reasoning ────────────────────────────────────────────────────────────

TENDER_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an expert tender analyst for a tunnelling and civil engineering contractor.
You are given excerpts from a tender document and must answer the question accurately.

Instructions:
- Answer only from the provided context. Do not invent values.
- If a value is not found, respond with "Not specified in document."
- For financial values, always include the currency and unit.
- Be concise and precise — a commercial team will act on your output.

Context from tender document:
{context}

Question: {question}

Answer:"""
)


def build_qa_chain(vector_store: FAISS, k: int = 6) -> RetrievalQA:
    """
    Build a retrieval-augmented QA chain.
    k=6 retrieves the 6 most relevant chunks per query.
    """
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": TENDER_EXTRACTION_PROMPT},
    )

    return chain


# ── Structured Extraction ─────────────────────────────────────────────────────

EXTRACTION_QUESTIONS = {
    "project_name": "What is the project name or title?",
    "client": "Who is the client or issuing authority?",
    "contract_reference": "What is the contract reference number?",
    "tender_deadline": "What is the tender submission deadline or return date?",
    "contract_value": "What is the total contract value or estimated value range?",
    "contract_duration": "What is the contract duration in months?",
    "start_date": "What is the planned contract start date or commencement date?",
    "scope_summary": "Summarise the main scope of works in 2-3 sentences.",
    "key_requirements": "What are the essential qualification requirements for tenderers?",
    "evaluation_criteria": "How will tenders be evaluated? List the criteria and weightings.",
    "key_risks": "What are the key risks identified in the tender document?",
    "boq_total": "What is the total Bills of Quantities amount?",
    "location": "Where is the project located?",
    "tunnel_type": "What type of tunnelling or underground works are involved?",
}


def extract_tender_intelligence(chain: RetrievalQA) -> dict:
    """
    Run structured extraction queries against the loaded tender document.
    Returns a dictionary of extracted fields.
    """
    results = {}

    print("\n  Running structured extraction queries...")
    for field, question in EXTRACTION_QUESTIONS.items():
        try:
            response = chain.invoke({"query": question})
            answer = response["result"].strip()
            sources = [
                doc.metadata.get("source_file", "unknown")
                for doc in response.get("source_documents", [])
            ]
            results[field] = {
                "value": answer,
                "sources": list(set(sources)),
            }
            print(f"    ✓ {field}")
        except Exception as e:
            results[field] = {"value": f"Extraction error: {str(e)}", "sources": []}
            print(f"    ✗ {field} — {e}")

    return results


def ask_question(chain: RetrievalQA, question: str) -> dict:
    """
    Answer a free-form question against the loaded tender documents.
    Returns answer text and source chunk references.
    """
    response = chain.invoke({"query": question})
    answer = response["result"].strip()

    source_chunks = []
    for doc in response.get("source_documents", []):
        source_chunks.append({
            "file": doc.metadata.get("source_file", "unknown"),
            "excerpt": doc.page_content[:200] + "...",
        })

    return {
        "question": question,
        "answer": answer,
        "source_chunks": source_chunks,
    }


# ── Pipeline Orchestrator ─────────────────────────────────────────────────────

def ingest_and_index(file_path: str, rebuild: bool = False) -> FAISS:
    """
    Full ingestion pipeline for a single document:
    Load → Chunk → Embed → Store in FAISS
    """
    print(f"\n[Ingestion] {Path(file_path).name}")
    documents = load_document(file_path)
    print(f"  Loaded {len(documents)} document page(s)")

    chunks = chunk_documents(documents)

    if rebuild or not Path(FAISS_INDEX_PATH).exists():
        vector_store = build_vector_store(chunks)
    else:
        vector_store = add_to_vector_store(chunks)

    return vector_store


def ingest_directory(directory: str, rebuild: bool = True) -> FAISS:
    """
    Ingest all .txt and .pdf files in a directory into one FAISS index.
    """
    dir_path = Path(directory)
    files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.pdf"))

    if not files:
        raise ValueError(f"No .txt or .pdf files found in {directory}")

    print(f"\n[Batch Ingestion] Found {len(files)} document(s) in {directory}")

    all_chunks = []
    for i, file in enumerate(files):
        print(f"\n  [{i+1}/{len(files)}] {file.name}")
        documents = load_document(str(file))
        chunks = chunk_documents(documents)
        all_chunks.extend(chunks)

    print(f"\n  Total chunks to embed: {len(all_chunks)}")
    vector_store = build_vector_store(all_chunks)

    return vector_store
