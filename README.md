# Tender Intelligence System

**LLM-powered document intelligence for tunnelling and civil engineering tenders.**

A production-style RAG (Retrieval-Augmented Generation) pipeline that ingests unstructured tender documents, extracts structured commercial intelligence, and answers free-form questions.

---

## What it does

Upload a tender document. The system automatically extracts:

- Project name, client, contract reference
- Contract value, duration, start date
- Scope of works summary
- Essential qualification requirements
- Evaluation criteria and weightings
- Key risks
- Bills of Quantities total

And answers free-form questions like:
> *"What are the ground conditions for this project?"*
> *"What insurance requirements are specified?"*
> *"Summarise the monitoring and instrumentation requirements."*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                       │
│                                                             │
│  Tender PDF/TXT                                             │
│       │                                                     │
│       ▼                                                     │
│  Document Loader ──────────────────────────────────────┐   │
│  (TextLoader / PyPDFLoader)                            │   │
│       │                                                │   │
│       ▼                                                │   │
│  RecursiveCharacterTextSplitter                        │   │
│  chunk_size=512, overlap=64                            │   │
│       │                                                │   │
│       ▼                                                │   │
│  OpenAI text-embedding-3-small                         │   │
│  → 1,536-dimension vectors                             │   │
│       │                                                │   │
│       ▼                                                │   │
│  FAISS Vector Index (persisted to disk) ───────────────┘   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    QUERY PIPELINE                           │
│                                                             │
│  User Question                                              │
│       │                                                     │
│       ▼                                                     │
│  Embed question → query vector                              │
│       │                                                     │
│       ▼                                                     │
│  FAISS similarity search (k=6)                              │
│  → top 6 most relevant chunks                               │
│       │                                                     │
│       ▼                                                     │
│  Prompt construction:                                       │
│  [System instructions] + [Context chunks] + [Question]      │
│       │                                                     │
│       ▼                                                     │
│  GPT-3.5-turbo (temperature=0)                              │
│       │                                                     │
│       ▼                                                     │
│  Structured answer + source citations                       │
└─────────────────────────────────────────────────────────────┘
```

### Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Chunk size | 512 tokens | Balances context preservation vs retrieval precision |
| Chunk overlap | 64 tokens | Prevents information loss at chunk boundaries |
| k (retrieved chunks) | 6 | Covers multi-part questions without context overflow |
| LLM temperature | 0 | Deterministic outputs — commercial decisions require consistency |
| Embedding model | text-embedding-3-small | Cost-efficient with strong semantic performance |
| Vector store | FAISS | Fast local similarity search, no external service required |

---

## Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-3.5-turbo |
| Embeddings | OpenAI text-embedding-3-small |
| RAG orchestration | LangChain |
| Vector store | FAISS (Facebook AI Similarity Search) |
| API backend | FastAPI |
| Frontend | Streamlit |
| Containerisation | Docker + Docker Compose |

---

## Project structure

```
tender-intelligence/
├── app/
│   ├── rag_pipeline.py      # Core RAG engine: ingestion, chunking, retrieval, LLM
│   ├── api.py               # FastAPI REST backend
│   └── streamlit_app.py     # Interactive Streamlit frontend
├── data/
│   └── sample_tenders/      # Three sample tunnelling tender documents
│       ├── tender_001_crossrail_tunnel.txt
│       ├── tender_002_utility_microtunnel.txt
│       └── tender_003_monitoring_instrumentation.txt
├── tests/
│   └── test_pipeline.py     # Unit and integration tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/girishkundal/tender-intelligence.git
cd tender-intelligence

cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the API backend

```bash
uvicorn app.api:app --reload
# API runs at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 4. Run the Streamlit frontend

```bash
streamlit run app/streamlit_app.py
# UI at http://localhost:8501
```

### 5. Run with Docker Compose (recommended)

```bash
docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check and index status |
| `POST` | `/ingest` | Upload and ingest a tender document |
| `POST` | `/ingest/sample` | Load included sample tender documents |
| `GET` | `/documents` | List all ingested documents |
| `POST` | `/analyse` | Run full structured extraction (14 fields) |
| `POST` | `/ask` | Ask a free-form question |
| `DELETE` | `/index` | Clear FAISS index and uploaded documents |

Interactive API docs: `http://localhost:8000/docs`

### Example: ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the contract value and duration?"}'
```

Response:
```json
{
  "question": "What is the contract value and duration?",
  "answer": "The total contract sum is £34,559,350 (excluding VAT). The contract duration is 22 months, commencing September 2024 with practical completion in June 2026.",
  "source_chunks": [
    {
      "file": "tender_001_crossrail_tunnel.txt",
      "excerpt": "TOTAL CONTRACT SUM (EXCLUDING VAT): £34,559,350..."
    }
  ],
  "model": "gpt-3.5-turbo"
}
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Sample documents

Three realistic tunnelling tender documents are included:

1. **Northern Line Extension — Tunnel Boring Works** · £34.5M · 22 months
   TBM tunnelling in London Clay, precast segmental lining, settlement monitoring

2. **Severn Water — Trunk Main, Microtunnelling** · £4.5M · 14 months
   Five microtunnel drives under roads, railway, and river in Coventry

3. **A303 Stonehenge — Monitoring and Instrumentation** · £1.8M · 36 months
   450 settlement points, real-time SCADA, heritage asset monitoring

---

## Extending to HBT's requirements

This system demonstrates the core tender intelligence capability described in the HBT KTP brief. Natural extensions include:

- **BoQ generation**: structured prompt chains to extract quantities and map to cost databases
- **Opportunity scoring**: ML classifier trained on historical bid/no-bid decisions
- **Multi-document comparison**: compare multiple tenders against HBT's capability profile
- **IoT integration**: extend to ingest sensor data alongside tender documents for the Digital Twin

---

## Author

**Girish Kundal** — MSc Artificial Intelligence (Distinction), Birmingham City University, 2025

[linkedin.com/in/girish-kundal](https://linkedin.com/in/girish-kundal) · [github.com/girishkundal](https://github.com/girishkundal)
