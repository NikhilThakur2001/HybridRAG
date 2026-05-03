# HybridRAG — Bromad Customer Complaint Classification System

AI-powered customer care assistant for **Bromad** (eSIM delivery service). Classifies incoming customer complaints using a hybrid RAG pipeline, surfacing similar historical cases and suggested resolutions for customer care agents.

## What It Does

Customer submits a complaint → agent dashboard calls `/classify` → system returns:
- Complaint category (one of 8 Bromad-specific labels)
- Confidence score + reasoning
- Top similar historical complaints from corpus
- Suggested resolution for the agent

## Tech Stack

| Component | Technology |
|-----------|-----------|
| API framework | FastAPI |
| LLM orchestration | LangChain |
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector store | ChromaDB (local persistence) |
| Metadata store | Supabase (PostgreSQL) |
| Keyword retrieval | scikit-learn TF-IDF |
| Output schema | Pydantic v2 |
| Evaluation | RAGAS |
| Observability | LangSmith |

## Architecture

### Ingestion Pipeline (run once)
```
FCC Complaint CSV
  └── filter: wireless/mobile category (~300-500K rows)
        └── chunk complaints
              ├── embed (ada-002) ──► ChromaDB
              ├── fit TF-IDF index ──► disk
              └── store metadata ──► Supabase
                  (id, text, fcc_category, bromad_label, state, date)
```

### Query Pipeline (per request)
```
POST /classify {complaint_text}
  ├── embed query ──► ChromaDB semantic search ──► top-K docs
  ├── tokenize ──► TF-IDF keyword search ──► top-K docs
  └── RRF fusion ──► merged ranked context
        └── LangChain LLM call (GPT-4o-mini)
              + Pydantic structured output
              ──► ClassificationResult
```

## Bromad Complaint Categories

| Category | Description |
|----------|-------------|
| `OTP_AUTH_FAILURE` | OTP not received or invalid during activation |
| `ESIM_ACTIVATION_FAILURE` | eSIM profile fails to activate on device |
| `BILLING_CHARGEBACK` | Incorrect charges, disputed transactions |
| `DEVICE_INCOMPATIBILITY` | Device does not support eSIM or Bromad profile |
| `DATA_PLAN_COVERAGE` | Data not working, coverage issues |
| `REFUND_REQUEST` | Customer requesting refund |
| `PROFILE_DOWNLOAD_ERROR` | QR code scan or profile download fails |
| `OTHER` | Does not fit above categories |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Classify complaint, return full ClassificationResult |
| `/ingest` | POST | Trigger background FCC dataset ingestion |
| `/evaluate` | POST | Trigger async RAGAS eval job, returns `{ job_id }` |
| `/evaluate/{job_id}` | GET | Poll job status + metrics when complete |
| `/health` | GET | Check ChromaDB, Supabase, OpenAI connectivity |

## Data Source

**FCC Consumer Complaint Database** — 3.49M rows, filtered to wireless/mobile subset.

Download: [opendata.fcc.gov](https://opendata.fcc.gov/Consumer/Consumer-Complaint-Data/3xyp-aqkj)

Filtering keeps complaints tagged as `Wireless Telephone Service` or `Internet` — reducing to ~300-500K rows relevant to eSIM/mobile service.

## Learning Objectives

This project is a learning implementation covering:
- Text embedding pipelines at scale (300-500K documents)
- Chunking strategies for complaint text
- Vector database ingestion and retrieval (ChromaDB)
- Hybrid retrieval: semantic (dense) + keyword (sparse) with RRF fusion
- Structured LLM outputs with Pydantic schema validation
- RAGAS evaluation metrics (faithfulness, context recall, answer relevance)
- LangSmith observability (token usage, latency, cost per query)
- Production API design with FastAPI

## Setup

```bash
# Clone and install
git clone https://github.com/your-username/HybridRAG
cd HybridRAG
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, LANGSMITH_API_KEY, SUPABASE_URL, SUPABASE_KEY

# Download FCC data (see docs/data-ingestion.md)

# Run ingestion (one-time, takes ~2-3 hours for 500K rows)
curl -X POST http://localhost:8000/ingest

# Start API
uvicorn app.main:app --reload
```

## Project Structure

```
HybridRAG/
├── app/
│   ├── main.py              # FastAPI app
│   ├── api/
│   │   ├── classify.py      # /classify endpoint
│   │   ├── ingest.py        # /ingest endpoint
│   │   └── evaluate.py      # /evaluate endpoint
│   ├── retrieval/
│   │   ├── chromadb.py      # ChromaDB client + semantic search
│   │   ├── tfidf.py         # TF-IDF index + keyword search
│   │   └── rrf.py           # Reciprocal Rank Fusion
│   ├── ingestion/
│   │   ├── pipeline.py      # FCC data download + filter + chunk
│   │   └── embedder.py      # Batch embedding with OpenAI
│   ├── llm/
│   │   ├── chain.py         # LangChain classification chain
│   │   └── schemas.py       # Pydantic output models
│   ├── evaluation/
│   │   └── ragas_runner.py  # RAGAS metrics evaluation
│   └── db/
│       └── supabase.py      # Supabase metadata client
├── data/                    # FCC dataset (gitignored)
├── chroma_db/               # ChromaDB persistence (gitignored)
├── models/                  # Saved TF-IDF index (gitignored)
├── docs/
│   └── superpowers/specs/   # Design documents
├── tests/
├── .env.example
└── requirements.txt
```
