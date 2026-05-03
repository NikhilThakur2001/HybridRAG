# HybridRAG — Payment Complaint Classification System

AI-powered customer care assistant for payment companies (Razorpay, Stripe, payment gateways). Classifies incoming customer complaints using a hybrid RAG pipeline, surfacing similar historical cases and suggested resolutions for customer care agents.

## What It Does

Customer submits a complaint → agent dashboard calls `/classify` → system returns:
- Complaint category (one of 8 payment-specific labels)
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
cfpb_telecom.parquet (527K rows, payment complaints)
  └── chunk narratives
        ├── embed (ada-002) ──► ChromaDB
        ├── fit TF-IDF index ──► disk (models/tfidf.pkl)
        └── store metadata ──► Supabase
            (id, text, product, issue, state, date, chunk_index)
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

## Payment Complaint Categories

| Category | Description |
|----------|-------------|
| `PAYMENT_FAILED` | Transaction declined or timed out |
| `REFUND_DELAYED` | Refund initiated but not received |
| `UNAUTHORIZED_TRANSACTION` | Fraud or unknown charge |
| `CHARGEBACK_DISPUTE` | Customer raised chargeback |
| `OTP_FAILURE` | OTP not received during payment 2FA |
| `ACCOUNT_BLOCKED` | Payment account suspended or frozen |
| `SETTLEMENT_DELAY` | Merchant settlement not credited |
| `WRONG_DEDUCTION` | Incorrect amount charged |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/classify` | POST | Classify complaint, return full ClassificationResult |
| `/ingest` | POST | Trigger background CFPB dataset ingestion |
| `/evaluate` | POST | Trigger async RAGAS eval job, returns `{ job_id }` |
| `/evaluate/{job_id}` | GET | Poll job status + metrics when complete |
| `/health` | GET | Check ChromaDB, Supabase, OpenAI connectivity |

## Data Sources

### CFPB Consumer Complaint Database
- **527,035 rows** of real payment complaint narratives (after filtering)
- Products: Credit card, Money transfer, Prepaid card, Checking/savings account
- Download: [consumerfinance.gov](https://www.consumerfinance.gov/data-research/consumer-complaints/#get-the-data)
- Direct zip: `https://files.consumerfinance.gov/ccdb/complaints.csv.zip`
- Filter script: `extract_parquet.py` → outputs `cfpb_telecom.parquet`

### Synthetic Labeled Corpus
- 8,000 GPT-4o-mini generated complaints (1,000 per category)
- Labels correct by construction
- 100 per category reserved as RAGAS test set
- Generation script: `app/ingestion/synthetic.py`

## Learning Objectives

- Text embedding pipelines at scale (527K documents)
- Chunking strategies for complaint text
- Vector database ingestion and retrieval (ChromaDB)
- Hybrid retrieval: semantic (dense) + keyword (sparse) with RRF fusion
- Structured LLM outputs with Pydantic schema validation
- RAGAS evaluation metrics (faithfulness, context recall, answer relevance)
- LangSmith observability (token usage, latency, cost per query)
- Production API design with FastAPI
- Async background job pattern (RAGAS eval endpoint)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Fill in: OPENAI_API_KEY, LANGSMITH_API_KEY, SUPABASE_URL, SUPABASE_KEY

# Download + filter CFPB data (one-time)
python extract_parquet.py

# Generate synthetic labeled data (one-time)
python -m app.ingestion.synthetic

# Run ingestion pipeline (one-time, ~2-3 hours for 527K rows)
curl -X POST http://localhost:8000/ingest

# Start API
uvicorn app.main:app --reload
```

## Project Structure

```
HybridRAG/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── classify.py
│   │   ├── ingest.py
│   │   └── evaluate.py
│   ├── retrieval/
│   │   ├── chromadb.py
│   │   ├── tfidf.py
│   │   └── rrf.py
│   ├── ingestion/
│   │   ├── pipeline.py
│   │   ├── embedder.py
│   │   └── synthetic.py
│   ├── llm/
│   │   ├── chain.py
│   │   └── schemas.py
│   ├── evaluation/
│   │   └── ragas_runner.py
│   └── db/
│       └── supabase.py
├── data/                    # gitignored
├── chroma_db/               # gitignored
├── models/                  # gitignored
├── extract_parquet.py
├── explore_parquet.py
├── .env.example
├── requirements.txt
├── CLAUDE.md
└── README.md
```
