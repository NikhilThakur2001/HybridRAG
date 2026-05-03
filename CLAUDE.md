# CLAUDE.md — HybridRAG Project Context

## Project

**HybridRAG** — Customer complaint classification system for payment companies (Razorpay, Stripe, payment gateways).
Primary goal: **learning project** covering RAG, embeddings, hybrid retrieval, vector DBs, LangSmith, RAGAS.
Secondary goal: production-ready FastAPI backend for a customer care agent dashboard — agent sees incoming complaint, AI returns classification + similar cases + suggested resolution, agent resolves faster.

## Domain Pivot (from eSIM telecom → payments)

Originally scoped for Bromad (eSIM). Pivoted to payment complaints (Razorpay/Stripe) because:
- Better data availability (CFPB has 527K payment complaint narratives)
- OTP + chargeback categories fit naturally in payments
- Same architecture, same agent dashboard use case

## Architecture Decisions (final)

| Decision | Choice | Reason |
|----------|--------|--------|
| Vector store | ChromaDB (local) | Free, handles 527K embeddings at scale |
| Metadata store | Supabase (PostgreSQL) | Cloud DB for complaint metadata + labels |
| Hybrid retrieval | ChromaDB semantic + sklearn TF-IDF + RRF | Custom implementation for learning |
| LLM | GPT-4o-mini via LangChain | Cost-efficient, LangSmith-compatible |
| Embeddings | OpenAI text-embedding-ada-002 | 1536 dims, standard |
| Output schema | Pydantic v2 | Deterministic structured output |
| API | FastAPI | Production-ready, async |
| Evaluation | RAGAS | faithfulness, context recall, answer relevance |
| Observability | LangSmith | token usage, latency, cost per query |

## Data

- **Source**: CFPB Consumer Complaint Database
- **Filter**: Payment products (credit card, money transfer, prepaid, checking/savings) + narrative not null
- **Size after filter**: 527,035 rows
- **Saved as**: `cfpb_telecom.parquet` (local, gitignored)
- **Column for embedding**: `Consumer complaint narrative`
- **XXXX**: CFPB anonymization of names — expected, harmless for embeddings
- **Labeled corpus**: Synthetic GPT-4o-mini generated (1K × 8 categories = 8K samples)
- **RAGAS test set**: 100 synthetic labeled samples reserved from above

## Payment Complaint Categories (classification targets)

```python
class PaymentCategory(str, Enum):
    PAYMENT_FAILED = "PAYMENT_FAILED"
    REFUND_DELAYED = "REFUND_DELAYED"
    UNAUTHORIZED_TRANSACTION = "UNAUTHORIZED_TRANSACTION"
    CHARGEBACK_DISPUTE = "CHARGEBACK_DISPUTE"
    OTP_FAILURE = "OTP_FAILURE"
    ACCOUNT_BLOCKED = "ACCOUNT_BLOCKED"
    SETTLEMENT_DELAY = "SETTLEMENT_DELAY"
    WRONG_DEDUCTION = "WRONG_DEDUCTION"
```

## Pydantic Output Schema

```python
class SimilarComplaint(BaseModel):
    id: str
    text: str
    product: str
    similarity_score: float

class ClassificationResult(BaseModel):
    category: PaymentCategory
    confidence: float             # 0.0–1.0
    reasoning: str
    similar_complaints: list[SimilarComplaint]  # top-k from retrieval
    suggested_resolution: str
```

## API Endpoints

- `POST /classify` — main endpoint for agent dashboard
- `POST /ingest` — one-time CFPB dataset ingestion trigger (background task)
- `POST /evaluate` — triggers async RAGAS eval job, returns `{ job_id }`
- `GET /evaluate/{job_id}` — poll job status + results when done
- `GET /health` — connectivity check (ChromaDB, Supabase, OpenAI)

### Async Evaluate Job Response
```json
{
  "job_id": "uuid",
  "status": "pending | running | complete | failed",
  "results": {
    "faithfulness": 0.87,
    "context_recall": 0.79,
    "answer_relevance": 0.91,
    "num_samples": 100,
    "duration_seconds": 142
  }
}
```

## Key Implementation Notes

- Ingestion is one-time/offline; query pipeline is online per request
- TF-IDF index fitted on corpus text, saved to disk (`models/tfidf.pkl`)
- RRF fusion: `score = sum(1 / (k + rank_i))` where k=60 (standard)
- ChromaDB collection name: `payment_complaints`
- Supabase tables: `complaints(id, text, product, issue, state, date, chunk_index)`, `eval_jobs(id uuid, status, created_at, completed_at, results jsonb)`
- Batch embed in chunks of 100 to stay within OpenAI rate limits
- LangSmith project name: `hybridrag-payments`
- Chunking: whole complaint = 1 chunk (avg 100-300 words, short enough)

## Environment Variables Required

```
OPENAI_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=hybridrag-payments
SUPABASE_URL=
SUPABASE_KEY=
CHROMA_PERSIST_DIR=./chroma_db
TFIDF_MODEL_PATH=./models/tfidf.pkl
```

## Project Structure

```
app/
├── main.py
├── api/           classify.py, ingest.py, evaluate.py
├── retrieval/     chromadb.py, tfidf.py, rrf.py
├── ingestion/     pipeline.py, embedder.py, synthetic.py
├── llm/           chain.py, schemas.py
├── evaluation/    ragas_runner.py
└── db/            supabase.py
```

## What NOT to do

- Do not use LangChain's EnsembleRetriever — retrieval is implemented manually for learning
- Do not store vectors in Supabase — ChromaDB only for vectors
- Do not use synchronous embedding calls — always batch async
- Do not skip LangSmith tracing on LLM calls
- Do not commit parquet, CSV, chroma_db/, or models/ to git
