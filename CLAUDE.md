# CLAUDE.md — HybridRAG Project Context

## Project

**Bromad HybridRAG** — Customer complaint classification system for Bromad, an eSIM delivery service.
Primary goal: **learning project** covering RAG, embeddings, hybrid retrieval, vector DBs, LangSmith, RAGAS.
Secondary goal: production-ready FastAPI backend for Bromad's customer care agent dashboard.

## Use Case

Customer submits complaint via Bromad UI → routed to customer care agent dashboard → agent calls `/classify` → AI returns classification + similar complaints + suggested resolution → agent resolves faster.

## Architecture Decisions (final)

| Decision | Choice | Reason |
|----------|--------|--------|
| Vector store | ChromaDB (local) | Free, handles ~3GB of 500K embeddings |
| Metadata store | Supabase (PostgreSQL) | Cloud DB for complaint metadata + Bromad labels |
| Hybrid retrieval | ChromaDB semantic + sklearn TF-IDF + RRF | Custom implementation for learning |
| LLM | GPT-4o-mini via LangChain | Cost-efficient, LangSmith-compatible |
| Embeddings | OpenAI text-embedding-ada-002 | 1536 dims, standard |
| Output schema | Pydantic v2 | Deterministic structured output |
| API | FastAPI | Production-ready, async |
| Evaluation | RAGAS | faithfulness, context recall, answer relevance |
| Observability | LangSmith | token usage, latency, cost per query |

## Data

- **Source**: FCC Consumer Complaint Database (3.49M rows total)
- **Filter**: Wireless/mobile category → ~300-500K rows
- **Approach C**: FCC data = retrieval corpus; Bromad taxonomy = classification targets
- **Embedding cost**: ~$3-4 for 300-500K rows at ada-002 rates

## Bromad Complaint Categories (classification targets)

```python
class BromadCategory(str, Enum):
    OTP_AUTH_FAILURE = "OTP_AUTH_FAILURE"
    ESIM_ACTIVATION_FAILURE = "ESIM_ACTIVATION_FAILURE"
    BILLING_CHARGEBACK = "BILLING_CHARGEBACK"
    DEVICE_INCOMPATIBILITY = "DEVICE_INCOMPATIBILITY"
    DATA_PLAN_COVERAGE = "DATA_PLAN_COVERAGE"
    REFUND_REQUEST = "REFUND_REQUEST"
    PROFILE_DOWNLOAD_ERROR = "PROFILE_DOWNLOAD_ERROR"
    OTHER = "OTHER"
```

## Pydantic Output Schema

```python
class SimilarComplaint(BaseModel):
    id: str
    text: str
    fcc_category: str
    similarity_score: float

class ClassificationResult(BaseModel):
    category: BromadCategory
    confidence: float          # 0.0–1.0
    reasoning: str
    similar_complaints: list[SimilarComplaint]  # top-k from retrieval
    suggested_resolution: str
```

## API Endpoints

- `POST /classify` — main endpoint for agent dashboard
- `POST /ingest` — one-time FCC dataset ingestion trigger
- `GET /evaluate` — RAGAS evaluation suite
- `GET /health` — connectivity check (ChromaDB, Supabase, OpenAI)

## Key Implementation Notes

- Ingestion is one-time/offline; query pipeline is online per request
- TF-IDF index fitted on corpus text, saved to disk (`models/tfidf.pkl`)
- RRF fusion: `score = sum(1 / (k + rank_i))` where k=60 (standard)
- ChromaDB collection name: `bromad_complaints`
- Supabase table: `complaints` with columns `(id, text, fcc_category, bromad_label, state, date, chunk_index)`
- Batch embed in chunks of 100 to stay within OpenAI rate limits
- LangSmith project name: `bromad-hybridrag`

## Environment Variables Required

```
OPENAI_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=bromad-hybridrag
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
├── ingestion/     pipeline.py, embedder.py
├── llm/           chain.py, schemas.py
├── evaluation/    ragas_runner.py
└── db/            supabase.py
```

## What NOT to do

- Do not use LangChain's EnsembleRetriever — retrieval is implemented manually for learning
- Do not store vectors in Supabase — ChromaDB only for vectors
- Do not use synchronous embedding calls — always batch async
- Do not skip LangSmith tracing on LLM calls
