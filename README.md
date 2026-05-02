# Agentic Knowledge Retrieval System

A production-ready, agent-based knowledge retrieval system for chaotic organizational data. Not naive RAG — this uses multi-hop reasoning, query decomposition, and iterative refinement across three storage layers.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query (POST /query)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Agent Orchestrator                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Classifier   │  │  Reasoning   │  │  Tool Executor   │  │
│  │  (query type) │──▶  Engine      │──▶  (dispatch)      │  │
│  │  + entities   │  │  (multi-hop) │  │  + refine loop   │  │
│  └──────────────┘  └──────┬───────┘  └──────────────────┘  │
└────────────────────────────┼────────────────────────────────┘
                             │
      ┌──────────────────────┼──────────────────────┐
      ▼                      ▼                      ▼
┌──────────┐          ┌──────────┐          ┌──────────────┐
│Vector    │          │ BM25     │          │ PostgreSQL   │
│Search    │          │Keyword   │          │ Structured   │
│(Qdrant)  │          │Search    │          │ Queries      │
└──────────┘          └──────────┘          └──────────────┘
   ┌──────┐             ┌──────┐
   │Logs  │             │Docs  │
   │      │             │Runbk │
   └──────┘             └──────┘
```

### Three-Layer Storage

| Layer | Technology | Stores |
|-------|-----------|--------|
| **Structured** | PostgreSQL | users, teams, services, hosts, incidents, timelines |
| **Vector** | Qdrant | document embeddings, log embeddings, enriched with metadata |
| **Raw** | File system | original files organized by type |

### Agent Tools

| Tool | Function | Use Case |
|------|----------|----------|
| `search_vector` | Semantic search (Qdrant) | Conceptual queries, "what" and "why" |
| `search_vector_logs` | Log-specific vector search | Error patterns, system behavior |
| `search_keyword` | BM25 exact-term search | Error codes, hostnames, IPs |
| `query_sql` | SELECT queries on PostgreSQL | Structured lookups, aggregations |
| `get_incidents` | Incident query with filters | Time-range, severity, service analysis |
| `entity_lookup` | Name-based entity search | "Who owns X?", "Find team Y" |

### Query Classification

The agent classifies queries into:

- **factual** — "Who owns payment-gateway?"
- **relational** — "What services does the Platform Team own?"
- **time_based** — "What incidents happened last week?"
- **causal** — "What caused the payment gateway outage?"
- **exploratory** — "Show me all critical incidents"
- **comparative** — "Compare incidents in prod vs staging"
- **host_status** — "Are the hosts healthy?"
- **multi_hop** — "Why were hosts down between X and Y?"

---

## Quick Start

### Prerequisites

- Docker & Docker Compose v2
- 4GB+ RAM (for the embedding model)

### Run

```bash
# Clone and enter
cd knowledge-retrieval

# Start everything — one command
docker compose up --build

# Wait for health checks (30-60s first run, model downloads)
# Seed sample data in another terminal
docker compose exec app python scripts/seed.py
```

### Verify

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Ask a question
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What caused the payment gateway outage?"}'
```

---

## Example Queries

### Factual
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Who owns payment-gateway?"}'
```

### Causal (multi-hop)
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Why were hosts down between 2026-04-10 and 2026-04-16?"}'
```

### Time-based with filters
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me all critical incidents last week",
    "filters": {"severity": "critical"},
    "top_k": 20
  }'
```

### Exploratory
```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Find incidents related to database issues"}'
```

### Response format
```json
{
  "answer": "Based on the retrieved evidence: ...",
  "sources": [
    {
      "title": "Tool: get_incidents",
      "source": "Step 3: retrieve",
      "snippet": "Got 2 results",
      "metadata": {"step": 3, "action": "retrieve"}
    }
  ],
  "reasoning_trace": [
    {"step": 1, "action": "analyze", "tool": null, "input_summary": "...", "output_summary": "..."}
  ],
  "query_type": "causal",
  "latency_ms": 452.3
}
```

---

## API Reference

### `POST /api/v1/query`

Submit a natural language query.

**Request body:**
```json
{
  "query": "string (required, 1-2000 chars)",
  "filters": {"service": "payment-gateway", "severity": "critical"},
  "time_range": ["2026-04-01T00:00:00Z", "2026-04-30T23:59:59Z"],
  "top_k": 10,
  "enable_refinement": true
}
```

### `GET /api/v1/health`

Service health check.

### Interactive docs

OpenAPI at `http://localhost:8000/docs` (Swagger) or `/redoc` (ReDoc).

---

## Development

### Architecture Decisions

**Why no LLM in the agent loop?**  
The agent uses deterministic classification, rule-based decomposition, and tool dispatch. This makes it:
- **Predictable** — same query → same trace
- **Testable** — unit tests verify exact behavior
- **Fast** — no LLM latency per hop
- **Extensible** — plug in an LLM at the synthesizer step when you need it

**Why BM25 + Vector (hybrid)?**  
Vector search handles semantics, BM25 handles precision. The reciprocal rank fusion (RRF) combines them without tuning. Metadata filtering on Qdrant further narrows results.

**Why adaptive chunking?**  
Fixed-size chunks break semantic boundaries. Adaptive chunking splits on headings, paragraphs, and sentences, then merges to respect min/max sizes with overlap.

### Adding a new tool

1. Create `app/tools/my_tool.py` extending `BaseTool`
2. Implement `spec` property and `run()` method
3. Register in `app/tools/__init__.py`
4. The agent automatically discovers it via `ToolRegistry`

```python
from app.tools.base import BaseTool, ToolSpec, ToolRegistry

class MyTool(BaseTool):
    @property
    def spec(self):
        return ToolSpec(name="my_tool", description="...", parameters={...})

    async def run(self, **kwargs):
        # Your logic here
        return {"success": True, "data": [...]}

ToolRegistry.register(MyTool())
```

### Testing

```bash
# Unit tests (no services needed)
pip install pytest pytest-asyncio
pytest

# Integration tests (need docker containers)
docker compose up -d
pytest -m integration
```

### Ingesting custom data

```bash
# Single file
docker compose exec app python scripts/ingest.py /path/to/runbook.md

# Entire directory
docker compose exec app python scripts/ingest.py /path/to/docs/

# With custom collection type (logs vs docs)
# The pipeline auto-detects based on filename patterns
```

---

## Project Structure

```
knowledge-retrieval/
├── app/
│   ├── agents/           # Classifier, reasoning engine, orchestrator
│   │   ├── classifier.py    # Query type detection + entity extraction
│   │   ├── reasoning.py     # Multi-hop decomposition + refinement
│   │   └── orchestrator.py  # End-to-end query pipeline
│   ├── api/
│   │   └── routes.py        # FastAPI endpoints
│   ├── ingestion/        # Parsing, chunking, extraction, embedding
│   │   ├── parser.py        # File format support (txt, json, logs, md, csv)
│   │   ├── chunker.py       # Adaptive boundary-aware chunking
│   │   ├── extractor.py     # Entity + summary extraction
│   │   ├── embedder.py      # Batch embedding + Qdrant upsert
│   │   └── pipeline.py      # Orchestrated ingestion flow
│   ├── models/
│   │   ├── db.py            # SQLAlchemy ORM (12 tables)
│   │   └── schemas.py       # Pydantic API schemas
│   ├── retrieval/
│   │   ├── vector_store.py  # Qdrant client + embedding model
│   │   ├── keyword_store.py # In-memory BM25 index
│   │   └── hybrid.py        # RRF fusion + HybridRetriever
│   ├── tools/            # Agent-callable tools
│   │   ├── base.py          # Abstract tool + registry
│   │   ├── vector_search.py # Semantic search tools
│   │   ├── keyword_search.py# BM25 keyword tool
│   │   ├── sql_query.py     # Safe SQL execution
│   │   ├── incident_query.py# Incident-specific queries
│   │   └── entity_lookup.py # EntityByName lookup
│   ├── config.py            # Environment-based settings
│   └── main.py              # FastAPI app factory
├── infra/
│   └── Dockerfile           # Multi-stage Python build
├── scripts/
│   ├── seed.py              # Populate DB + vectors with sample data
│   └── ingest.py            # CLI for file ingestion
├── data/
│   └── sample/              # Sample organizational data
│       ├── services.json    # 5 services, 7 users, 5 teams
│       ├── incidents.json   # 5 incidents with timelines
│       ├── runbook.md       # Payment gateway runbook
│       └── logs.txt         # System logs from multiple services
├── tests/
│   ├── test_agent.py        # Classifier + reasoning tests
│   ├── test_retrieval.py    # BM25 + hybrid fusion tests
│   └── test_ingestion.py    # Parser, chunker, extractor tests
├── docker-compose.yml       # app + postgres + qdrant (+ optional redis)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Production Considerations

### Performance
- **Embedding cache**: The `sentence-transformers` model is cached in Docker volume (`model_cache`)
- **Connection pooling**: SQLAlchemy + asyncpg handle DB pool management
- **Batch embeddings**: EmbeddingProcessor batches upserts by 64 documents
- **BM25 persistence**: Rebuilt from DB on startup (consider Redis for real-time)

### Scaling
- **Qdrant**: Can be clustered for high availability
- **PostgreSQL**: pgvector is available as extension if you want to move vectors to PG
- **Workers**: Use `gunicorn -k uvicorn.workers.UvicornWorker` for multi-worker

### Security
- SQL queries are restricted to SELECT/WITH only
- No arbitrary code execution in tools
- CORS configured open — restrict in production

### Monitoring
- `/api/v1/health` endpoint with service dependency checks
- Structured logging with `appname` context
- Latency tracking per query (`latency_ms` in response)

---

## Comparison: Agent System vs Naive RAG

| Aspect | Naive RAG | This Agent System |
|--------|-----------|-------------------|
| Retrieval | Single-shot vector search | Multi-hop, iterative refinement |
| Query handling | "Chunk → embed → search → generate" | Classify → decompose → multi-tool → cross-reference → synthesize |
| Structured data | Ignored or re-chunked | Native SQL queries against normalized schemas |
| Time awareness | None (requires filtering hack) | Native time-range support across all tools |
| Reasoning | None (all in generation) | Explicit reasoning trace, self-ask pattern |
| Explainability | Black box | Full reasoning_trace with per-step details |
| Entity linking | None | EntityLookup tool cross-references across layers |
| Failure mode | Hallucination in generation | Predictable tool execution trace |
