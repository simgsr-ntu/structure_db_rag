# BBTC Sermon Intelligence

A **Hybrid Agentic RAG pipeline** for the Bethesda Bedok-Tampines Church (BBTC) sermon archive. It combines structured SQL metadata querying with semantic vector search, backed by a LangGraph ReAct agent that intelligently routes each question to the right data source.

Built as a capstone project demonstrating end-to-end ML engineering: data ingestion, dual-layer storage, LLM-powered metadata extraction, agent orchestration, and a production-ready chat interface.

---

## Features

- **Agentic RAG** — LangGraph ReAct agent dynamically routes queries between SQL, vector search, and chart generation
- **Hybrid storage** — SQLite for structured metadata (speaker, date, series, verse) + ChromaDB for semantic content search
- **CrossEncoder reranking** — improves retrieval quality by reranking top-20 candidates with a cross-encoder model
- **Data visualisations** — bar charts and scatter plots generated on demand from live SQLite data
- **Automated ingestion** — Dagster pipeline scrapes and indexes new sermons weekly

---

## Architecture

```
BBTC Website → BBTCScraper → data/staging/ (raw files)
                           → data/sermons/ (.txt extracts)
                           → SQLite (data/sermons.db) [status: extracted]
                               ↓ MetadataExtractor (Ollama)
                           → SQLite [status: indexed, with speaker/date/verse]
                           → ChromaDB (data/chroma_db/) [vector chunks]
                               ↓
                        Gradio UI → LangGraph ReAct Agent → Tools → Response
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| `SermonRegistry` | `src/storage/sqlite_store.py` | SQLite CRUD; tracks ingestion status |
| `SermonVectorStore` | `src/storage/chroma_store.py` | ChromaDB with Ollama embeddings + CrossEncoder reranker |
| `BBTCScraper` | `src/scraper/bbtc_scraper.py` | Cloudflare-bypass scraper; downloads + text-extracts PDFs/PPTX/DOCX |
| `MetadataExtractor` | `src/ingestion/metadata_extractor.py` | Ollama LLM extracts speaker/date/series/verse from sermon text |
| `get_llm()` | `src/llm.py` | Returns a `ChatOllama` instance |
| `dagster_pipeline.py` | root | Dagster asset — weekly scrape + ingest schedule |
| `app.py` | root | Gradio UI + LangGraph ReAct agent |

### Agent Tools

| Tool | When used |
|---|---|
| `sql_query_tool` | Counts, stats, date lookups, top-N queries |
| `search_sermons_tool` | Semantic content search with optional year/speaker filter |
| `matplotlib_tool` | On-demand charts: `sermons_per_speaker`, `sermons_per_year`, `top_bible_books`, `sermons_scatter` |
| `compare_bible_versions` | Bible translation comparisons |

---

## Local Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with these models pulled:
  ```bash
  ollama pull llama3.1:8b        # chat agent
  ollama pull llama3.2:3b        # metadata extraction
  ollama pull nomic-embed-text   # embeddings
  ```

### Install and run

```bash
git clone <repo-url>
cd structure_db_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://localhost:7860](http://localhost:7860).

### SQLite Schema

```sql
sermons(
  sermon_id TEXT PRIMARY KEY,
  filename TEXT,
  url TEXT UNIQUE,
  speaker TEXT,
  date TEXT,          -- YYYY-MM-DD
  series TEXT,
  bible_book TEXT,
  primary_verse TEXT, -- e.g. "Romans 8:28"
  language TEXT,      -- "English" | "Mandarin"
  file_type TEXT,     -- pdf | pptx | docx
  year INTEGER,
  status TEXT,        -- extracted → indexed | failed
  date_scraped TEXT
)
```

---

## Data Pipeline

```bash
# Full pipeline: scrape + extract + vectorise (via Dagster)
dagster asset materialize --select sermon_ingestion_summary -m dagster_pipeline

# Dagster web UI (to trigger/monitor)
DAGSTER_HOME=$(mktemp -d) dagster dev -m dagster_pipeline

# Vectorise already-extracted sermons without re-scraping
python quick_ingest.py

# Scrape a single year only
python src/scraper/bbtc_scraper.py 2024
```

---

## Deployment

The `Dockerfile` runs the Gradio interface on port `7860`. Deploy on [Render](https://render.com) using `render.yaml`:

```bash
# Build and test locally
docker build -t sermon-intelligence .
docker run -p 7860:7860 -v $(pwd)/data:/app/data sermon-intelligence
```

Mount `data/` to a persistent volume to preserve the SQLite database and ChromaDB vector store across restarts.

---

## Running Tests

```bash
python -m pytest tests/ -v
```
