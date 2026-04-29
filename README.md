# BBTC Sermon Intelligence

A **Hybrid Agentic RAG pipeline** for the Bethesda Bedok-Tampines Church (BBTC) sermon archive. Combines structured SQL metadata querying with semantic vector search, backed by a LangGraph ReAct agent that intelligently routes each question to the right data source.

Built as a capstone project demonstrating end-to-end ML engineering: web scraping, document classification, LLM-powered metadata extraction, dual-layer storage, agent orchestration, and a production-ready chat interface.

---

## Features

- **Agentic RAG** — LangGraph ReAct agent dynamically routes queries between SQL, vector search, visualisation, and Bible lookup tools
- **Hybrid storage** — SQLite for structured metadata (speaker, date, topic, verse) + ChromaDB for semantic content search
- **CrossEncoder reranking** — improves retrieval quality by reranking top-20 candidates before returning results
- **Bible archive** — 5 translations (KJV, ASV, YLT, NIV, ESV) indexed for verse lookup and semantic search
- **Dynamic visualisations** — interactive Plotly charts generated on demand from live SQLite data
- **Automated ingestion** — Dagster pipeline scrapes and indexes new sermons on a weekly schedule

---

## Architecture

```
BBTC Website → BBTCScraper (classify-before-download, skips handouts)
    ↓
data/staging/  (NG + PS PDF files)
    ↓
ingest.py
  ├── CLASSIFY  (file_classifier.py)  → ng | ps | handout
  ├── GROUP     (sermon_grouper.py)   → SermonGroup(ng, ps[])
  ├── EXTRACT   (ng_extractor.py)     → TOPIC/SPEAKER/THEME/DATE via regex
  │             (ps_extractor.py)     → key verse from PS filename
  ├── SUMMARIZE (llama3.1:8b)         → unified NG+PS summary
  └── EMBED     (chroma_store.py)     → BGE-M3 → sermon_collection
    ↓
SQLite (data/sermons.db)              ← structured metadata + verses
ChromaDB (data/chroma_db/)            ← body chunks + summaries + bible verses
    ↓
LangGraph ReAct Agent (5 tools)
    ↓
Gradio Chat UI
```

### Key Components

| Component | File | Purpose |
|---|---|---|
| `SermonRegistry` | `src/storage/sqlite_store.py` | SQLite CRUD; sermons, verses, and reference tables |
| `SermonVectorStore` | `src/storage/chroma_store.py` | ChromaDB with BGE-M3 embeddings + CrossEncoder reranker |
| `BBTCScraper` | `src/scraper/bbtc_scraper.py` | Cloudflare-bypass scraper; classify-before-download |
| `classify_file` | `src/ingestion/file_classifier.py` | Returns `ng` \| `ps` \| `handout` |
| `group_sermon_files` | `src/ingestion/sermon_grouper.py` | Pairs NG+PS files by date proximity and topic overlap |
| `extract_ng_metadata` | `src/ingestion/ng_extractor.py` | Regex on labeled PDF fields; filename fallback |
| `parse_verses_from_filename` | `src/ingestion/ps_extractor.py` | Verse regex on PS filenames |
| `normalize_book` | `src/storage/normalize_book.py` | Canonical 66-book name normalization |
| `ingest_bible` | `src/ingestion/bible/bible_ingest.py` | Fetches Scrollmapper JSON + parses EPUBs → `bible_collection` |
| `BibleEpubParser` | `src/ingestion/bible/epub_parser.py` | Extracts verse-by-verse text from EPUB files |
| `run_pipeline` | `ingest.py` | Orchestrates full classify → group → extract → embed |
| `dagster_pipeline.py` | root | Weekly Saturday schedule wrapping `ingest.py` |
| `app.py` | root | Gradio UI + LangGraph ReAct agent |

### Agent Tools

| Tool | Purpose |
|---|---|
| `sql_query_tool` | SQL against `data/sermons.db`; counts, stats, verse aggregations |
| `search_sermons_tool` | BGE-M3 semantic search over `sermon_collection` |
| `viz_tool` | Interactive Plotly charts: `sermons_per_speaker`, `sermons_per_year`, `verses_per_book`, `sermons_scatter` |
| `get_bible_versions_tool` | Returns all stored translations of a specific verse reference |
| `search_bible_tool` | Semantic search over `bible_collection` for topic-based passage lookup |

---

## Local Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with the following models:

```bash
ollama pull bge-m3          # embeddings (1.2 GB, multilingual)
ollama pull llama3.1:8b     # metadata extraction + summary generation
```

### Install and run

```bash
git clone <repo-url>
cd structure_db_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optional: add GROQ_API_KEY for cloud inference

# Run the Gradio UI (starts Ollama automatically if not running)
python app.py
```

Open [http://localhost:7860](http://localhost:7860).

### Bible archive setup

The bible archive supports 5 translations. KJV, ASV, and YLT are downloaded automatically from Scrollmapper (public domain). NIV and ESV require EPUB files that you must supply yourself (copyrighted — not included in this repo).

Place your owned copies here before running the bible ingest:

```
data/bibles/NIV.epub
data/bibles/ESV The Holy Bible.epub
```

Then ingest:

```bash
# All 5 translations (NIV/ESV skipped automatically if files are absent)
python -m src.ingestion.bible.bible_ingest

# Public-domain only (no EPUB files needed)
python -m src.ingestion.bible.bible_ingest --versions KJV ASV YLT
```

The `get_bible_versions_tool` in the chat UI will only return translations that have been ingested.

---

## Data Pipeline

```bash
# Full sermon ingest from scratch (wipe + rebuild)
python ingest.py --wipe

# Incremental ingest (new files only)
python ingest.py

# Ingest a specific year
python ingest.py --year 2024

# Scrape a single year from the BBTC website
python src/scraper/bbtc_scraper.py 2024

# Dagster web UI (weekly scheduler)
DAGSTER_HOME=$(mktemp -d) dagster dev -m dagster_pipeline
```

---

## Database Schema

### SQLite (`data/sermons.db`)

```sql
sermons(
  sermon_id  TEXT PRIMARY KEY,  -- "2024-01-06-the-heart-of-discipleship"
  date       TEXT,              -- YYYY-MM-DD
  year       INTEGER,
  language   TEXT,              -- "English" | "Mandarin"
  speaker    TEXT,
  topic      TEXT,
  theme      TEXT,
  summary    TEXT,              -- LLM-generated from NG+PS
  key_verse  TEXT,              -- first verse from PS filename
  ng_file    TEXT,
  ps_file    TEXT,
  status     TEXT               -- grouped → extracted → indexed | failed
)

verses(
  id           INTEGER PRIMARY KEY,
  sermon_id    TEXT,            -- FK → sermons
  verse_ref    TEXT,            -- "Luke 9:23"
  book         TEXT,            -- "Luke"
  chapter      INTEGER,
  verse_start  INTEGER,
  verse_end    INTEGER,
  is_key_verse INTEGER          -- 1 = key verse
)

bible_books(
  book_name  TEXT PRIMARY KEY,  -- canonical name (e.g. "1 Samuel")
  testament  TEXT,              -- "OT" | "NT"
  book_order INTEGER            -- 1–66
)

book_aliases(
  alias     TEXT PRIMARY KEY,   -- lowercase variant (e.g. "1sam", "gen")
  canonical TEXT                -- FK → bible_books
)

bible_versions(
  version_id   TEXT PRIMARY KEY, -- "KJV", "NIV", etc.
  filename     TEXT,
  status       TEXT,             -- "indexed"
  date_indexed TEXT
)
```

### ChromaDB (`data/chroma_db/`)

**`sermon_collection`**
- Chunks: NG body text (800 tokens / 150 overlap) + LLM summary (single chunk) per sermon
- Metadata: `{sermon_id, doc_type, speaker, date, year, topic, theme, language, key_verse}`
- Embeddings: BGE-M3 via Ollama

**`bible_collection`**
- ~102,790 chunks across 5 translations (~31,000 verses each)
- Sources: KJV, ASV, YLT from Scrollmapper (public domain); NIV, ESV from local EPUBs
- Metadata: `{book, chapter, verse, version, reference}`
- Embeddings: BGE-M3 via Ollama

---

## Running Tests

```bash
python -m pytest tests/ -v
```

103 tests covering: file classification, filename parsing, metadata extraction, verse normalization, sermon grouping, vector retrieval, UI helpers, and SQLite storage.

---

## Project Structure

```
.
├── app.py                        # Gradio UI + LangGraph agent
├── ingest.py                     # Sermon ingestion pipeline
├── dagster_pipeline.py           # Weekly Dagster schedule
├── requirements.txt
├── .env.example
├── src/
│   ├── ingestion/
│   │   ├── bible/
│   │   │   ├── bible_ingest.py   # Bible translation ingestion
│   │   │   └── epub_parser.py    # EPUB verse extractor
│   │   ├── file_classifier.py
│   │   ├── filename_parser.py
│   │   ├── ng_extractor.py
│   │   ├── ps_extractor.py
│   │   └── sermon_grouper.py
│   ├── scraper/
│   │   └── bbtc_scraper.py
│   ├── storage/
│   │   ├── chroma_store.py
│   │   ├── normalize_book.py
│   │   ├── normalize_speaker.py
│   │   ├── reranker.py
│   │   └── sqlite_store.py
│   ├── tools/
│   │   ├── bible_tool.py
│   │   ├── sql_tool.py
│   │   ├── vector_tool.py
│   │   └── viz_tool.py
│   ├── llm.py
│   └── ui_helpers.py
├── tests/                        # 103 unit tests
├── scripts/
│   └── normalize_books.py        # One-time book-name migration utility
└── docs/
    ├── design/                   # Architecture and feature design specs
    └── plans/                    # Implementation plans
```

---

## Deployment

The Gradio interface runs on port `7860`. To run with Docker:

```bash
docker build -t sermon-intelligence .
docker run -p 7860:7860 -v $(pwd)/data:/app/data sermon-intelligence
```

Mount `data/` to a persistent volume to preserve the SQLite database and ChromaDB vector store across restarts.
