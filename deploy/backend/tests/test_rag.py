# deploy/backend/tests/test_rag.py
import os
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture()
def mock_rag(tmp_path):
    """RAGPipeline with mocked heavy dependencies."""
    os.environ["DATA_DIR"] = str(tmp_path)
    os.environ["GROQ_API_KEY"] = "test-key"

    # Create a minimal SQLite DB
    import sqlite3
    db = tmp_path / "sermons.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute("""CREATE TABLE sermons (
            sermon_id TEXT PRIMARY KEY, filename TEXT, url TEXT UNIQUE,
            speaker TEXT, date TEXT, series TEXT, bible_book TEXT,
            primary_verse TEXT, language TEXT, file_type TEXT,
            year INTEGER, status TEXT, date_scraped TEXT
        )""")

    with (
        patch("deploy.backend.api.rag.SentenceTransformer") as mock_st,
        patch("deploy.backend.api.rag.CrossEncoder") as mock_ce,
        patch("deploy.backend.api.rag.chromadb.PersistentClient") as mock_chroma,
        patch("deploy.backend.api.rag.get_llm") as mock_llm,
        patch("deploy.backend.api.rag.create_react_agent") as mock_agent,
    ):
        # Embedder returns a fixed vector
        mock_st.return_value.encode.return_value = [[0.1] * 384]

        # CrossEncoder returns scores
        mock_ce.return_value.predict.return_value = [0.9, 0.7, 0.5]

        # ChromaDB returns 3 results
        mock_col = MagicMock()
        mock_col.count.return_value = 3
        mock_col.query.return_value = {
            "documents": [["chunk 1", "chunk 2", "chunk 3"]],
            "metadatas": [[
                {"filename": "a.pdf", "speaker": "Pastor A", "date": "2022-01-01", "primary_verse": "John 3:16"},
                {"filename": "b.pdf", "speaker": "Pastor B", "date": "2023-01-01", "primary_verse": "Rom 8:28"},
                {"filename": "c.pdf", "speaker": "Pastor A", "date": "2024-01-01", "primary_verse": "John 1:1"},
            ]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_col

        # Agent returns a simple message
        from langchain_core.messages import AIMessage
        mock_agent.return_value.invoke.return_value = {
            "messages": [AIMessage(content="Grace is God's unmerited favour.")]
        }

        from deploy.backend.api.rag import RAGPipeline
        pipeline = RAGPipeline(data_dir=str(tmp_path))
        yield pipeline


def test_query_returns_answer(mock_rag):
    result = mock_rag.query("What is grace?")
    assert "answer" in result
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_query_returns_citations(mock_rag):
    result = mock_rag.query("What is grace?")
    assert "citations" in result
    assert len(result["citations"]) > 0
    first = result["citations"][0]
    assert "filename" in first


def test_query_empty_collection(mock_rag):
    mock_rag._collection.count.return_value = 0
    result = mock_rag.query("What is grace?")
    assert isinstance(result["answer"], str)


def test_query_with_year_filter(mock_rag):
    mock_rag._collection.count.return_value = 3
    mock_rag.query("sermon about faith", year_filter=2023)
    call_kwargs = mock_rag._collection.query.call_args[1]
    assert call_kwargs.get("where") == {"year": {"$eq": 2023}}


def test_query_with_speaker_filter(mock_rag):
    mock_rag._collection.count.return_value = 3
    mock_rag.query("sermon", speaker_filter="Pastor A")
    call_kwargs = mock_rag._collection.query.call_args[1]
    assert call_kwargs.get("where") == {"speaker": {"$eq": "Pastor A"}}
