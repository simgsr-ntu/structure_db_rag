# deploy/backend/api/rag.py
import os
import sqlite3
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from src.llm import get_llm

_SYSTEM_PROMPT = (
    "You are the BBTC Sermon Intelligence Assistant.\n"
    "Answer ONLY from the sermon excerpts provided in the prompt. Never invent facts.\n"
    "For every claim, cite the sermon filename and speaker.\n"
    "Use 'sql_query_tool' for counts/statistics. "
    "Key column names: primary_verse (not verse), speaker, year, bible_book.\n"
    "If the answer is not in the excerpts, say so explicitly."
)


class RAGPipeline:
    def __init__(self, data_dir: str = "data"):
        self._data_dir = data_dir
        model_cache = os.getenv("MODEL_CACHE_DIR", os.path.join(data_dir, "models"))
        os.makedirs(model_cache, exist_ok=True)

        self._embedder = SentenceTransformer(
            "BAAI/bge-small-en-v1.5", cache_folder=model_cache
        )
        self._reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
            cache_folder=model_cache,
        )

        chroma_path = os.path.join(data_dir, "chroma_db")
        self._chroma = chromadb.PersistentClient(path=chroma_path)
        self._collection = self._chroma.get_or_create_collection(
            "sermon_collection", metadata={"hnsw:space": "cosine"}
        )

        self._db_path = os.path.join(data_dir, "sermons.db")

        llm = get_llm(provider_type="groq", temperature=0.1)
        self._agent = create_react_agent(
            llm,
            tools=[self._make_sql_tool()],
            prompt=_SYSTEM_PROMPT,
        )

    def _make_sql_tool(self):
        db_path = self._db_path

        @tool
        def sql_query_tool(query: str) -> str:
            """Runs SQL against the sermons database for stats and counts.
            Schema: sermons(sermon_id, filename, speaker, date, series,
            bible_book, primary_verse, language, year, status)."""
            try:
                with sqlite3.connect(db_path) as conn:
                    cursor = conn.execute(query)
                    cols = [d[0] for d in cursor.description]
                    rows = cursor.fetchall()
                    if not rows:
                        return "No results."
                    result = f"Columns: {', '.join(cols)}\n"
                    result += "\n".join(str(r) for r in rows[:50])
                    return result
            except Exception as e:
                return f"SQL Error: {e}"

        return sql_query_tool

    def _semantic_search(
        self, query: str, year_filter: int | None, speaker_filter: str | None
    ) -> tuple[str, list[dict]]:
        """Returns (formatted_context, citations)."""
        n = self._collection.count()
        if n == 0:
            return "", []

        raw_embedding = self._embedder.encode(query)
        embedding = raw_embedding.tolist() if hasattr(raw_embedding, "tolist") else list(raw_embedding)

        kwargs: dict = {
            "query_embeddings": [embedding],
            "n_results": min(20, n),
            "include": ["documents", "metadatas"],
        }

        where: dict = {}
        if year_filter is not None:
            where["year"] = {"$eq": year_filter}
        if speaker_filter:
            where["speaker"] = {"$eq": speaker_filter}
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        docs = results["documents"][0]
        metas = results["metadatas"][0]

        if not docs:
            return "", []

        # CrossEncoder rerank
        pairs = [[query, doc] for doc in docs]
        scores = self._reranker.predict(pairs)
        ranked = sorted(zip(scores, docs, metas), reverse=True)[:5]

        parts = []
        citations = []
        for _, doc, meta in ranked:
            filename = meta.get("filename", "unknown")
            speaker = meta.get("speaker") or "Unknown"
            date = meta.get("date") or ""
            verse = meta.get("primary_verse") or ""
            parts.append(f"[{filename} | {speaker} | {date}]\n{doc}")
            citations.append(
                {"filename": filename, "speaker": speaker, "date": date, "verse": verse}
            )

        return "\n\n---\n\n".join(parts), citations

    def query(
        self,
        question: str,
        year_filter: int | None = None,
        speaker_filter: str | None = None,
    ) -> dict:
        context, citations = self._semantic_search(question, year_filter, speaker_filter)

        if context:
            augmented = f"Sermon excerpts:\n\n{context}\n\nQuestion: {question}"
        else:
            augmented = question

        result = self._agent.invoke({"messages": [HumanMessage(content=augmented)]})
        answer = result["messages"][-1].content

        return {"answer": answer, "citations": citations}
