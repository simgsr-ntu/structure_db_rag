from langchain_core.tools import tool
from src.storage.chroma_store import SermonVectorStore


def make_vector_tool(vector_store: SermonVectorStore):

    @tool
    def search_sermons_tool(query: str, year: int | None = None, speaker: str | None = None) -> str:
        """Searches sermon text and summaries using semantic similarity.
        Use for 'What did the pastor say about X?' or 'Find sermons about Y'.
        Optionally filter by year (integer e.g. 2024) or speaker (partial name e.g. 'Chua').
        Returns excerpts with topic, speaker, date, and key verse."""

        where: dict | None = None
        if year is not None and speaker:
            where = {"$and": [{"year": {"$eq": year}}, {"speaker": {"$eq": speaker}}]}
        elif year is not None:
            where = {"year": {"$eq": year}}
        elif speaker:
            where = {"speaker": {"$eq": speaker}}

        results = vector_store.search_sermons(query, k=5, where=where)
        if not results:
            return "No relevant sermon content found."

        parts = []
        for res in results:
            m = res.get("metadata") or {}
            header = (
                f"[{m.get('topic') or 'Unknown Topic'} | {m.get('speaker') or 'Unknown'} "
                f"| {m.get('date') or ''} | {m.get('key_verse') or ''}]"
            )
            parts.append(f"{header}\n{res['content']}")

        return "\n\n---\n\n".join(parts)

    return search_sermons_tool
