from langchain_core.tools import tool
from src.storage.chroma_store import SermonVectorStore

def make_vector_tool(vector_store: SermonVectorStore):
    @tool
    def search_sermons_tool(query: str) -> str:
        """Searches the sermon text for relevant excerpts based on a semantic query. 
        Use this for 'What did the pastor say about X?' or 'Find sermons about Y'."""
        results = vector_store.search_sermons(query, k=8)
        if not results:
            return "No relevant sermon excerpts found."
        
        context = "\n\n".join([
            f"--- Source: {res['metadata'].get('filename')} ---\n{res['content']}" 
            for res in results
        ])
        return context

    return search_sermons_tool
