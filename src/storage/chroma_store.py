# src/storage/chroma_store.py
import chromadb
from langchain_ollama import OllamaEmbeddings
from src.storage.reranker import Reranker

class SermonVectorStore:
    def __init__(self, persist_dir: str = "data/chroma_db", embeddings=None):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embeddings = embeddings
        
        # If no embeddings provided, try Ollama, otherwise use Chroma default
        if self._embeddings is None:
            try:
                from langchain_ollama import OllamaEmbeddings
                self._embeddings = OllamaEmbeddings(model="nomic-embed-text")
                # Quick test
                self._embeddings.embed_query("test")
            except Exception:
                print("⚠️  Ollama not available. Using ChromaDB default embeddings (local).")
                self._embeddings = None # Chroma will use its default
        
        self._sermons = self._client.get_or_create_collection("sermon_collection")
        self._bible = self._client.get_or_create_collection("bible_collection")
        self._reranker = Reranker()

    def _embed(self, texts: list[str]) -> list[list[float]] | None:
        if self._embeddings:
            return self._embeddings.embed_documents(texts)
        return None # Let Chroma handle it

    _MAX_BATCH = 500

    def _upsert_in_batches(self, collection, chunks: list[str], metadatas: list[dict], ids: list[str]):
        for start in range(0, len(chunks), self._MAX_BATCH):
            end = start + self._MAX_BATCH
            batch_chunks = chunks[start:end]
            batch_embeddings = self._embed(batch_chunks)
            
            kwargs = {
                "documents": batch_chunks,
                "metadatas": metadatas[start:end],
                "ids": ids[start:end],
            }
            if batch_embeddings:
                kwargs["embeddings"] = batch_embeddings
            
            collection.upsert(**kwargs)

    def upsert_sermon_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
        self._upsert_in_batches(self._sermons, chunks, metadatas, ids)

    def upsert_bible_chunks(self, chunks: list[str], metadatas: list[dict], ids: list[str]):
        self._upsert_in_batches(self._bible, chunks, metadatas, ids)

    def _search(self, collection, query: str, k: int, where: dict | None) -> list[dict]:
        n = collection.count()
        if n == 0:
            return []
        
        k_fetch = min(max(k * 3, 12), n)
        kwargs = {
            "n_results": k_fetch,
            "include": ["documents", "metadatas", "distances"],
        }
        
        if self._embeddings:
            kwargs["query_embeddings"] = [self._embeddings.embed_query(query)]
        else:
            kwargs["query_texts"] = [query]
            
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        candidates = [
            {"content": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]
        return self._reranker.rerank(query, candidates, top_k=k)

    def search_sermons(self, query: str, k: int = 4, where: dict | None = None) -> list[dict]:
        return self._search(self._sermons, query, k, where)

    def search_bible(self, query: str, k: int = 4, where: dict | None = None) -> list[dict]:
        return self._search(self._bible, query, k, where)

    def counts(self) -> dict:
        return {
            "sermon_collection": self._sermons.count(),
            "bible_collection": self._bible.count(),
        }
