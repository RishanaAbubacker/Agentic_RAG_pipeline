"""Local Chroma store. Same interface will work when you move to a cloud
vector DB later — only this file changes, nothing upstream does."""
import chromadb
from chromadb.utils import embedding_functions
import config


def _get_embedding_fn():
    if config.EMBEDDING_BACKEND == "openai":
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=config.OPENAI_API_KEY,
            api_base=config.OPENAI_BASE_URL,
            model_name=config.OPENAI_EMBEDDING_MODEL,
        )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.LOCAL_EMBEDDING_MODEL
    )


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            embedding_function=_get_embedding_fn(),
        )

    def is_empty(self) -> bool:
        return self.collection.count() == 0

    def index(self, docs: list[dict], batch_size: int = 500):
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            self.collection.upsert(
                ids=[d["id"] for d in batch],
                documents=[d["text"] for d in batch],
                metadatas=[d["metadata"] for d in batch],
            )

    def search(self, query: str, k: int = config.TOP_K, where: dict | None = None) -> list[dict]:
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where if where else None,
        )
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({"text": doc, "metadata": meta, "distance": dist})
        return out
