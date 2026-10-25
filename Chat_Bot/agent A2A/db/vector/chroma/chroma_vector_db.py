"""Chroma vector database persisted on the local file system.

No API keys or network needed. Data lives in ``agent/db/data/chroma/`` (gitignored) and is
(re)built from the JSON knowledge files with::

    python -m agent.db.vector.chroma.ingest_chroma

Embeddings use Chroma's default local model (all-MiniLM-L6-v2 via ONNX). The model
(~80 MB) is downloaded once on first use and cached by Chroma.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.db.vector.base_vector_db import VectorDatabase

CHROMA_PATH = Path(__file__).resolve().parents[2] / "data" / "chroma"


class ChromaDatabase(VectorDatabase):
    """One Chroma collection (= one knowledge base) on disk."""

    def __init__(self, collection_name: str, path: Path = CHROMA_PATH):
        self.collection_name = collection_name
        self.path = path
        self.client = None
        self.collection = None

    # ---- BaseDatabase ---------------------------------------------------
    def connect(self) -> None:
        import chromadb  # local import: keeps startup fast when Pinecone is selected

        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Chroma collection '{self.collection_name}' ready "
              f"({self.collection.count()} records) at {self.path}")

    def close(self) -> None:
        self.client = None
        self.collection = None

    def get_session(self):
        if self.collection is None:
            self.connect()
        return self.collection

    # ---- VectorDatabase ---------------------------------------------------
    async def query(self, query: str, top_k: int = 5, include_metadata: bool = True) -> Dict[str, Any]:
        try:
            collection = self.get_session()
            count = collection.count()
            if count == 0:
                return {"matches": [], "error": f"Chroma collection '{self.collection_name}' is empty. "
                                                 "Run: python -m agent.db.vector.chroma.ingest_chroma"}
            result = collection.query(
                query_texts=[query],
                n_results=min(top_k, count),
                include=["metadatas", "documents", "distances"],
            )
            matches = []
            for i, id_ in enumerate(result["ids"][0]):
                metadata = dict(result["metadatas"][0][i] or {}) if include_metadata else {}
                if include_metadata and "text" not in metadata:
                    metadata["text"] = result["documents"][0][i]
                matches.append({
                    "id": id_,
                    "score": 1.0 - float(result["distances"][0][i]),  # cosine distance -> similarity
                    "metadata": metadata,
                })
            return {"matches": matches}
        except Exception as e:
            print(f"Error querying Chroma: {str(e)}")
            return {"matches": [], "error": str(e)}

    # ---- ingestion ----------------------------------------------------------
    def upsert(self, records: List[Dict[str, Any]], text_field: str = "text") -> int:
        """Insert or update records. Each record needs an ``id`` and the text field;
        every other key becomes metadata (None values are dropped, Chroma rejects them)."""
        collection = self.get_session()
        ids, documents, metadatas = [], [], []
        for r in records:
            ids.append(str(r["id"]))
            documents.append(str(r[text_field]))
            metadatas.append({k: v for k, v in r.items() if k != "id" and v is not None})
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(ids)

    def clear(self) -> None:
        """Drop and recreate the collection."""
        self.get_session()  # make sure the client exists
        self.client.delete_collection(self.collection_name)
        self.collection = None
        self.connect()
