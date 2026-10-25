"""Vector database clients, one knowledge base each, behind a provider switch.

    VECTOR_DB_PROVIDER=chroma    local, on disk (agent/db/data/chroma), no keys needed
    VECTOR_DB_PROVIDER=pinecone  cloud index + Cohere embeddings (needs API keys)

Agents never construct a client directly; they call ``get_vector_db("disease")`` or
``get_vector_db("help")`` and get back something with the ``VectorDatabase`` interface.
"""
from typing import Literal

from agent.config import settings
from agent.db.vector.base_vector_db import VectorDatabase

KnowledgeBase = Literal["disease", "help"]


def get_vector_db(kb: KnowledgeBase) -> VectorDatabase:
    provider = settings.VECTOR_DB_PROVIDER.lower()

    if provider == "chroma":
        from agent.db.vector.chroma.chroma_vector_db import ChromaDatabase

        name = {
            "disease": settings.CHROMA_DISEASE_COLLECTION,
            "help": settings.CHROMA_HELP_COLLECTION,
        }[kb]
        return ChromaDatabase(collection_name=name)

    if provider == "pinecone":
        if kb == "disease":
            from agent.db.vector.pinecone.disease_vector_db import PineconeDatabase
        else:
            from agent.db.vector.pinecone.help_vector_db import PineconeDatabase
        return PineconeDatabase()

    raise ValueError(f"Unknown VECTOR_DB_PROVIDER '{settings.VECTOR_DB_PROVIDER}' (use 'chroma' or 'pinecone')")


__all__ = ["VectorDatabase", "KnowledgeBase", "get_vector_db"]
