from fastapi import Depends
from app.db.help_vector_db import PineconeDatabase
# from app.db.graph_db import Neo4jDatabase

pinecone_db = PineconeDatabase()
# neo4j_db = Neo4jDatabase()



def get_pinecone_index():
    """Dependency to provide Pinecone index for FAQ agent."""
    return pinecone_db.get_session()

# def get_neo4j_session():
#     """Dependency to provide Neo4j session for navigation agent."""
#     return neo4j_db.get_session()