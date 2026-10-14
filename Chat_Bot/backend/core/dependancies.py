from agent.db.vector import get_vector_db
# from agent.db.graph_db import Neo4jDatabase

help_vector_db = get_vector_db("help")
# neo4j_db = Neo4jDatabase()


def get_help_vector_db():
    """Dependency providing the company-help vector DB (Chroma or Pinecone per settings)."""
    return help_vector_db

# def get_neo4j_session():
#     """Dependency to provide Neo4j session for navigation agent."""
#     return neo4j_db.get_session()
