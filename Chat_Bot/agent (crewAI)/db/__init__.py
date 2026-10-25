"""Database access for the agent.

  base_db.py   shared abstract interface
  sqlite/      SQLite access (read-only farmer reference data)
  vector/      Pinecone vector DB clients (disease KB, company help KB)
  data/        the data files themselves (.db, .json) - no code
"""
