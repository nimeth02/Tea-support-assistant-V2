"""Build the local Chroma knowledge bases from the JSON files in agent/db/data/knowledge/.

Run from the project root (Chat_Bot):

    python -m agent.db.vector.chroma.ingest_chroma            # upsert all knowledge bases
    python -m agent.db.vector.chroma.ingest_chroma --reset    # drop collections first
    python -m agent.db.vector.chroma.ingest_chroma disease    # only one knowledge base

Each JSON file is a list of records with an "id" and a text field; all other keys are
stored as metadata and come back to the RAG tools unchanged.
"""
import argparse
import json
from pathlib import Path

from agent.config import settings
from agent.db.vector.chroma.chroma_vector_db import ChromaDatabase

KNOWLEDGE_DIR = Path(__file__).resolve().parents[2] / "data" / "knowledge"

# knowledge base -> (json file, chroma collection, field embedded as the document text)
KNOWLEDGE_BASES = {
    "disease": (KNOWLEDGE_DIR / "tea_disease.json", settings.CHROMA_DISEASE_COLLECTION, "text"),
    "help": (KNOWLEDGE_DIR / "company_help.json", settings.CHROMA_HELP_COLLECTION, "text"),
}


def ingest(kb: str, reset: bool = False) -> int:
    json_path, collection, text_field = KNOWLEDGE_BASES[kb]
    records = json.loads(json_path.read_text(encoding="utf-8"))
    db = ChromaDatabase(collection_name=collection)
    if reset:
        db.clear()
    n = db.upsert(records, text_field=text_field)
    print(f"[{kb}] {n} records from {json_path.name} -> collection '{collection}' "
          f"(now {db.get_session().count()} total)")
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kbs", nargs="*", choices=list(KNOWLEDGE_BASES), help="knowledge bases (default: all)")
    parser.add_argument("--reset", action="store_true", help="drop the collection(s) before ingesting")
    args = parser.parse_args()
    for kb in args.kbs or KNOWLEDGE_BASES:
        ingest(kb, reset=args.reset)


if __name__ == "__main__":
    main()
