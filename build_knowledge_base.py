"""
build_knowledge_base.py
------------------------
Reads training_tickets.csv, embeds every ticket using the
sentence-transformer model, and stores them in ChromaDB.

Run this ONCE after getting your 1-year CSV export from ServiceNow.
Run again with --rebuild if you get a new CSV export.

Usage:
    python build_knowledge_base.py
    python build_knowledge_base.py --rebuild
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from agents.embedding_agent      import EmbeddingAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true",
                        help="Wipe existing knowledge base and rebuild from scratch")
    args = parser.parse_args()

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    csv_path    = config["data"]["csv_path"]
    db_path     = config["vector_db"]["path"]
    collection  = config["vector_db"]["collection"]
    embed_model = config["embedding"]["model"]

    print("")
    print("=" * 60)
    print("  ARIA -- Building Knowledge Base")
    print("=" * 60)
    print("")
    print("  CSV path       : " + csv_path)
    print("  Vector DB path : " + db_path)
    print("  Embed model    : " + embed_model)
    print("")

    if not os.path.exists(csv_path):
        print("  [ERROR] CSV not found: " + csv_path)
        print("")
        print("  CSV must have these columns:")
        print("    Short Description  - ticket subject")
        print("    Description        - ticket body")
        print("    Assignment Team    - which team resolved it")
        print("")
        sys.exit(1)

    # Load embedding model
    embed_agent = EmbeddingAgent(model_name=embed_model)
    embed_agent.load()

    # Build knowledge base
    kb_agent = KnowledgeBaseAgent(db_path=db_path, collection_name=collection)
    total    = kb_agent.build(
        csv_path       = csv_path,
        embedding_agent = embed_agent,
        force_rebuild  = args.rebuild,
    )

    print("")
    print("  [OK] Knowledge base ready with " + str(total) + " tickets.")
    print("")
    print("  Next:")
    print("    python predict.py")
    print("")


if __name__ == "__main__":
    main()
