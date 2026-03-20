"""
build_knowledge_base.py
------------------------
Reads training_tickets.csv, embeds every ticket using the
sentence-transformer model, and stores them in ChromaDB.

Supports incremental loading — run in chunks of 10,000 rows at a time.
Each run ADDS to the existing knowledge base without overwriting it.

Usage:
    python build_knowledge_base.py --start 0     --end 10000
    python build_knowledge_base.py --start 10000 --end 20000
    python build_knowledge_base.py --start 20000 --end 30000
    ... and so on until all rows are loaded

    To wipe everything and start fresh:
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
    parser.add_argument("--start", type=int, default=0,
                        help="Start row index (default: 0)")
    parser.add_argument("--end", type=int, default=10000,
                        help="End row index (default: 10000)")
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
    print("  Row range      : " + str(args.start) + " to " + str(args.end))
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

    # Build / append knowledge base
    kb_agent = KnowledgeBaseAgent(db_path=db_path, collection_name=collection)
    total    = kb_agent.build(
        csv_path        = csv_path,
        embedding_agent = embed_agent,
        force_rebuild   = args.rebuild,
        start           = args.start,
        end             = args.end,
    )

    print("")
    print("  [OK] Knowledge base now has " + str(total) + " tickets total.")
    print("")
    print("  Next:")
    if args.end < 125000:
        print("    Load next batch:")
        print("    python build_knowledge_base.py --start " + str(args.end) +
              " --end " + str(args.end + 10000))
    else:
        print("    All rows loaded. Start predicting:")
        print("    python predict.py")
    print("")


if __name__ == "__main__":

    main()