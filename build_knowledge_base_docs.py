"""
build_knowledge_base_docs.py
-----------------------------
Ingests KB articles and documents (TXT, MD, PDF, HTML) from a directory
into the ChromaDB knowledge base, alongside the existing CSV ticket data.

This script does NOT touch the CSV-trained tickets already in the DB.
It ADDS document chunks on top of the existing knowledge base so that
predict.py can use both tickets AND articles when answering queries.

Supported document types:
  .txt   plain text articles
  .md    Markdown articles
  .pdf   PDF documents (requires:  pip install pypdf)
  .html  HTML pages  (requires:  pip install beautifulsoup4  for best results)

Optional team tag in document front-matter (first 1-3 lines):
  TEAM: IT-Service Desk
  Assignment Group: IT-Network Support
  Group: IT-Wintel Support

  If present, the tag is stripped from the text and used as the
  assignment_group in ChromaDB metadata so the LLM gets a strong hint.

  Alternatively, place documents in a sub-folder named after the team:
    data/kb_docs/IT-Service Desk/vpn_guide.md
  The folder name is used as the team tag automatically.

Usage:
    # Ingest from default folder (data/kb_docs/)
    python build_knowledge_base_docs.py

    # Ingest from a custom folder
    python build_knowledge_base_docs.py --docs-path path/to/my/articles

    # Wipe existing document chunks and re-ingest from scratch
    python build_knowledge_base_docs.py --rebuild-docs

    # List documents currently in the knowledge base
    python build_knowledge_base_docs.py --list
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))

from agents.embedding_agent          import EmbeddingAgent
from agents.knowledge_base_agent     import KnowledgeBaseAgent
from agents.document_ingestion_agent import DocumentIngestionAgent


def main():
    parser = argparse.ArgumentParser(
        description="ARIA -- Ingest KB documents / articles into the knowledge base"
    )
    parser.add_argument(
        "--docs-path",
        type=str,
        default=None,
        help="Path to folder containing documents (default: data/kb_docs/)",
    )
    parser.add_argument(
        "--rebuild-docs",
        action="store_true",
        help="Delete all existing document chunks and re-ingest from scratch",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List documents currently stored in the knowledge base and exit",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Words per chunk (overrides config; default: 400)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Word overlap between consecutive chunks (default: 80)",
    )
    args = parser.parse_args()

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    db_path     = config["vector_db"]["path"]
    collection  = config["vector_db"]["collection"]
    embed_model = config["embedding"]["model"]

    # Docs path resolution order:
    #   1. --docs-path CLI argument
    #   2. data.docs_path in config.yaml
    #   3. default: data/kb_docs
    docs_path = (
        args.docs_path
        or config.get("data", {}).get("docs_path", "data/kb_docs")
    )

    chunk_size    = args.chunk_size    or config.get("docs", {}).get("chunk_size",    400)
    chunk_overlap = args.chunk_overlap or config.get("docs", {}).get("chunk_overlap", 80)

    print("")
    print("=" * 60)
    print("  ARIA -- KB Document Ingestion")
    print("=" * 60)
    print("")
    print("  Docs path      : " + docs_path)
    print("  Vector DB path : " + db_path)
    print("  Collection     : " + collection)
    print("  Embed model    : " + embed_model)
    print("  Chunk size     : " + str(chunk_size) + " words")
    print("  Chunk overlap  : " + str(chunk_overlap) + " words")
    print("")

    # Connect to KB
    kb_agent = KnowledgeBaseAgent(db_path=db_path, collection_name=collection)
    kb_agent._connect()

    # ── List mode ────────────────────────────────────────────────────────────
    if args.list:
        doc_agent = DocumentIngestionAgent(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        docs = doc_agent.list_ingested(kb_agent)
        if not docs:
            print("  No documents currently in the knowledge base.")
        else:
            print("  Documents in knowledge base:")
            print("")
            print("  {:<40} {:<35} {:>7}".format(
                "File", "Team Tag", "Chunks"))
            print("  " + "-" * 40 + " " + "-" * 35 + " " + "-" * 7)
            for d in docs:
                print("  {:<40} {:<35} {:>7}".format(
                    d["file_name"][:39],
                    (d["assignment_group"] or "(none)")[:34],
                    str(d["chunk_count"]),
                ))
            total = sum(d["chunk_count"] for d in docs)
            print("")
            print("  Total: " + str(len(docs)) + " document(s), "
                  + str(total) + " chunk(s) in knowledge base.")
        print("")
        return

    # ── Ingest mode ──────────────────────────────────────────────────────────
    if not os.path.isdir(docs_path):
        print("  [INFO] Docs folder does not exist yet. Creating: " + docs_path)
        os.makedirs(docs_path, exist_ok=True)
        print("")
        print("  Place your .txt / .md / .pdf / .html articles in:")
        print("    " + docs_path)
        print("")
        print("  Optional: tag each document with its target team in the")
        print("  first line of the file:")
        print("    TEAM: IT-Service Desk")
        print("")
        print("  Or organise files in sub-folders named after the team:")
        print("    " + docs_path + "/IT-Service Desk/vpn_guide.md")
        print("")
        sys.exit(0)

    # Load embedding model
    embed_agent = EmbeddingAgent(model_name=embed_model)
    embed_agent.load()

    # Ingest documents
    doc_agent = DocumentIngestionAgent(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    total_chunks = doc_agent.ingest(
        docs_path      = docs_path,
        embedding_agent= embed_agent,
        kb_agent       = kb_agent,
        force_rebuild  = args.rebuild_docs,
    )

    ticket_count = kb_agent.count()

    print("")
    print("  [OK] " + str(total_chunks) + " document chunk(s) ingested.")
    print("  [OK] Knowledge base total (tickets + docs): " + str(ticket_count) + " entries.")
    print("")
    print("  Next steps:")
    print("    List ingested documents :")
    print("      python build_knowledge_base_docs.py --list")
    print("    Start predicting         :")
    print("      python predict.py")
    print("")


if __name__ == "__main__":
    main()
