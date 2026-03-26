"""
Knowledge Base Agent
---------------------
Builds and queries the ChromaDB vector database from the CSV training data.

Flow:
  BUILD  : Read CSV -> embed each ticket -> store in ChromaDB with metadata
  SEARCH : Embed query -> find top-K most similar tickets -> return with metadata

ChromaDB stores:
  - The embedding vector for each ticket
  - Metadata: short_description, description, assignment_group
  - Persists to disk so rebuild is only needed when CSV changes

Incremental loading:
  The build() method accepts start and end row indices so the CSV can be
  loaded in chunks (e.g. 10,000 rows at a time) to avoid memory errors
  on machines with limited RAM.

  Each chunk ADDS to the existing knowledge base — nothing is overwritten.
  IDs use the global row index (start + i) so they never clash across runs.

  Usage:
    python build_knowledge_base.py --start 0     --end 10000
    python build_knowledge_base.py --start 10000 --end 20000
    ... and so on

Similarity scores:
  Raw cosine similarity (0-1) is scaled to 1-10 for display.
  The raw value is preserved as similarity_raw for weighted-vote math
  in the LLM agent.
"""

import csv
import logging
import os

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class KnowledgeBaseAgent:

    def __init__(self, db_path: str, collection_name: str):
        self.db_path         = db_path
        self.collection_name = collection_name
        self.client          = None
        self.collection      = None

    def _connect(self):
        """Connect to ChromaDB persistent storage."""
        os.makedirs(self.db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine similarity
        )

    def build(self, csv_path: str, embedding_agent,
              force_rebuild: bool = False,
              start: int = 0,
              end: int = 10000):
        """
        Build or append to the knowledge base from the CSV file.

        Args:
            csv_path        : path to training_tickets.csv
            embedding_agent : EmbeddingAgent instance
            force_rebuild   : if True, wipe existing data and start fresh
            start           : first row index to process (0-based)
            end             : last row index to process (exclusive)

        Reads the full CSV but only embeds and stores rows in [start, end).
        IDs are set to str(start + i) so multiple runs never produce
        duplicate or clashing IDs in ChromaDB.

        Embedding is done in chunks of 2000 rows to keep memory usage low.
        ChromaDB insertion is done in batches of 500.
        """
        self._connect()

        existing_count = self.collection.count()

        # Wipe and rebuild if requested
        if force_rebuild and existing_count > 0:
            print("  Rebuilding knowledge base from scratch...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            existing_count = 0

        if existing_count > 0:
            print("  Existing knowledge base: " + str(existing_count) + " tickets.")
            print("  Appending new batch (rows " + str(start) + " to " + str(end) + ")...")
        else:
            print("  Starting fresh knowledge base.")
            print("  Loading rows " + str(start) + " to " + str(end) + "...")

        # Read full CSV — track global row index for stable IDs
        all_rows = []
        with open(csv_path, newline="", encoding="latin-1") as fh:
            for row in csv.DictReader(fh):
                team = row.get("Assignment Team", "").strip()
                if not team:
                    continue
                sd   = row.get("Short Description", "").strip()
                desc = row.get("Description", "").strip()
                if not sd:
                    continue
                all_rows.append({
                    "short_description": sd,
                    "description":       desc,
                    "assignment_group":  team,
                    "text":              sd + " " + desc,
                })

        # Slice the requested range
        rows = all_rows[start:end]

        if not rows:
            print("  [WARNING] No rows found in range " + str(start) + " to " + str(end))
            print("  Total rows in CSV: " + str(len(all_rows)))
            return self.collection.count()

        print("  " + str(len(rows)) + " tickets to embed in this batch...")

        # Embed in chunks of 2000 to keep RAM usage low
        chunk_size = 2000
        embeddings = []
        for i in range(0, len(rows), chunk_size):
            chunk = [r["text"] for r in rows[i : i + chunk_size]]
            embeddings.extend(embedding_agent.embed_batch(chunk))
            done = min(i + chunk_size, len(rows))
            print("  Embedded " + str(done) + " / " + str(len(rows)) + " tickets...")

        # Assign globally unique IDs using the start offset
        for i, row in enumerate(rows):
            row["id"] = str(start + i)

        # Store in ChromaDB in batches of 500
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch     = rows[i : i + batch_size]
            batch_emb = embeddings[i : i + batch_size]
            self.collection.add(
                ids        = [r["id"] for r in batch],
                embeddings = batch_emb,
                metadatas  = [
                    {
                        "short_description": r["short_description"],
                        "description":       r["description"][:500],
                        "assignment_group":  r["assignment_group"],
                        "source_type":       "ticket",
                        "file_name":         "",
                    }
                    for r in batch
                ],
                documents = [r["short_description"] for r in batch],
            )

        total = self.collection.count()
        print("  Batch complete. Knowledge base total: " + str(total) + " tickets.")
        return total

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Find the top-K most similar entries (tickets OR document chunks)
        to the query embedding.

        Returns list of dicts with:
          short_description : str
          description       : str
          assignment_group  : str
          similarity_score  : float  1.0-10.0  (scaled for display)
          similarity_raw    : float  0.0-1.0   (kept for weighted-vote math)
          source_type       : "ticket" | "document"
          file_name         : str  (document chunks only, else "")
        """
        if self.collection is None:
            self._connect()

        results = self.collection.query(
            query_embeddings = [query_embedding],
            n_results        = top_k,
            include          = ["metadatas", "distances", "documents"],
        )

        tickets = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            # cosine similarity in [0,1]; scale to 1-10 for display
            raw_sim = round(1 - results["distances"][0][i], 4)
            scaled  = round(1 + raw_sim * 9, 1)   # 0.0 -> 1.0,  1.0 -> 10.0
            source_type = meta.get("source_type", "ticket")
            tickets.append({
                "short_description": meta.get("short_description", ""),
                "description":       meta.get("description", ""),
                "assignment_group":  meta.get("assignment_group", ""),
                "similarity_score":  scaled,       # 1.0 – 10.0  (display)
                "similarity_raw":    raw_sim,      # 0.0 – 1.0   (vote math)
                "source_type":       source_type,  # "ticket" or "document"
                "file_name":         meta.get("file_name", ""),
            })

        return tickets

    def count(self) -> int:
        """Return number of tickets in the knowledge base."""
        if self.collection is None:
            self._connect()
        return self.collection.count()
