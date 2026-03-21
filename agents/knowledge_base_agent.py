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

    def build(self, csv_path: str, embedding_agent, force_rebuild: bool = False):
        """
        Build the knowledge base from the CSV file.

        Reads every row from training_tickets.csv, embeds the
        short_description + description text, and stores in ChromaDB.

        Set force_rebuild=True to wipe and rebuild from scratch.
        """
        self._connect()

        existing_count = self.collection.count()
        if existing_count > 0 and not force_rebuild:
            print("  Knowledge base already exists with " + str(existing_count) + " tickets.")
            print("  Skipping build. Use --rebuild flag to force rebuild.")
            return existing_count

        if force_rebuild and existing_count > 0:
            print("  Rebuilding knowledge base from scratch...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        # Read CSV
        rows = []
        with open(csv_path, newline="", encoding="latin-1") as fh:
            for row in csv.DictReader(fh):
                team = row.get("Assignment Team", "").strip()
                if not team:
                    continue
                sd   = row.get("Short Description", "").strip()
                desc = row.get("Description", "").strip()
                if not sd:
                    continue
                rows.append({
                    "id":               str(len(rows)),
                    "short_description": sd,
                    "description":       desc,
                    "assignment_group":  team,
                    "text":              sd + " " + desc,
                })

        if not rows:
            print("  [ERROR] No valid rows found in CSV.")
            return 0

        print("  Embedding " + str(len(rows)) + " tickets into knowledge base...")
        texts = [r["text"] for r in rows]
        embeddings = embedding_agent.embed_batch(texts)

        # Store in ChromaDB in batches
        batch_size = 100
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
                    }
                    for r in batch
                ],
                documents = [r["short_description"] for r in batch],
            )

        total = self.collection.count()
        print("  Knowledge base built with " + str(total) + " tickets.")
        return total

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """
        Find the top-K most similar tickets to the query embedding.

        Returns list of dicts with:
          short_description, description, assignment_group, distance
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
            # cosine similarity in [0, 1]; scale to 1-10 for display
            raw_sim  = round(1 - results["distances"][0][i], 4)
            scaled   = round(1 + raw_sim * 9, 1)   # 0.0 -> 1.0, 1.0 -> 10.0
            tickets.append({
                "short_description": meta.get("short_description", ""),
                "description":       meta.get("description", ""),
                "assignment_group":  meta.get("assignment_group", ""),
                "similarity_score":  scaled,        # 1.0 – 10.0
                "similarity_raw":    raw_sim,        # kept for weighted-vote math
            })

        return tickets

    def count(self) -> int:
        """Return number of tickets in the knowledge base."""
        if self.collection is None:
            self._connect()
        return self.collection.count()
