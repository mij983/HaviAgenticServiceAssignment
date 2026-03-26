"""
Document Ingestion Agent
-------------------------
Reads knowledge-base articles and documents (TXT, MD, PDF, HTML)
from a directory and ingests them into ChromaDB alongside the existing
CSV-trained tickets.

Each document is split into overlapping chunks so that long articles
can be embedded without exceeding the model's context window.

Supported file types:
  .txt   plain text
  .md    Markdown (stored as plain text)
  .pdf   PDF (text extracted page-by-page)
  .html  HTML (tags stripped, plain text extracted)

How it works:
  1. Walk the docs_path directory recursively for supported files
  2. Extract text from each file
  3. Split into overlapping chunks (chunk_size / chunk_overlap configurable)
  4. Embed each chunk with EmbeddingAgent
  5. Store in ChromaDB with metadata:
       source_type   : "document"
       file_name     : original filename
       chunk_index   : which chunk within the file
       assignment_group : tag extracted from the file's front-matter
                          (e.g. first line "TEAM: IT-Service Desk")
                          or inferred from folder name, or left as ""

IDs are prefixed with "doc_" + stable hash of (filename + chunk_index)
so they never clash with CSV ticket IDs (which are plain integers).

ChromaDB collection:
  Documents share the SAME collection as CSV tickets (snow_tickets).
  The metadata field source_type ("ticket" vs "document") lets the
  search layer know what kind of result each hit is, and the LLM
  agent receives both types ranked together.

Usage (via build_knowledge_base_docs.py):
  python build_knowledge_base_docs.py
  python build_knowledge_base_docs.py --docs-path data/kb_docs
  python build_knowledge_base_docs.py --rebuild-docs
"""

import hashlib
import logging
import os
import re

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Optional imports — graceful fallback if libraries not installed
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pypdf import PdfReader as _PdfReader        # pip install pypdf
    _HAS_PDF = True
except ImportError:
    _HAS_PDF = False

try:
    from bs4 import BeautifulSoup as _BS             # pip install beautifulsoup4
    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}


class DocumentIngestionAgent:
    """
    Reads documents from a directory, chunks them, embeds them,
    and stores them in the ChromaDB knowledge-base collection.
    """

    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
    ):
        """
        Args:
            chunk_size    : approximate word count per chunk
            chunk_overlap : number of words to overlap between consecutive chunks
        """
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def ingest(
        self,
        docs_path:       str,
        embedding_agent,
        kb_agent,
        force_rebuild:   bool = False,
    ) -> int:
        """
        Walk docs_path, embed all supported documents, and store them
        in the knowledge base.

        Args:
            docs_path       : directory containing KB articles / documents
            embedding_agent : EmbeddingAgent instance
            kb_agent        : KnowledgeBaseAgent instance (already connected)
            force_rebuild   : if True, delete all existing "document" entries
                              before re-ingesting

        Returns:
            int : total number of document chunks ingested in this run
        """
        if not os.path.isdir(docs_path):
            raise FileNotFoundError(
                f"Docs directory not found: {docs_path}\n"
                "Create the folder and place your .txt / .md / .pdf / .html "
                "articles inside it."
            )

        # Ensure KB collection is open
        if kb_agent.collection is None:
            kb_agent._connect()

        if force_rebuild:
            self._delete_existing_docs(kb_agent)

        files = self._collect_files(docs_path)
        if not files:
            print("  [WARNING] No supported documents found in: " + docs_path)
            print("  Supported types: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
            return 0

        print("  Found " + str(len(files)) + " document(s) to ingest...")

        total_chunks = 0

        for file_path in files:
            rel = os.path.relpath(file_path, docs_path)
            print("  Processing: " + rel)

            try:
                text, assignment_group = self._extract_text(file_path, docs_path)
            except Exception as e:
                print("    [SKIP] Could not read file: " + str(e))
                continue

            if not text.strip():
                print("    [SKIP] Empty document.")
                continue

            chunks = self._split_into_chunks(text)
            if not chunks:
                print("    [SKIP] No usable content after chunking.")
                continue

            print("    " + str(len(chunks)) + " chunk(s) — team tag: " +
                  (assignment_group or "(none)"))

            # Embed chunks in one batch
            embeddings = embedding_agent.embed_batch(chunks)

            # Build records
            ids        = []
            metadatas  = []
            documents  = []

            file_name = os.path.basename(file_path)

            for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_id = self._make_id(file_name, idx)
                ids.append(chunk_id)
                metadatas.append({
                    "source_type":      "document",
                    "file_name":        file_name,
                    "relative_path":    rel,
                    "chunk_index":      str(idx),
                    "assignment_group": assignment_group,
                    "short_description": self._make_title(file_name, idx),
                    "description":      chunk[:500],
                })
                documents.append(chunk[:200])   # ChromaDB document field (preview)

            # Upsert so re-runs don't create duplicates
            batch_size = 500
            for i in range(0, len(ids), batch_size):
                kb_agent.collection.upsert(
                    ids        = ids[i : i + batch_size],
                    embeddings = embeddings[i : i + batch_size],
                    metadatas  = metadatas[i : i + batch_size],
                    documents  = documents[i : i + batch_size],
                )

            total_chunks += len(chunks)

        return total_chunks

    def list_ingested(self, kb_agent) -> list[dict]:
        """
        Return a summary of all documents currently in the knowledge base.
        Each entry: {file_name, relative_path, chunk_count, assignment_group}
        """
        if kb_agent.collection is None:
            kb_agent._connect()

        results = kb_agent.collection.get(
            where={"source_type": "document"},
            include=["metadatas"],
        )

        # Group by file
        files: dict[str, dict] = {}
        for meta in results.get("metadatas", []):
            fname = meta.get("file_name", "?")
            if fname not in files:
                files[fname] = {
                    "file_name":        fname,
                    "relative_path":    meta.get("relative_path", ""),
                    "assignment_group": meta.get("assignment_group", ""),
                    "chunk_count":      0,
                }
            files[fname]["chunk_count"] += 1

        return sorted(files.values(), key=lambda x: x["file_name"])

    # ─────────────────────────────────────────────────────────────────────────
    # Text extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_text(self, file_path: str, docs_root: str) -> tuple[str, str]:
        """
        Extract plain text from the file.

        Returns:
            (text, assignment_group)
            assignment_group may be "" if no tag found.
        """
        ext = os.path.splitext(file_path)[1].lower()

        if ext in (".txt", ".md"):
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                raw = fh.read()

        elif ext == ".pdf":
            if not _HAS_PDF:
                raise ImportError(
                    "pypdf is required for PDF support.\n"
                    "Install it with:  pip install pypdf"
                )
            reader = _PdfReader(file_path)
            raw    = "\n".join(
                (page.extract_text() or "") for page in reader.pages
            )

        elif ext in (".html", ".htm"):
            with open(file_path, encoding="utf-8", errors="replace") as fh:
                html = fh.read()
            if _HAS_BS4:
                soup = _BS(html, "html.parser")
                raw  = soup.get_text(separator="\n")
            else:
                # Naive tag stripper fallback
                raw = re.sub(r"<[^>]+>", " ", html)

        else:
            raise ValueError("Unsupported file type: " + ext)

        # Extract optional team tag from front-matter
        assignment_group, raw = self._extract_team_tag(raw, file_path, docs_root)

        return raw, assignment_group

    def _extract_team_tag(
        self, text: str, file_path: str, docs_root: str
    ) -> tuple[str, str]:
        """
        Look for a team annotation in the first 3 lines of the document.

        Recognised formats:
            TEAM: IT-Service Desk
            Assignment Group: IT-Network Support
            Group: IT-Wintel Support

        If found, the line is stripped from the returned text.

        Falls back to the immediate parent folder name if it looks like
        an assignment group (contains "IT-" or "HR-" or "BUS-").

        Returns:
            (assignment_group_str, cleaned_text)
        """
        tag_pattern = re.compile(
            r"^(?:TEAM|ASSIGNMENT\s+GROUP|GROUP)\s*:\s*(.+)$",
            re.IGNORECASE,
        )

        lines = text.splitlines()
        for i, line in enumerate(lines[:3]):
            m = tag_pattern.match(line.strip())
            if m:
                group = m.group(1).strip()
                remaining = "\n".join(lines[:i] + lines[i + 1:])
                return group, remaining

        # Fallback: folder name heuristic
        parent = os.path.basename(os.path.dirname(file_path))
        if any(prefix in parent for prefix in ("IT-", "HR-", "BUS-")):
            return parent, text

        return "", text

    # ─────────────────────────────────────────────────────────────────────────
    # Text chunking
    # ─────────────────────────────────────────────────────────────────────────

    def _split_into_chunks(self, text: str) -> list[str]:
        """
        Split text into overlapping word-based chunks.

        Uses word count (not character count) as the unit so chunk size
        is predictable regardless of word length.
        """
        # Normalise whitespace
        text  = re.sub(r"\n{3,}", "\n\n", text)
        text  = text.strip()

        words = text.split()
        if not words:
            return []

        chunks  = []
        start   = 0
        size    = self.chunk_size
        overlap = self.chunk_overlap

        while start < len(words):
            end   = min(start + size, len(words))
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(words):
                break
            start = end - overlap  # step back by overlap for continuity

        return chunks

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_files(self, docs_path: str) -> list[str]:
        """Walk docs_path recursively and return all supported file paths."""
        found = []
        for root, _dirs, files in os.walk(docs_path):
            for fname in sorted(files):
                ext = os.path.splitext(fname)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    found.append(os.path.join(root, fname))
        return found

    def _make_id(self, file_name: str, chunk_index: int) -> str:
        """
        Create a stable unique ID for a chunk.
        Prefixed with "doc_" so it never collides with CSV ticket IDs (integers).
        """
        raw    = file_name + "|" + str(chunk_index)
        digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
        return "doc_" + digest

    def _make_title(self, file_name: str, chunk_index: int) -> str:
        """Human-readable title for a document chunk."""
        base = os.path.splitext(file_name)[0].replace("_", " ").replace("-", " ")
        if chunk_index == 0:
            return base
        return base + " (part " + str(chunk_index + 1) + ")"

    def _delete_existing_docs(self, kb_agent) -> None:
        """Delete all document chunks from the collection."""
        print("  Removing existing document chunks from knowledge base...")
        try:
            existing = kb_agent.collection.get(
                where={"source_type": "document"},
                include=[],
            )
            ids_to_delete = existing.get("ids", [])
            if ids_to_delete:
                # ChromaDB delete accepts up to 5461 IDs at once
                batch = 5000
                for i in range(0, len(ids_to_delete), batch):
                    kb_agent.collection.delete(ids=ids_to_delete[i : i + batch])
                print("  Removed " + str(len(ids_to_delete)) + " existing document chunk(s).")
            else:
                print("  No existing document chunks found.")
        except Exception as e:
            logger.warning("Could not delete existing docs: %s", e)
