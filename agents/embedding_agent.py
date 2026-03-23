"""
Embedding Agent
---------------
Converts ticket text into dense vector embeddings using a
sentence-transformer model running fully locally.

Model: all-MiniLM-L6-v2
  - 384 dimensional embeddings
  - Fast, lightweight, runs on CPU
  - No API key required
  - Downloads once (~90MB), cached to models/ folder locally
    so it never re-downloads on subsequent runs

Cache:
  Model files are saved to models/ inside the project folder.
  On Windows, make sure HF_TOKEN is set to avoid rate limiting:
    setx HF_TOKEN "hf_your_token_here"
  Then close and reopen the command prompt before running.
"""

import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingAgent:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model      = None

    def load(self):
        """
        Load the embedding model. Called once at startup.
        Model is cached to models/ folder so it downloads only once.
        """
        print("  Loading embedding model: " + self.model_name)
        self.model = SentenceTransformer(self.model_name, cache_folder="models/")
        print("  Embedding model ready.")

    def embed(self, text: str) -> list[float]:
        """Convert a single text string into an embedding vector."""
        if self.model is None:
            self.load()
        text   = self._preprocess(text)
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of texts into embedding vectors.
        Faster than calling embed() one by one — uses batched encoding internally.
        """
        if self.model is None:
            self.load()
        cleaned = [self._preprocess(t) for t in texts]
        vectors = self.model.encode(cleaned, convert_to_numpy=True, show_progress_bar=True)
        return vectors.tolist()

    def _preprocess(self, text: str) -> str:
        """Clean text before embedding."""
        if not text:
            return ""
        # Lowercase and strip extra whitespace
        text = text.lower().strip()
        text = " ".join(text.split())
        return text
