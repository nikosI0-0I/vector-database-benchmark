import logging
import numpy as np
from typing import List, Optional
from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class EmbeddingService:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        self._model = None
        self._initialized = False

    def initialize(self) -> None:
        """Lazily initialize the embedding model."""
        if self._initialized:
            return

        try:
            from fastembed import TextEmbedding
            from pathlib import Path
            import os

            # Use workspace cache directory or temp cache
            cache_dir = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "fastembed"
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Loading FastEmbed model: {self.model_name}...")
            # If temp cache has the model, also accept default or local_files_only if available
            self._model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(cache_dir),
                local_files_only=False
            )
            self._initialized = True
            logger.info("FastEmbed model loaded successfully.")
        except Exception as e:
            # Try loading from default temp cache before fallback
            try:
                self._model = TextEmbedding(model_name=self.model_name)
                self._initialized = True
                logger.info("FastEmbed model loaded from default cache.")
            except Exception as e2:
                logger.warning(
                    f"Could not load FastEmbed ({e2}). Falling back to deterministic pseudo-embeddings."
                )
                self._model = None
                self._initialized = True

    def _fallback_embed(self, text: str) -> List[float]:
        """Deterministic pseudo-embedding generator for testing/offline mode."""
        # Use simple hash seed to generate normalized pseudo-random vector
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding vector for a single string."""
        if not self._initialized:
            self.initialize()

        if self._model:
            # fastembed embed returns a generator of numpy arrays
            embeddings = list(self._model.embed([text]))
            return embeddings[0].tolist()
        return self._fallback_embed(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts in batch."""
        if not self._initialized:
            self.initialize()

        if self._model:
            embeddings = list(self._model.embed(texts))
            return [emb.tolist() for emb in embeddings]
        return [self._fallback_embed(t) for t in texts]


embedding_service = EmbeddingService()
