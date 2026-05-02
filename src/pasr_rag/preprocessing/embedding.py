from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import AppConfig


@dataclass(frozen=True)
class EncoderSpec:
    model_id: str
    model_path: str
    batch_size: int


class BGEEmbeddingEncoder:
    """Shared BGE encoder for preprocessing, routing, and local retrieval."""

    _model_cache: dict[EncoderSpec, "SentenceTransformer"] = {}

    def __init__(self, model_id: str, model_path: str, batch_size: int = 32) -> None:
        self.spec = EncoderSpec(
            model_id=model_id,
            model_path=str(self._resolve_model_path(model_id, model_path)),
            batch_size=batch_size,
        )
        self._model = self._load_model(self.spec)

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        normalized_texts = [self._prepare_query(text) if is_query else text for text in texts]
        embeddings = self._model.encode(
            normalized_texts,
            batch_size=self.spec.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    @classmethod
    def _load_model(cls, spec: EncoderSpec) -> "SentenceTransformer":
        cached = cls._model_cache.get(spec)
        if cached is not None:
            return cached

        from sentence_transformers import SentenceTransformer
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(spec.model_path, device=device)
        cls._model_cache[spec] = model
        return model

    def _resolve_model_path(self, model_id: str, model_path: str) -> Path:
        path = Path(model_path)
        if path.exists():
            return path.resolve()

        model_name = model_id.split("/")[-1]
        local_candidate = Path.cwd() / "models" / model_name
        if local_candidate.exists():
            return local_candidate.resolve()

        return Path(model_id)

    def _prepare_query(self, text: str) -> str:
        return f"Represent this sentence for searching relevant passages: {text.strip()}"


def build_embedding_encoder(config: AppConfig) -> BGEEmbeddingEncoder:
    return BGEEmbeddingEncoder(
        model_id=config.models.embedding_model,
        model_path=config.models.embedding_model_path,
        batch_size=config.models.embedding_batch_size,
    )
