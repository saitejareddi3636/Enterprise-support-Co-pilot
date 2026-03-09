from __future__ import annotations

from typing import Sequence

from openai import OpenAI
from google import genai

from .core.config import settings
from . import observability


class EmbeddingError(Exception):
    pass


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise EmbeddingError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=settings.openai_api_key)


def _gemini_client() -> genai.Client:
    if not settings.gemini_api_key:
        raise EmbeddingError("GEMINI_API_KEY is not configured.")
    return genai.Client(api_key=settings.gemini_api_key)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    if settings.llm_provider == "gemini":
        client = _gemini_client()
        with observability.observation(
            "embed_texts",
            as_type="generation",
            metadata={"provider": "gemini", "model": settings.gemini_embedding_model, "count": len(texts)},
        ):
            try:
                result = client.models.embed_content(
                    model=settings.gemini_embedding_model,
                    contents=list(texts),
                )
            except Exception as exc:
                raise EmbeddingError("Failed to generate embeddings.") from exc

        vectors: list[list[float]] = []
        for emb in getattr(result, "embeddings", []) or []:
            values = getattr(emb, "values", None)
            if values is None:
                values = getattr(emb, "embedding", None)
            vectors.append(list(values or []))
    else:
        client = _client()
        with observability.observation(
            "embed_texts",
            as_type="generation",
            metadata={"provider": "openai", "model": settings.openai_embedding_model, "count": len(texts)},
        ):
            try:
                response = client.embeddings.create(
                    model=settings.openai_embedding_model,
                    input=list(texts),
                )
            except Exception as exc:
                raise EmbeddingError("Failed to generate embeddings.") from exc

        vectors: list[list[float]] = []
        for item in response.data:
            vectors.append(list(item.embedding))

    if len(vectors) != len(texts):
        raise EmbeddingError("Embedding count mismatch.")

    expected_dim = 1536
    first = vectors[0] if vectors else []
    if first and len(first) != expected_dim:
        raise EmbeddingError(
            f"Embedding dimension mismatch: got {len(first)}, expected {expected_dim}. "
            f"Your database is configured for vector({expected_dim})."
        )

    return vectors

