from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from .core.config import settings


class EmbeddingError(Exception):
    pass


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise EmbeddingError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=settings.openai_api_key)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []

    client = _client()

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

    return vectors

