from __future__ import annotations

from typing import Sequence

from openai import OpenAI

from .core.config import settings
from .observability import observation
from .retrieval import RetrievedItem


class RerankError(Exception):
    pass


def _client() -> OpenAI:
    if not settings.openai_api_key:
        raise RerankError("OPENAI_API_KEY is not configured.")
    return OpenAI(api_key=settings.openai_api_key)


def rerank_items(query: str, items: Sequence[RetrievedItem]) -> list[RetrievedItem]:
    if not settings.rerank_enabled:
        return list(items)

    if not items:
        return []

    client = _client()
    scored: list[tuple[RetrievedItem, float]] = []

    try:
        with observation(
            "rerank",
            as_type="span",
            input={"query": query, "candidate_count": len(items)},
            metadata={"model": settings.rerank_model},
        ) as obs:
            for item in items:
                prompt = (
                    "You are ranking a single text chunk for how useful it is to answer "
                    "a user question.\n\n"
                    f"Question:\n{query}\n\n"
                    "Chunk:\n"
                    f"{item.chunk.content}\n\n"
                    "Respond with a single number between 0 and 1 where 1 means "
                    "the chunk is highly relevant and 0 means it is not relevant at all. "
                    "Do not include any other text."
                )

                response = client.chat.completions.create(
                    model=settings.rerank_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only a relevance score between 0 and 1.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    temperature=0,
                )
                content = response.choices[0].message.content or ""
                score = float(content.strip())
                if score < 0.0:
                    score = 0.0
                if score > 1.0:
                    score = 1.0
                scored.append((item, score))

            if obs is not None:
                obs.update(
                    output={
                        "scores": [
                            {
                                "chunk_id": str(item.chunk.id),
                                "document_id": str(item.document.id),
                                "score": score,
                            }
                            for item, score in scored
                        ]
                    }
                )
    except Exception as exc:
        raise RerankError("Failed to rerank chunks.") from exc

    scored.sort(key=lambda pair: pair[1], reverse=True)

    reranked: list[RetrievedItem] = []
    for item, score in scored:
        reranked.append(
            RetrievedItem(
                chunk=item.chunk,
                document=item.document,
                score=score,
            )
        )

    return reranked

