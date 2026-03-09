from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from sqlalchemy import func
from sqlalchemy.orm import Session

from . import models
from .core.config import settings


@dataclass
class RetrievalFilters:
    source: str | None = None
    product_area: str | None = None
    release_version: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None


@dataclass
class RetrievedItem:
    chunk: models.Chunk
    document: models.Document
    score: float


def _apply_filters(query, filters: RetrievalFilters | None):
    if not filters:
        return query

    if filters.source:
        query = query.filter(models.Document.source == filters.source)
    if filters.product_area:
        query = query.filter(models.Document.product_area == filters.product_area)
    if filters.release_version:
        query = query.filter(models.Document.release_version == filters.release_version)
    if filters.start_date:
        query = query.filter(models.Document.created_at >= filters.start_date)
    if filters.end_date:
        query = query.filter(models.Document.created_at <= filters.end_date)
    return query


def retrieve_semantic_chunks(
    db: Session,
    query_embedding: Sequence[float],
    limit: int,
    filters: RetrievalFilters | None = None,
) -> list[RetrievedItem]:
    distance_expr = models.Chunk.embedding.cosine_distance(query_embedding)

    query = (
        db.query(
            models.Chunk,
            models.Document,
            distance_expr.label("distance"),
        )
        .join(models.Document, models.Chunk.document_id == models.Document.id)
    )

    query = _apply_filters(query, filters)

    rows = query.order_by(distance_expr).limit(limit).all()

    results: list[RetrievedItem] = []
    for chunk, document, distance in rows:
        similarity = 1.0 - float(distance) if distance is not None else 0.0
        results.append(
            RetrievedItem(
                chunk=chunk,
                document=document,
                score=similarity,
            )
        )

    return results


def retrieve_keyword_chunks(
    db: Session,
    query_text: str,
    limit: int,
    filters: RetrievalFilters | None = None,
) -> list[RetrievedItem]:
    if not query_text.strip():
        return []

    tsvector = func.to_tsvector("english", models.Chunk.content)
    tsquery = func.plainto_tsquery("english", query_text)
    rank_expr = func.ts_rank_cd(tsvector, tsquery)

    query = (
        db.query(
            models.Chunk,
            models.Document,
            rank_expr.label("rank"),
        )
        .join(models.Document, models.Chunk.document_id == models.Document.id)
        .filter(tsvector.op("@@")(tsquery))
    )

    query = _apply_filters(query, filters)

    rows = query.order_by(rank_expr.desc()).limit(limit).all()

    results: list[RetrievedItem] = []
    for chunk, document, _rank in rows:
        # Use rank ordering only; score will be set by fusion later.
        results.append(
            RetrievedItem(
                chunk=chunk,
                document=document,
                score=1.0,
            )
        )

    return results


def _reciprocal_rank_fusion(
    semantic: list[RetrievedItem],
    keyword: list[RetrievedItem],
    k: int,
    limit: int,
) -> list[RetrievedItem]:
    scores: dict[str, float] = {}
    items: dict[str, RetrievedItem] = {}

    for rank, item in enumerate(semantic, start=1):
        key = str(item.chunk.id)
        items[key] = item
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    for rank, item in enumerate(keyword, start=1):
        key = str(item.chunk.id)
        if key not in items:
            items[key] = item
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ordered_keys = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    fused: list[RetrievedItem] = []
    for key, fused_score in ordered_keys[:limit]:
        base_item = items[key]
        fused.append(
            RetrievedItem(
                chunk=base_item.chunk,
                document=base_item.document,
                score=fused_score,
            )
        )

    return fused


def hybrid_retrieve_chunks(
    db: Session,
    query_text: str,
    query_embedding: Sequence[float],
    top_k: int,
    filters: RetrievalFilters | None = None,
) -> list[RetrievedItem]:
    semantic_limit = min(top_k, settings.semantic_candidates)
    keyword_limit = min(top_k, settings.keyword_candidates)

    semantic_items = retrieve_semantic_chunks(
        db=db,
        query_embedding=query_embedding,
        limit=semantic_limit,
        filters=filters,
    )

    keyword_items: list[RetrievedItem] = []
    if keyword_limit > 0:
        keyword_items = retrieve_keyword_chunks(
            db=db,
            query_text=query_text,
            limit=keyword_limit,
            filters=filters,
        )

    if not keyword_items:
        return semantic_items[:top_k]

    # k parameter controls how quickly contributions from lower-ranked
    # results decay. A modest constant keeps the formula stable.
    return _reciprocal_rank_fusion(
        semantic=semantic_items,
        keyword=keyword_items,
        k=60,
        limit=top_k,
    )


