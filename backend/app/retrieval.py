from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from sqlalchemy.orm import Session

from . import models


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


def retrieve_chunks(
  db: Session,
  query_embedding: Sequence[float],
  top_k: int,
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

  if filters:
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

  rows = query.order_by(distance_expr).limit(top_k).all()

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

