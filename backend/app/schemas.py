from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DocumentBase(BaseModel):
    id: uuid.UUID
    title: str
    source: str | None = None
    content_type: str | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentCreateResponse(DocumentBase):
    pass


class DocumentListItem(DocumentBase):
    raw_text_preview: str | None = None


class AskRequest(BaseModel):
    query: str
    top_k: int | None = None
    source: str | None = None
    product_area: str | None = None
    release_version: str | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None


class RetrievedChunk(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_title: str
    source: str | None = None
    product_area: str | None = None
    release_version: str | None = None
    created_at: datetime
    index: int
    heading: str | None = None
    content: str
    score: float


class AskResponse(BaseModel):
    answer: str
    chunks: list[RetrievedChunk]
    documents: list[str]

