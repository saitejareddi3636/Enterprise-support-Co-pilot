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

