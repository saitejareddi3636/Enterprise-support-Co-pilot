from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Mapping

from langfuse import get_client

from .core.config import settings


try:
    _client = get_client() if settings.langfuse_enabled else None
except Exception:
    _client = None


def observation(
    name: str,
    *,
    as_type: str = "span",
    input: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
    **kwargs: Any,
):
    if not _client:
        return nullcontext()

    return _client.start_as_current_observation(
        as_type=as_type,
        name=name,
        input=input,
        metadata=metadata,
        **kwargs,
    )


