from __future__ import annotations

from dataclasses import dataclass

from .core.config import settings


@dataclass
class ChunkResult:
    index: int
    text: str
    heading: str | None


def _split_into_lines(text: str) -> list[str]:
    return text.splitlines()


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("#"):
        return True
    if stripped.endswith(":") and len(stripped.split()) <= 10:
        return True
    return False


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[ChunkResult]:
    if not text.strip():
        return []

    size = chunk_size or settings.chunk_size
    overlap = chunk_overlap or settings.chunk_overlap
    if size <= 0:
        size = 800
    if overlap < 0:
        overlap = 0
    if overlap >= size:
        overlap = int(size / 4)

    lines = _split_into_lines(text)

    heading: str | None = None
    buffers: list[tuple[str | None, str]] = []
    current_lines: list[str] = []

    for line in lines:
        if _is_heading(line):
            if current_lines:
                buffers.append((heading, "\n".join(current_lines).strip()))
                current_lines = []
            heading = line.strip()
            continue

        current_lines.append(line)

    if current_lines:
        buffers.append((heading, "\n".join(current_lines).strip()))

    chunks: list[ChunkResult] = []
    index = 0

    for heading_text, block in buffers:
        block = block.strip()
        if not block:
            continue

        if len(block) <= size:
            chunks.append(ChunkResult(index=index, text=block, heading=heading_text))
            index += 1
            continue

        start = 0
        while start < len(block):
            end = min(start + size, len(block))
            window = block[start:end].strip()
            if window:
                chunks.append(
                    ChunkResult(index=index, text=window, heading=heading_text)
                )
                index += 1
            if end == len(block):
                break
            start = max(end - overlap, 0)

    return chunks

