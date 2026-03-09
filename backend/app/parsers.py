from __future__ import annotations

from typing import Final

from fastapi import UploadFile
from pypdf import PdfReader


class UnsupportedFileTypeError(Exception):
    pass


class FileParsingError(Exception):
    pass


_MAX_TEXT_LENGTH: Final[int] = 1_000_000


def _read_text_file(upload_file: UploadFile) -> str:
    try:
        data = upload_file.file.read()
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="ignore")
        else:
            text = str(data)
    except Exception as exc:
        raise FileParsingError("Could not read text file") from exc
    finally:
        upload_file.file.seek(0)

    return text[:_MAX_TEXT_LENGTH]


def _read_pdf(upload_file: UploadFile) -> str:
    try:
        upload_file.file.seek(0)
        reader = PdfReader(upload_file.file)
        pages: list[str] = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        text = "\n".join(pages)
    except Exception as exc:
        raise FileParsingError("Could not read PDF file") from exc
    finally:
        upload_file.file.seek(0)

    if not text.strip():
        raise FileParsingError("PDF appears to be empty or unreadable")

    return text[:_MAX_TEXT_LENGTH]


def extract_text(upload_file: UploadFile) -> str:
    filename = (upload_file.filename or "").lower()

    if filename.endswith(".pdf"):
        text = _read_pdf(upload_file)
    elif filename.endswith(".md") or filename.endswith(".markdown"):
        text = _read_text_file(upload_file)
    elif filename.endswith(".txt"):
        text = _read_text_file(upload_file)
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type for '{upload_file.filename}'.")

    if not text.strip():
        raise FileParsingError("File appears to be empty or unreadable")

    return text

