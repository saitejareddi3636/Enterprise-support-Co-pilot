from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from .core.config import settings
from .db import Base, engine, get_db
from . import chunking, embeddings, models, parsers, schemas


def create_app() -> FastAPI:
    app = FastAPI(title="Enterprise Support Copilot API")

    @app.on_event("startup")
    def on_startup() -> None:
        Base.metadata.create_all(bind=engine)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "env": settings.app_env}

    @app.post(
        "/documents/upload",
        response_model=schemas.DocumentCreateResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def upload_document(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
    ) -> models.Document:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File name is required.",
            )

        try:
            text = parsers.extract_text(file)
        except parsers.UnsupportedFileTypeError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except parsers.FileParsingError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        document = models.Document(
            title=file.filename,
            source="upload",
            content_type=file.content_type,
            raw_text=text,
        )
        db.add(document)

        chunks = chunking.chunk_text(
            text=text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

        if chunks:
            texts = [c.text for c in chunks]
            try:
                vectors = embeddings.embed_texts(texts)
            except embeddings.EmbeddingError as exc:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=str(exc),
                ) from exc

            for chunk_result, vector in zip(chunks, vectors, strict=True):
                chunk = models.Chunk(
                    document=document,
                    index=chunk_result.index,
                    content=chunk_result.text,
                    heading=chunk_result.heading,
                    metadata=None,
                    embedding=vector,
                )
                db.add(chunk)

        db.commit()
        db.refresh(document)

        return document

    @app.get(
        "/documents",
        response_model=list[schemas.DocumentListItem],
    )
    def list_documents(db: Session = Depends(get_db)) -> list[schemas.DocumentListItem]:
        documents = (
            db.query(models.Document)
            .order_by(models.Document.created_at.desc())
            .limit(100)
            .all()
        )

        items: list[schemas.DocumentListItem] = []
        for doc in documents:
            preview = (doc.raw_text or "").strip()
            if preview:
                preview = preview[:200]

            items.append(
                schemas.DocumentListItem(
                    id=doc.id,
                    title=doc.title,
                    source=doc.source,
                    content_type=doc.content_type,
                    created_at=doc.created_at,
                    raw_text_preview=preview or None,
                )
            )

        return items

    return app


app = create_app()

