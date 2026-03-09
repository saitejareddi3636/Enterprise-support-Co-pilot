from __future__ import annotations

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from .core.config import settings
from .db import Base, engine, get_db
from . import chunking, embeddings, models, parsers, qa, retrieval, schemas


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

    @app.post(
        "/ask",
        response_model=schemas.AskResponse,
    )
    def ask(
        payload: schemas.AskRequest,
        db: Session = Depends(get_db),
    ) -> schemas.AskResponse:
        query = payload.query.strip()
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query is required.",
            )

        top_k = payload.top_k or 8
        if top_k <= 0:
            top_k = 8
        if top_k > 20:
            top_k = 20

        try:
            query_embedding = embeddings.embed_texts([query])[0]
        except embeddings.EmbeddingError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        filters = retrieval.RetrievalFilters(
            source=payload.source,
            product_area=payload.product_area,
            release_version=payload.release_version,
            start_date=payload.start_date,
            end_date=payload.end_date,
        )

        items = retrieval.retrieve_chunks(
            db=db,
            query_embedding=query_embedding,
            top_k=top_k,
            filters=filters,
        )

        if not items:
            empty_response = schemas.AskResponse(
                answer=(
                    "The answer cannot be confidently determined from the available "
                    "documents."
                ),
                chunks=[],
                documents=[],
            )
            return empty_response

        context_chunks: list[qa.ContextChunk] = []
        retrieved_chunks: list[schemas.RetrievedChunk] = []
        seen_documents: dict[str, str] = {}

        for item in items:
            chunk = item.chunk
            document = item.document

            context_chunks.append(
                qa.ContextChunk(
                    content=chunk.content,
                    document_title=document.title,
                    source=document.source,
                    product_area=document.product_area,
                    release_version=document.release_version,
                    heading=chunk.heading,
                    score=item.score,
                    index=chunk.index,
                )
            )

            retrieved_chunks.append(
                schemas.RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_title=document.title,
                    source=document.source,
                    product_area=document.product_area,
                    release_version=document.release_version,
                    created_at=document.created_at,
                    index=chunk.index,
                    heading=chunk.heading,
                    content=chunk.content,
                    score=item.score,
                )
            )

            doc_key = str(document.id)
            if doc_key not in seen_documents:
                seen_documents[doc_key] = document.title

        try:
            answer_text = qa.generate_answer(query, context_chunks)
        except qa.AnswerGenerationError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        response = schemas.AskResponse(
            answer=answer_text,
            chunks=retrieved_chunks,
            documents=list(seen_documents.values()),
        )

        return response

    return app


app = create_app()

