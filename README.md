# Enterprise Support Copilot

Minimal full‑stack RAG starter for an enterprise support copilot.  
You can upload support documents, index them into pgvector, and ask grounded questions with semantic + keyword search, reranking, and tracing.

---

## 1. Architecture

- `frontend/` – Next.js (App Router, TypeScript) demo UI
- `backend/` – FastAPI API with:
  - SQLAlchemy + pgvector for storage
  - OpenAI embeddings + chat models
  - Hybrid semantic + keyword retrieval
  - Optional reranking and Langfuse tracing
- `docker-compose.yml` – PostgreSQL with pgvector enabled

High‑level flow:

1. Documents are uploaded to the backend and stored in Postgres.
2. Text is chunked and embedded into a `chunks` table with a pgvector column.
3. At query time the system runs hybrid retrieval (vector + full‑text), optional reranking, and answer generation constrained to retrieved context.

---

## 2. Stack

- **Frontend**: Next.js 14 (App Router), React 18, TypeScript
- **Backend**: FastAPI, SQLAlchemy 2.x
- **Database**: PostgreSQL + pgvector
- **Vector search**: pgvector cosine distance
- **Keyword search**: Postgres full‑text search (tsvector / tsquery)
- **Models**: OpenAI embeddings + chat completions
- **Tracing (optional)**: Langfuse Python SDK

---

## 3. Data model

- `documents`  
  - `id` (UUID)  
  - `title`  
  - `source` (e.g. `"upload"`)  
  - `product_area` (optional)  
  - `release_version` (optional)  
  - `content_type`  
  - `raw_text` – full parsed text  
  - `created_at`

- `chunks`  
  - `id` (UUID)  
  - `document_id` → `documents.id`  
  - `index` – chunk index within document  
  - `content` – chunk text  
  - `heading` (optional)  
  - `metadata` (optional text blob)  
  - `embedding` – `vector(1536)`  
  - `created_at`

---

## 4. Ingestion flow

Triggered by: `POST /documents/upload`

1. **Upload**
   - Frontend sends a single file (`pdf`, `md`, `txt`) via multipart form‑data.
   - Backend validates filename and content type.
2. **Parsing**
   - `parsers.extract_text`:
     - PDFs via `pypdf`
     - Markdown / text via UTF‑8 decoding
   - Raw text is truncated to a safe upper bound.
3. **Document storage**
   - A `Document` row is created with:
     - `title` = filename
     - `source = "upload"`
     - `content_type`
     - `raw_text`
4. **Chunking**
   - `chunking.chunk_text`:
     - Detects simple headings (`#`, or short lines ending with `:`).
     - Splits text into blocks under headings.
     - Within each block, creates fixed‑size character windows with overlap:
       - Size: `CHUNK_SIZE` (default `800`)
       - Overlap: `CHUNK_OVERLAP` (default `200`)
5. **Embedding**
   - `embeddings.embed_texts` calls OpenAI embeddings (`text-embedding-3-small` by default).
   - Embeddings are stored in `chunks.embedding`.
6. **Chunk storage**
   - For each chunk:
     - `Chunk` row is created with `index`, `content`, `heading`, optional `metadata`, and `embedding`.

Optional: Langfuse spans are recorded around ingestion when `LANGFUSE_ENABLED=true`.

---

## 5. Retrieval flow

Triggered by: `POST /ask`

### 5.1 Request

`AskRequest` body:

- `query` (string, required)
- `top_k` (optional, default 8, clamped to [1, 20])
- Optional filters:
  - `source`
  - `product_area`
  - `release_version`
  - `start_date`, `end_date` (filter on document `created_at`)

### 5.2 Semantic retrieval

1. `embeddings.embed_texts([query])` generates a query embedding.
2. `retrieve_semantic_chunks`:
   - Uses `Chunk.embedding.cosine_distance(query_embedding)` with pgvector.
   - Applies any metadata filters.
   - Returns ranked semantic candidates (high similarity = low distance).

### 5.3 Keyword retrieval

1. `retrieve_keyword_chunks`:
   - `to_tsvector('english', chunks.content)` + `plainto_tsquery('english', query)`.
   - Filters via `@@` and ranks using `ts_rank_cd`.
   - Applies the same metadata filters.

Semantic and keyword candidate counts are limited via:

- `SEMANTIC_CANDIDATES` (default `32`)
- `KEYWORD_CANDIDATES` (default `32`)

---

## 6. Hybrid search (semantic + keyword)

`hybrid_retrieve_chunks` combines both signals:

1. Runs semantic and keyword retrieval as above.
2. If there are no keyword hits, returns the top semantic candidates only.
3. Otherwise, uses **Reciprocal Rank Fusion (RRF)**:
   - For each candidate list, contributes \(1 / (k + \text{rank})\) to a per‑chunk score (with a modest constant `k=60`).
   - Sums contributions from semantic and keyword lists per chunk.
   - Sorts by the fused score and returns the top `top_k` chunks.

This approach is simple, model‑free, and balances dense similarity with exact term matching.

---

## 7. Reranking

Reranking is optional and controlled by:

- `RERANK_ENABLED` (`true` / `false`, default `false`)
- `RERANK_MODEL` (default `gpt-4o-mini`)

When enabled:

1. The fused candidates from hybrid retrieval are passed to `rerank.rerank_items`.
2. For each `(query, chunk)` pair, a small OpenAI chat completion:
   - Asks the model to output a single numeric relevance score in \[0, 1\].
3. Candidates are sorted by this rerank score and returned as `ranked_items`.

If reranking fails for any reason (missing API key, errors, etc.), the system falls back to the hybrid retrieval ranking.

---

## 8. Confidence‑aware answers

Before calling the answer model, the backend applies a simple confidence policy:

- It looks at:
  - `top_score` – score of the best ranked chunk.
  - `avg_top` – average score of the top 3 chunks (when present).
- If both:
  - `top_score < 0.25` **and**
  - `avg_top < 0.35`
  - then evidence is treated as weak.

Behavior:

- **Weak evidence**:
  - Returns a fixed answer:
    - “The answer cannot be confidently determined from the available documents.”
  - Still returns the best matching chunks as citations.
- **Stronger evidence**:
  - Calls `qa.generate_answer` with the selected context chunks.

The frontend shows a clear message when an answer is not well supported.

---

## 9. Tracing and evaluation (Langfuse)

Tracing is optional and controlled by:

- `LANGFUSE_ENABLED` (`true` / `false`, default `false`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL`

When enabled, the app sends observations to Langfuse:

- **Ingestion**
  - `document_ingestion` trace with:
    - Filename, content‑type.
    - Output: document ID, number of chunks.
  - Nested embedding spans via `embed_texts`.
- **Ask**
  - `ask` trace with:
    - Query, `top_k`.
    - Child observations:
      - `embed_texts` for query embedding.
      - `hybrid_retrieval` with semantic/keyword/fused candidates.
      - `rerank` (when enabled) with per‑chunk scores.
      - `answer_generation` with final answer content.
    - Output: selected chunks and whether the answer was supported.

If Langfuse is disabled or misconfigured, all tracing calls become no‑ops and the app continues to function normally.

---

## 10. Setup steps

### 10.1 Root

```bash
cp .env.example .env
```

The root `.env` is used by Docker Compose for PostgreSQL defaults.

### 10.2 Database

```bash
docker compose up -d
```

This starts PostgreSQL with pgvector on `localhost:5432`.

### 10.3 Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
```

Set at minimum:

- `OPENAI_API_KEY`

Optional:

- `OPENAI_EMBEDDING_MODEL`
- `OPENAI_CHAT_MODEL`
- `SEMANTIC_CANDIDATES`, `KEYWORD_CANDIDATES`
- `RERANK_ENABLED`, `RERANK_MODEL`
- `LANGFUSE_*` variables

Run the API:

```bash
uvicorn app.main:app --reload --port 8000
```

Health check:

- `GET http://localhost:8000/health`

### 10.4 Frontend

```bash
cd frontend
npm install
cp .env.example .env
```

Make sure:

- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`

Run the dev server:

```bash
npm run dev
```

Open `http://localhost:3000`.

---

## 11. Demo steps

1. **Start services**
   - Postgres: `docker compose up -d`
   - Backend: `uvicorn app.main:app --reload --port 8000`
   - Frontend: `npm run dev` in `frontend`
2. **Upload a document**
   - Visit `http://localhost:3000`.
   - In “Upload documents”:
     - Choose a `.pdf`, `.md`, or `.txt` support document.
     - Click **Upload** (file is sent to `/documents/upload`).
3. **Ask a question**
   - In “Ask a question”:
     - Enter a question that the uploaded docs can answer.
     - Click **Ask**.
4. **Review answer and citations**
   - The “Answer” panel shows:
     - The answer text.
     - If evidence is weak, a note that the answer could not be verified.
     - A list of citations with document names, chunk indices, scores, headings, and previews.
5. **Inspect traces (optional)**
   - If Langfuse is configured, open the Langfuse UI to inspect:
     - Ingestion traces for uploads.
     - Ask traces for questions, including retrieval and model calls.

---

## 12. Known limitations

- No authentication or authorization.
- No schema migrations (tables are created automatically on startup; use with a dev database).
- No rate limiting or multi‑tenant separation.
- Reranking uses a simple per‑chunk chat call, which may be slow or costly for very large candidate sets.
- No streaming responses; answers are returned after full completion.
- Frontend is a single‑page demo and not production UI.

These trade‑offs keep the codebase small and easy to review while showing the core RAG patterns end‑to‑end. 

