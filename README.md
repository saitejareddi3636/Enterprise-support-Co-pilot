# Enterprise Support Copilot

Minimal full-stack starter for an Enterprise Support Copilot using:

- Next.js (TypeScript) frontend
- FastAPI backend
- PostgreSQL with pgvector
- Docker Compose for local database

## Structure

- `frontend/` – Next.js app (App Router)
- `backend/` – FastAPI app with SQLAlchemy models
- `docker-compose.yml` – PostgreSQL + pgvector

## Prerequisites

- Node.js 18+
- Python 3.11+
- Docker + Docker Compose

## Environment variables

### Root

Copy the root example file:

```bash
cp .env.example .env
```

The root `.env` is used by Docker Compose for PostgreSQL defaults.

### Backend

```bash
cp backend/.env.example backend/.env
```

Key variables:

- `APP_ENV` – environment name (e.g. `local`)
- `APP_PORT` – FastAPI port (default `8000`)
- `DATABASE_URL` – SQLAlchemy connection string

### Frontend

```bash
cp frontend/.env.example frontend/.env
```

Key variables:

- `NEXT_PUBLIC_API_BASE_URL` – base URL for the backend API

## Running PostgreSQL with pgvector

From the project root:

```bash
docker compose up -d
```

This starts PostgreSQL with pgvector enabled on port `5432`.

## Backend: FastAPI

Install dependencies:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run the app:

```bash
uvicorn app.main:app --reload --port 8000
```

Health check:

- `GET http://localhost:8000/health`

On startup, the API creates tables for documents and chunks.

## Frontend: Next.js

Install dependencies:

```bash
cd frontend
npm install
```

Run the dev server:

```bash
npm run dev
```

Open `http://localhost:3000` in your browser.

Home page sections:

- Upload documents
- Ask a question
- Answer display

The UI is intentionally minimal so you can wire up RAG flows next.

