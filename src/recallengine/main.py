import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recallengine import db
from recallengine.embedder import embed_text, load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize DB pool and load embedding model into RAM
    await db.init_db()
    await asyncio.to_thread(load_model)
    yield
    # Shutdown: cleanly close DB connection pool
    await db.close_db()


app = FastAPI(
    title="RecallEngine",
    description="Semantic knowledge retrieval backend powered by PostgreSQL pgvector.",
    version="0.1.0",
    lifespan=lifespan,
)


class NoteCreate(BaseModel):
    content: str = Field(..., min_length=3, max_length=2000)


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=20)
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)


@app.post("/notes", status_code=201)
async def create_note(payload: NoteCreate):
    if db.pool is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    # Offload CPU embedding computation so the async event loop stays non-blocking
    vector = await asyncio.to_thread(embed_text, payload.content)

    async with db.pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO notes (content, embedding)
            VALUES ($1, $2)
            RETURNING id, created_at;
            """,
            payload.content,
            vector,
        )

    return {
        "id": row["id"],
        "content": payload.content,
        "created_at": row["created_at"],
    }


@app.post("/search")
async def search_notes(payload: SearchQuery):
    if db.pool is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    query_vector = await asyncio.to_thread(embed_text, payload.query)
    max_distance = 1.0 - payload.min_similarity

    async with db.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id,
                content,
                1 - (embedding <=> $1) AS similarity
            FROM notes
            WHERE (embedding <=> $1) <= $2
            ORDER BY embedding <=> $1 ASC
            LIMIT $3;
            """,
            query_vector,
            max_distance,
            payload.top_k,
        )

    return {
        "query": payload.query,
        "min_similarity": payload.min_similarity,
        "results": [
            {
                "id": r["id"],
                "content": r["content"],
                "similarity": round(float(r["similarity"]), 4),
            }
            for r in rows
        ],
    }
