import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from recallengine import db
from recallengine.chunker import chunk_text
from recallengine.embedder import embed_text, embed_texts, load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init_db()
    await asyncio.to_thread(load_model)
    yield
    await db.close_db()


app = FastAPI(
    title="RecallEngine",
    description="Semantic knowledge retrieval backend powered by PostgreSQL pgvector.",
    version="1.0.0",
    lifespan=lifespan,
)


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    content: str = Field(..., min_length=10, max_length=50000)


class SearchQuery(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    top_k: int = Field(default=3, ge=1, le=20)
    min_similarity: float = Field(default=0.35, ge=0.0, le=1.0)


async def _ingest_document(title: str, content: str) -> dict:
    """Chunk text, compute batch embeddings, and persist atomically."""
    if db.pool is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    chunks = chunk_text(content, chunk_size=150, overlap=30)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document content cannot be blank")

    vectors = await asyncio.to_thread(embed_texts, chunks)

    async with db.pool.acquire() as conn, conn.transaction():
        doc_row = await conn.fetchrow(
            """
                INSERT INTO documents (title)
                VALUES ($1)
                RETURNING id, created_at;
                """,
            title,
        )
        doc_id = doc_row["id"]

        records = [
            (doc_id, idx, chunk_text_item, vec)
            for idx, (chunk_text_item, vec) in enumerate(zip(chunks, vectors))
        ]
    await conn.executemany(
        """
        INSERT INTO chunks (document_id, chunk_index, content, embedding)
        VALUES ($1, $2, $3, $4);
        """,
        records,
    )
    return {
        "document_id": doc_id,
        "title": title,
        "chunks_created": len(chunks),
        "created_at": doc_row["created_at"],
    }


@app.post("/documents", status_code=201)
async def create_document(payload: DocumentCreate):
    return await _ingest_document(payload.title, payload.content)


@app.get("/documents")
async def list_documents(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    if db.pool is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    async with db.pool.acquire() as conn:
        total = await conn.fetchval("SELECT COUNT(*) FROM documents;")
        rows = await conn.fetch(
            """
            SELECT
                d.id,
                d.title,
                d.created_at,
                COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC, d.id DESC
            LIMIT $1 OFFSET $2;
            """,
            limit,
            offset,
        )

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": [
            {
                "id": r["id"],
                "title": r["title"],
                "chunk_count": r["chunk_count"],
                "created_at": r["created_at"],
            }
            for r in rows
        ],
    }


@app.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: int):
    if db.pool is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    async with db.pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM documents WHERE id = $1;",
            document_id,
        )

    if result == "DELETE 0":
        raise HTTPException(status_code=404, detail="Document not found")


@app.post("/search")
async def search_documents(payload: SearchQuery):
    if db.pool is None:
        raise HTTPException(status_code=503, detail="Database not initialized")

    query_vector = await asyncio.to_thread(embed_text, payload.query)
    max_distance = 1.0 - payload.min_similarity

    async with db.pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                c.id AS chunk_id,
                d.id AS document_id,
                d.title AS document_title,
                c.chunk_index,
                c.content,
                1 - (c.embedding <=> $1) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE (c.embedding <=> $1) <= $2
            ORDER BY c.embedding <=> $1 ASC
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
                "document_id": r["document_id"],
                "document_title": r["document_title"],
                "chunk_index": r["chunk_index"],
                "content": r["content"],
                "similarity": round(float(r["similarity"]), 4),
            }
            for r in rows
        ],
    }
