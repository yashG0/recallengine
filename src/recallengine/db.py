import asyncpg
from pgvector.asyncpg import register_vector

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/recall_db"

pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global pool

    setup_conn = await asyncpg.connect(DATABASE_URL)
    try:
        await setup_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    finally:
        await setup_conn.close()

    async def _init_connection(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=2,
        max_size=10,
        init=_init_connection,
    )

    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding vector(384) NOT NULL,
                UNIQUE (document_id, chunk_index)
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS chunks_embedding_idx
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """)


async def close_db() -> None:
    if pool is not None:
        await pool.close()
