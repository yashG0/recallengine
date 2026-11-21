import asyncpg
from pgvector.asyncpg import register_vector

DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/recall_db"

pool: asyncpg.Pool | None = None


async def init_db() -> None:
    global pool

    # 1. Ensure the extension exists before registering the vector type on pooled connections
    setup_conn = await asyncpg.connect(DATABASE_URL)
    try:
        await setup_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    finally:
        await setup_conn.close()

    # 2. Teach every connection in the pool how to read/write the 'vector' type
    async def _init_connection(conn: asyncpg.Connection) -> None:
        await register_vector(conn)

    pool = await asyncpg.create_pool(
        dsn=DATABASE_URL,
        min_size=2,
        max_size=10,
        init=_init_connection,
    )

    # 3. Create the table and HNSW cosine index if they don't exist yet
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                embedding vector(384) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS notes_embedding_idx
            ON notes USING hnsw (embedding vector_cosine_ops);
        """)


async def close_db() -> None:
    if pool is not None:
        await pool.close()
