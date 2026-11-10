import asyncpg
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional
from pgvector.asyncpg import register_vector
from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class PostgresClient:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Initialize asyncpg connection pool and register pgvector."""
        if self.pool is not None:
            return

        async def init_connection(conn: asyncpg.Connection):
            await register_vector(conn)

        try:
            self.pool = await asyncpg.create_pool(
                dsn=settings.postgres_dsn,
                min_size=settings.POSTGRES_POOL_MIN_SIZE,
                max_size=settings.POSTGRES_POOL_MAX_SIZE,
                init=init_connection,
                timeout=10.0,
            )
            logger.info("Connected to PostgreSQL successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("Closed PostgreSQL connection pool.")

    async def check_health(self) -> Dict[str, Any]:
        """Check PostgreSQL server health and pgvector extension status."""
        if not self.pool:
            return {"status": "disconnected", "database": "postgresql"}
        try:
            async with self.pool.acquire() as conn:
                version = await conn.fetchval("SELECT version();")
                pgvector_version = await conn.fetchval(
                    "SELECT extversion FROM pg_extension WHERE extname = 'vector';"
                )
                return {
                    "status": "healthy",
                    "database": "postgresql",
                    "pg_version": version,
                    "pgvector_version": pgvector_version or "not_installed",
                }
        except Exception as e:
            return {
                "status": "error",
                "database": "postgresql",
                "error": str(e),
            }

    async def init_schema(self, dimension: int = 384) -> None:
        """Enable pgvector extension and create the benchmark schema."""
        async with self.pool.acquire() as conn:
            # Enable vector extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS benchmark_documents (
                    id BIGSERIAL PRIMARY KEY,
                    doc_id VARCHAR(64) UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    category VARCHAR(64),
                    metadata JSONB DEFAULT '{{}}'::jsonb,
                    embedding vector({dimension}) NOT NULL
                );
            """)
            logger.info("Postgres schema initialized.")

    async def create_hnsw_index(self, m: int = 16, ef_construction: int = 64) -> None:
        """Create HNSW index with cosine distance operator."""
        async with self.pool.acquire() as conn:
            await conn.execute("DROP INDEX IF EXISTS idx_docs_hnsw_cosine;")
            logger.info(f"Building Postgres HNSW index (m={m}, ef_construction={ef_construction})...")
            await conn.execute(f"""
                CREATE INDEX idx_docs_hnsw_cosine ON benchmark_documents
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = {m}, ef_construction = {ef_construction});
            """)
            logger.info("Postgres HNSW index created successfully.")

    async def drop_index(self) -> None:
        """Drop vector index to force exact flat scans."""
        async with self.pool.acquire() as conn:
            await conn.execute("DROP INDEX IF EXISTS idx_docs_hnsw_cosine;")
            logger.info("Dropped Postgres HNSW index.")

    async def insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Bulk insert documents with embeddings."""
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO benchmark_documents (doc_id, content, category, metadata, embedding)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (doc_id) DO UPDATE 
                SET content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding;
                """,
                [
                    (
                        doc["doc_id"],
                        doc["content"],
                        doc.get("category"),
                        doc.get("metadata", {}),
                        doc["embedding"],
                    )
                    for doc in documents
                ],
            )

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        ef_search: Optional[int] = None,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search (cosine distance)."""
        async with self.pool.acquire() as conn:
            if ef_search:
                await conn.execute(f"SET LOCAL hnsw.ef_search = {ef_search};")

            if category_filter:
                query = """
                    SELECT doc_id, content, category, metadata,
                           1 - (embedding <=> $1) AS similarity
                    FROM benchmark_documents
                    WHERE category = $2
                    ORDER BY embedding <=> $1
                    LIMIT $3;
                """
                rows = await conn.fetch(query, query_embedding, category_filter, limit)
            else:
                query = """
                    SELECT doc_id, content, category, metadata,
                           1 - (embedding <=> $1) AS similarity
                    FROM benchmark_documents
                    ORDER BY embedding <=> $1
                    LIMIT $2;
                """
                rows = await conn.fetch(query, query_embedding, limit)

            return [
                {
                    "doc_id": r["doc_id"],
                    "content": r["content"],
                    "category": r["category"],
                    "metadata": r["metadata"],
                    "similarity": float(r["similarity"]),
                }
                for r in rows
            ]


postgres_client = PostgresClient()
