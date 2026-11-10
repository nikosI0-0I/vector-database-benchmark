import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.app.core.config import get_settings
from backend.app.databases.postgres import postgres_client
from backend.app.databases.qdrant import qdrant_client_wrapper
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.search import router as search_router
from backend.app.api.routes.benchmark import router as benchmark_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vector_benchmark")
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events."""
    logger.info("Initializing vector benchmark backend...")
    # Attempt connections (fail softly so server starts even if DBs are spinning up)
    try:
        await postgres_client.connect()
    except Exception as e:
        logger.warning(f"PostgreSQL connection deferred: {e}")

    try:
        await qdrant_client_wrapper.connect()
    except Exception as e:
        logger.warning(f"Qdrant connection deferred: {e}")

    yield

    logger.info("Shutting down vector benchmark backend...")
    await postgres_client.close()
    await qdrant_client_wrapper.close()

app = FastAPI(
    title="Vector Database Benchmark API",
    description="Benchmarking PostgreSQL (pgvector) vs. Qdrant",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS configuration for Frontend (Vite)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(health_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(benchmark_router, prefix="/api")

@app.get("/")
async def root():
    return {
        "project": "Vector Database Benchmark",
        "description": "PostgreSQL (pgvector) vs Qdrant Benchmark Suite",
        "docs_url": "/docs",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
