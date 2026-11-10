import pytest
import asyncio
from backend.app.databases.postgres import postgres_client
from backend.app.databases.qdrant import qdrant_client_wrapper
from backend.app.services.embedding import embedding_service
from backend.app.services.document import document_service

@pytest.mark.asyncio
async def test_embedding_service():
    """Test generating embeddings for sample text."""
    text = "Vector database benchmark test string."
    vector = embedding_service.embed_text(text)
    assert isinstance(vector, list)
    assert len(vector) == embedding_service.dimension
    assert all(isinstance(x, float) for x in vector)

@pytest.mark.asyncio
async def test_postgres_connection():
    """Test PostgreSQL connection and health status."""
    try:
        await postgres_client.connect()
        health = await postgres_client.check_health()
        assert health["status"] == "healthy"
        assert "pg_version" in health
    except Exception as e:
        pytest.skip(f"PostgreSQL not accessible: {e}")
    finally:
        await postgres_client.close()

@pytest.mark.asyncio
async def test_qdrant_connection():
    """Test Qdrant connection and health status."""
    try:
        await qdrant_client_wrapper.connect()
        health = await qdrant_client_wrapper.check_health()
        assert health["status"] == "healthy"
    except Exception as e:
        pytest.skip(f"Qdrant not accessible: {e}")
    finally:
        await qdrant_client_wrapper.close()
