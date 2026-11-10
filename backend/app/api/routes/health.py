from fastapi import APIRouter
from backend.app.databases.postgres import postgres_client
from backend.app.databases.qdrant import qdrant_client_wrapper

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
async def get_health():
    """Healthcheck endpoint for backend API and databases."""
    pg_health = await postgres_client.check_health()
    qd_health = await qdrant_client_wrapper.check_health()

    all_healthy = (
        pg_health.get("status") == "healthy" and
        qd_health.get("status") == "healthy"
    )

    return {
        "status": "healthy" if all_healthy else "degraded",
        "databases": {
            "postgresql": pg_health,
            "qdrant": qd_health,
        }
    }
