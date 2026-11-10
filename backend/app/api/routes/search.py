from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel
from backend.app.services.search import search_service

router = APIRouter(prefix="/search", tags=["Semantic Search"])

class SearchRequest(BaseModel):
    query: str
    target_db: str = "both"  # "postgres", "qdrant", "both"
    limit: int = 10
    category_filter: Optional[str] = None
    ef_search: Optional[int] = None

@router.post("")
async def search_endpoint(req: SearchRequest):
    """Execute semantic vector search against PostgreSQL and/or Qdrant."""
    return await search_service.search(
        query=req.query,
        target_db=req.target_db,
        limit=req.limit,
        category_filter=req.category_filter,
        ef_search=req.ef_search,
    )
