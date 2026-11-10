from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from backend.app.services.document import document_service
from backend.app.databases.postgres import postgres_client
from backend.app.databases.qdrant import qdrant_client_wrapper

router = APIRouter(prefix="/benchmark", tags=["Benchmark Operations"])

class LoadDocumentsRequest(BaseModel):
    documents: Optional[List[Dict[str, Any]]] = None

@router.post("/init-schemas")
async def init_schemas(dimension: int = 384):
    """Initialize tables, extensions, and collections in PostgreSQL and Qdrant."""
    await postgres_client.init_schema(dimension=dimension)
    await qdrant_client_wrapper.init_collection(dimension=dimension, recreate=True)
    return {"status": "initialized", "dimension": dimension}

@router.post("/index-samples")
async def index_samples(req: Optional[LoadDocumentsRequest] = None):
    """Index sample documents into both PostgreSQL and Qdrant."""
    docs = req.documents if req and req.documents else document_service.load_sample_documents()
    res = await document_service.index_documents(docs)
    return {"status": "success", "result": res}

@router.post("/build-hnsw-indexes")
async def build_hnsw_indexes(m: int = 16, ef_construction: int = 64):
    """Create HNSW vector index in PostgreSQL."""
    await postgres_client.create_hnsw_index(m=m, ef_construction=ef_construction)
    return {"status": "indices_built", "m": m, "ef_construction": ef_construction}
