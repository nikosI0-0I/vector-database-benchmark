import logging
from typing import Any, Dict, List, Optional
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, SearchParams
from backend.app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class QdrantClientWrapper:
    def __init__(self):
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name = "benchmark_documents"

    async def connect(self) -> None:
        """Initialize AsyncQdrantClient."""
        if self.client is not None:
            return

        try:
            self.client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                grpc_port=settings.QDRANT_GRPC_PORT,
                api_key=settings.QDRANT_API_KEY,
                prefer_grpc=settings.QDRANT_PREFER_GRPC,
                timeout=10.0,
            )
            logger.info("Connected to Qdrant successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise

    async def close(self) -> None:
        """Close Qdrant client connection."""
        if self.client:
            await self.client.close()
            self.client = None
            logger.info("Closed Qdrant client.")

    async def check_health(self) -> Dict[str, Any]:
        """Check Qdrant health status."""
        if not self.client:
            return {"status": "disconnected", "database": "qdrant"}
        try:
            # Check cluster/collections status
            collections = await self.client.get_collections()
            return {
                "status": "healthy",
                "database": "qdrant",
                "collections_count": len(collections.collections),
            }
        except Exception as e:
            return {
                "status": "error",
                "database": "qdrant",
                "error": str(e),
            }

    async def init_collection(
        self,
        dimension: int = 384,
        m: int = 16,
        ef_construct: int = 64,
        recreate: bool = False,
    ) -> None:
        """Create or recreate the benchmark collection with specific HNSW configs."""
        collections = await self.client.get_collections()
        exists = any(c.name == self.collection_name for c in collections.collections)

        if exists and recreate:
            await self.client.delete_collection(self.collection_name)
            exists = False

        if not exists:
            logger.info(f"Creating Qdrant collection {self.collection_name} (dim={dimension})...")
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dimension,
                    distance=Distance.COSINE,
                ),
                hnsw_config=rest.HnswConfigDiff(
                    m=m,
                    ef_construct=ef_construct,
                ),
            )
            # Create payload index for category to ensure single-stage filtered search
            await self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="category",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Collection {self.collection_name} created with category payload index.")

    async def insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Upsert points into Qdrant in batches."""
        points = [
            PointStruct(
                id=doc["int_id"] if "int_id" in doc else idx,
                vector=doc["embedding"],
                payload={
                    "doc_id": doc["doc_id"],
                    "content": doc["content"],
                    "category": doc.get("category"),
                    "metadata": doc.get("metadata", {}),
                },
            )
            for idx, doc in enumerate(documents, start=1)
        ]
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    async def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        ef_search: Optional[int] = None,
        exact: bool = False,
        category_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Perform similarity search on Qdrant."""
        query_filter = None
        if category_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=category_filter),
                    )
                ]
            )

        search_params = SearchParams(
            hnsw_ef=ef_search,
            exact=exact,
        )

        results = await self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=query_filter,
            search_params=search_params,
        )

        return [
            {
                "doc_id": hit.payload.get("doc_id"),
                "content": hit.payload.get("content"),
                "category": hit.payload.get("category"),
                "metadata": hit.payload.get("metadata"),
                "similarity": hit.score,
            }
            for hit in results
        ]


qdrant_client_wrapper = QdrantClientWrapper()
