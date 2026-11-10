import time
import logging
from typing import Any, Dict, List, Optional
from backend.app.databases.postgres import postgres_client
from backend.app.databases.qdrant import qdrant_client_wrapper
from backend.app.services.embedding import embedding_service

logger = logging.getLogger(__name__)

class SearchService:
    @staticmethod
    async def search(
        query: str,
        target_db: str = "both",  # "postgres", "qdrant", or "both"
        limit: int = 10,
        category_filter: Optional[str] = None,
        ef_search: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate embedding for query and execute vector search against chosen databases."""
        t0 = time.perf_counter()
        query_embedding = embedding_service.embed_text(query)
        embed_time_ms = (time.perf_counter() - t0) * 1000

        response: Dict[str, Any] = {
            "query": query,
            "embed_time_ms": round(embed_time_ms, 2),
            "results": {},
        }

        if target_db in ("postgres", "both"):
            t_pg = time.perf_counter()
            pg_res = await postgres_client.search(
                query_embedding=query_embedding,
                limit=limit,
                ef_search=ef_search,
                category_filter=category_filter,
            )
            pg_time_ms = (time.perf_counter() - t_pg) * 1000
            response["results"]["postgres"] = {
                "latency_ms": round(pg_time_ms, 2),
                "count": len(pg_res),
                "hits": pg_res,
            }

        if target_db in ("qdrant", "both"):
            t_qd = time.perf_counter()
            qd_res = await qdrant_client_wrapper.search(
                query_embedding=query_embedding,
                limit=limit,
                ef_search=ef_search,
                category_filter=category_filter,
            )
            qd_time_ms = (time.perf_counter() - t_qd) * 1000
            response["results"]["qdrant"] = {
                "latency_ms": round(qd_time_ms, 2),
                "count": len(qd_res),
                "hits": qd_res,
            }

        return response


search_service = SearchService()
