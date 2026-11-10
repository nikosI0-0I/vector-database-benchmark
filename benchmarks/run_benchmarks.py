import asyncio
import time
import json
import logging
from typing import List, Dict, Any
from backend.app.databases.postgres import postgres_client
from backend.app.databases.qdrant import qdrant_client_wrapper
from backend.app.services.document import document_service
from backend.app.services.embedding import embedding_service
from benchmarks.metrics import (
    calculate_recall_at_k,
    calculate_ndcg_at_k,
    calculate_latency_percentiles,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_runner")

async def run_benchmark_suite():
    logger.info("Connecting to databases...")
    await postgres_client.connect()
    await qdrant_client_wrapper.connect()

    dimension = embedding_service.dimension
    logger.info(f"Initializing schemas for dimension {dimension}...")
    await postgres_client.init_schema(dimension=dimension)
    await qdrant_client_wrapper.init_collection(dimension=dimension, recreate=True)

    # Load and index documents
    docs = document_service.load_sample_documents()
    logger.info(f"Loaded {len(docs)} sample documents.")
    t0 = time.perf_counter()
    index_res = await document_service.index_documents(docs)
    ingest_time = time.perf_counter() - t0
    logger.info(f"Indexed documents in {ingest_time:.2f}s: {index_res}")

    # Build HNSW index in PostgreSQL
    t_idx = time.perf_counter()
    await postgres_client.create_hnsw_index(m=16, ef_construction=64)
    pg_index_build_sec = time.perf_counter() - t_idx
    logger.info(f"PostgreSQL HNSW index built in {pg_index_build_sec:.2f}s.")

    # Test queries
    test_queries = [
        "What is PostgreSQL pgvector?",
        "Vector search engine written in Rust",
        "Algorithmic trading and financial markets",
        "Healthcare AI and medical diagnosis",
        "Rocket boosters and low orbit satellites",
    ]

    logger.info(f"Running benchmark on {len(test_queries)} queries...")
    pg_latencies = []
    qd_latencies = []
    recalls = []

    for q in test_queries:
        query_vec = embedding_service.embed_text(q)

        # 1. PostgreSQL search
        t_pg = time.perf_counter()
        pg_results = await postgres_client.search(query_vec, limit=5, ef_search=40)
        pg_lat = (time.perf_counter() - t_pg) * 1000
        pg_latencies.append(pg_lat)

        # 2. Qdrant search
        t_qd = time.perf_counter()
        qd_results = await qdrant_client_wrapper.search(query_vec, limit=5, ef_search=40)
        qd_lat = (time.perf_counter() - t_qd) * 1000
        qd_latencies.append(qd_lat)

        # Compare top-5 retrieval agreement (Recall@5 proxy between engines)
        pg_ids = [r["doc_id"] for r in pg_results]
        qd_ids = [r["doc_id"] for r in qd_results]
        recall = calculate_recall_at_k(pg_ids, qd_ids, k=5)
        recalls.append(recall)

        logger.info(f"Query: '{q}' -> PG Latency: {pg_lat:.2f}ms | QD Latency: {qd_lat:.2f}ms | Overlap Recall@5: {recall:.2f}")

    pg_stats = calculate_latency_percentiles(pg_latencies, sum(pg_latencies) / 1000.0)
    qd_stats = calculate_latency_percentiles(qd_latencies, sum(qd_latencies) / 1000.0)

    summary = {
        "dataset_size": len(docs),
        "embedding_dimension": dimension,
        "pg_index_build_time_sec": round(pg_index_build_sec, 3),
        "mean_recall_overlap": round(sum(recalls) / len(recalls), 3),
        "postgresql_metrics": pg_stats,
        "qdrant_metrics": qd_stats,
    }

    logger.info(f"Benchmark Summary: {json.dumps(summary, indent=2)}")

    await postgres_client.close()
    await qdrant_client_wrapper.close()
    return summary

if __name__ == "__main__":
    asyncio.run(run_benchmark_suite())
