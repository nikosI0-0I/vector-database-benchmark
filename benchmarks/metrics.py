import numpy as np
from typing import List, Dict, Any, Set

def calculate_recall_at_k(retrieved: List[str], ground_truth: List[str], k: int = 10) -> float:
    """Calculate Recall@k against exact ground truth."""
    if not ground_truth:
        return 0.0
    k_retrieved = set(retrieved[:k])
    k_truth = set(ground_truth[:k])
    return len(k_retrieved.intersection(k_truth)) / len(k_truth)

def calculate_precision_at_k(retrieved: List[str], ground_truth: List[str], k: int = 10) -> float:
    """Calculate Precision@k."""
    if not retrieved or k <= 0:
        return 0.0
    k_retrieved = set(retrieved[:k])
    k_truth = set(ground_truth[:k])
    return len(k_retrieved.intersection(k_truth)) / min(k, len(k_retrieved))

def calculate_ndcg_at_k(retrieved: List[str], ground_truth: List[str], k: int = 10) -> float:
    """Calculate Normalized Discounted Cumulative Gain at k (binary relevance)."""
    k_truth_set = set(ground_truth[:k])
    dcg = 0.0
    for idx, item in enumerate(retrieved[:k]):
        if item in k_truth_set:
            dcg += 1.0 / np.log2(idx + 2)  # idx+2 because idx starts at 0

    idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(k, len(k_truth_set))))
    return (dcg / idcg) if idcg > 0 else 0.0

def calculate_latency_percentiles(latencies_ms: List[float], total_duration_sec: float) -> Dict[str, float]:
    """Calculate p50, p90, p95, p99, mean latency in ms and QPS."""
    if not latencies_ms:
        return {"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "qps": 0.0}

    arr = np.array(latencies_ms)
    qps = len(latencies_ms) / total_duration_sec if total_duration_sec > 0 else 0.0
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "qps": round(qps, 2),
    }
