import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.databases.postgres import postgres_client
from backend.app.databases.qdrant import qdrant_client_wrapper
from backend.app.services.embedding import embedding_service

logger = logging.getLogger(__name__)

class DocumentService:
    @staticmethod
    def load_sample_documents(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load sample documents from json file."""
        if file_path is None:
            # Default to data/sample_documents/documents.json
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            file_path = base_dir / "data" / "sample_documents" / "documents.json"
        else:
            file_path = Path(file_path)

        if not file_path.exists():
            logger.warning(f"File {file_path} not found. Returning empty list.")
            return []

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    @staticmethod
    async def index_documents(
        documents: List[Dict[str, Any]],
        sync_postgres: bool = True,
        sync_qdrant: bool = True,
    ) -> Dict[str, Any]:
        """Generate embeddings and index documents in both databases."""
        if not documents:
            return {"status": "empty", "count": 0}

        texts = [doc["content"] for doc in documents]
        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = embedding_service.embed_batch(texts)

        enriched_docs = []
        for idx, (doc, emb) in enumerate(zip(documents, embeddings), start=1):
            enriched_docs.append({
                "int_id": idx,
                "doc_id": doc.get("doc_id", f"doc_{idx}"),
                "content": doc["content"],
                "category": doc.get("category", "general"),
                "metadata": doc.get("metadata", {}),
                "embedding": emb,
            })

        results = {"count": len(enriched_docs)}

        if sync_postgres:
            await postgres_client.insert_documents(enriched_docs)
            results["postgres"] = "synced"

        if sync_qdrant:
            await qdrant_client_wrapper.insert_documents(enriched_docs)
            results["qdrant"] = "synced"

        return results


document_service = DocumentService()
