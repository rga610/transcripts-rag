"""Vector store operations powered by Qdrant Cloud."""

import uuid
from typing import Dict, List

from config import (
    QDRANT_API_KEY,
    QDRANT_CLUSTER_ENDPOINT,
    QDRANT_COLLECTION,
    TOP_K_RETRIEVAL,
)
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    PayloadSchemaType,
    VectorParams,
)

_client: QdrantClient | None = None


def _ensure_qdrant_configured() -> None:
    if not QDRANT_CLUSTER_ENDPOINT or not QDRANT_API_KEY:
        raise RuntimeError(
            "QDRANT_CLUSTER_ENDPOINT and QDRANT_API_KEY must be set to use the vector store."
        )
    if not QDRANT_COLLECTION:
        raise RuntimeError("QDRANT_COLLECTION must be set to identify the collection to use.")


def _get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _ensure_qdrant_configured()
        _client = QdrantClient(
            url=QDRANT_CLUSTER_ENDPOINT,
            api_key=QDRANT_API_KEY,
        )
    return _client


def _collection_exists(client: QdrantClient) -> bool:
    response = client.get_collections()
    return any(coll.name == QDRANT_COLLECTION for coll in response.collections)


def _ensure_collection(client: QdrantClient, vector_size: int) -> None:
    created = False
    if not _collection_exists(client):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        created = True

    # Ensure filterable index on conversation_id for fast query filters
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="conversation_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
    except Exception:
        # Index likely exists already; ignore
        if created:
            # If we just created the collection, re-raise unexpected errors
            raise


def store_document_chunks(chunks: List[Dict], conversation_id: str):
    """Store SOP chunks (metadata + embeddings) in Qdrant."""
    if not chunks:
        return

    vector_size = len(chunks[0]["embedding"])
    client = _get_qdrant_client()
    _ensure_collection(client, vector_size)

    points = []
    for chunk_data in chunks:
        payload = dict(chunk_data["metadata"])
        payload["chunk_text"] = chunk_data["chunk_text"]
        payload["conversation_id"] = str(conversation_id)
        payload["chunk_index"] = chunk_data["chunk_index"]
        
        # Qdrant requires point IDs to be proper UUIDs or unsigned integers
        point_id = str(uuid.uuid4())

        point = PointStruct(
            id=point_id,
            vector=chunk_data["embedding"],
            payload=payload,
        )
        points.append(point)

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )


def search_similar_chunks(
    query_embedding: List[float],
    conversation_id: str,
    top_k: int = TOP_K_RETRIEVAL,
) -> List[Dict]:
    """Retrieve the most similar SOP chunks for the supplied conversation."""
    client = _get_qdrant_client()
    filter_cond = Filter(
        must=[
            FieldCondition(
                key="conversation_id",
                match=MatchValue(value=str(conversation_id)),
            )
        ]
    )

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
        query_filter=filter_cond,
    )

    formatted = []
    for hit in results.points:
        formatted.append(
            {
                "id": hit.id,
                "chunk_text": hit.payload.get("chunk_text"),
                "metadata": hit.payload,
                "similarity": hit.score,
            }
        )
    return formatted

