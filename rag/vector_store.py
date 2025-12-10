"""Vector store operations for Supabase pgvector."""
from db.connection import get_postgres_connection
from typing import List, Dict
import psycopg2.extras
from config import TOP_K_RETRIEVAL


def store_document_chunks(chunks: List[Dict], conversation_id: str):
    """Store document chunks with embeddings in Supabase, linked to a conversation.
    
    Args:
        chunks: List of chunk dictionaries with embedding, chunk_text, etc.
        conversation_id: UUID of the conversation these chunks belong to
    """
    conn = get_postgres_connection()
    cur = conn.cursor()
    
    try:
        for chunk_data in chunks:
            # Convert embedding list to string format for pgvector: "[1,2,3,...]"
            embedding = chunk_data["embedding"]
            embedding_str = "[" + ",".join(str(float(x)) for x in embedding) + "]"
            
            # Convert metadata dict to JSONB format using psycopg2's Json adapter
            metadata = chunk_data.get("metadata", {})
            metadata_json = psycopg2.extras.Json(metadata) if isinstance(metadata, dict) else psycopg2.extras.Json({})
            
            cur.execute(
                """
                INSERT INTO document_chunks (filename, chunk_text, chunk_index, embedding, metadata, conversation_id)
                VALUES (%s, %s, %s, %s::vector, %s, %s)
                """,
                (
                    chunk_data["filename"],
                    chunk_data["chunk_text"],
                    chunk_data["chunk_index"],
                    embedding_str,
                    metadata_json,
                    conversation_id
                )
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cur.close()
        conn.close()


def search_similar_chunks(query_embedding: List[float], conversation_id: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
    """Search for similar chunks using cosine similarity, filtered by conversation.
    
    Args:
        query_embedding: The embedding vector to search for
        conversation_id: UUID of the conversation to search within
        top_k: Number of results to return
    
    Returns:
        List of dictionaries with chunk data and similarity scores
    """
    conn = get_postgres_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    try:
        cur.execute(
            """
            SELECT 
                id,
                filename,
                chunk_text,
                chunk_index,
                metadata,
                1 - (embedding <=> %s::vector) as similarity
            FROM document_chunks
            WHERE conversation_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, conversation_id, query_embedding, top_k)
        )
        
        results = cur.fetchall()
        return [dict(row) for row in results]
    finally:
        cur.close()
        conn.close()

