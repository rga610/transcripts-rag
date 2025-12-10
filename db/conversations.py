"""Conversation and message management."""
from supabase import Client
from db.connection import get_supabase_client
from uuid import uuid4
from typing import Optional, List, Dict


def create_conversation(title: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
    """Create a new conversation and return its ID."""
    supabase: Client = get_supabase_client()
    
    data = {
        "title": title,
        "metadata": metadata or {}
    }
    
    result = supabase.table("conversations").insert(data).execute()
    return result.data[0]["id"]


def add_message(conversation_id: str, role: str, content: str) -> Dict:
    """Add a message to a conversation."""
    supabase: Client = get_supabase_client()
    
    data = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content
    }
    
    result = supabase.table("messages").insert(data).execute()
    return result.data[0]


def get_conversation_messages(conversation_id: str) -> List[Dict]:
    """Retrieve all messages for a conversation."""
    supabase: Client = get_supabase_client()
    
    result = supabase.table("messages")\
        .select("*")\
        .eq("conversation_id", conversation_id)\
        .order("created_at")\
        .execute()
    
    return result.data


def get_recent_conversations(limit: int = 10) -> List[Dict]:
    """Get recent conversations."""
    supabase: Client = get_supabase_client()
    
    result = supabase.table("conversations")\
        .select("*")\
        .order("created_at", desc=True)\
        .limit(limit)\
        .execute()
    
    return result.data

