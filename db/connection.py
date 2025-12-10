"""Supabase database connection and utilities."""
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_KEY
import psycopg2
from config import DATABASE_URL


def get_supabase_client() -> Client:
    """Create and return a Supabase client."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_postgres_connection():
    """Get a direct Postgres connection for vector operations."""
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL must be set in environment variables")
    
    # Validate DATABASE_URL format
    if "port" in DATABASE_URL.lower() and ":" not in DATABASE_URL.split("@")[-1].split("/")[0]:
        raise ValueError(
            "DATABASE_URL appears malformed. Expected format: "
            "postgresql://user:password@host:port/database\n"
            f"Current value starts with: {DATABASE_URL[:50]}..."
        )
    
    try:
        return psycopg2.connect(DATABASE_URL)
    except psycopg2.OperationalError as e:
        raise ValueError(
            f"Failed to connect to database. Check your DATABASE_URL.\n"
            f"Expected format: postgresql://postgres:YOUR_PASSWORD@db.xxxxx.supabase.co:5432/postgres\n"
            f"Error: {str(e)}"
        )

