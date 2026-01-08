"""Configuration settings for the RAG Assistant."""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")

# RAG Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 8  # Increased to get more context for better accuracy

# Qdrant configuration
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sop_chunks")

# File Upload Configuration
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = [".pdf"]

