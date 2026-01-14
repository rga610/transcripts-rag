# SOP RAG Assistant

A specialized RAG application for Standard Operating Procedures (SOPs) that preserves document structure, extracts metadata, and provides cited answers.

Built with **Streamlit**, **LangChain**, **OpenAI**, **Qdrant**, and **MarkItDown**.

## Features

- 📄 **Structure-Aware Processing**: Converts PDFs to Markdown, preserving headers, sections, and formatting
- 🏷️ **Metadata Extraction**: Automatically extracts SOP ID, version, department, and last updated date
- 📑 **Header-Based Chunking**: Splits documents by logical sections (not arbitrary character counts)
- 💬 **Chat Interface**: Ask natural language questions about procedures
- 📚 **Cited Answers**: Every answer references the source SOP and section
- 🔒 **Conversation Isolation**: Each conversation only sees its own uploaded files

## How It Works

1. **Upload**: Users upload SOP PDFs via the web interface
2. **Convert**: MarkItDown converts PDFs to structured Markdown (preserves layout)
3. **Extract**: System extracts SOP metadata (ID, version, department)
4. **Chunk**: Documents are split by Markdown headers (`#`, `##`, `###`) to keep logical sections together
5. **Embed**: Each chunk is converted to a vector embedding
6. **Store**: Chunks, embeddings, and metadata are stored in Qdrant
7. **Query**: When users ask questions:
   - Question is embedded
   - Similar chunks are retrieved (filtered by conversation)
   - Context is formatted with citations
   - GPT-4o-mini generates an answer with source references

## Technical Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: Qdrant Cloud
- **Relational DB**: Supabase (PostgreSQL) for conversations
- **PDF Processing**: MarkItDown (layout-aware Markdown conversion)
- **Chunking**: LangChain `MarkdownHeaderTextSplitter`
- **Framework**: LangChain

## Quick Start

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables (see `.env.example`):
   - `OPENAI_API_KEY`
   - `QDRANT_CLUSTER_ENDPOINT`, `QDRANT_API_KEY`, `QDRANT_COLLECTION`
   - `SUPABASE_URL`, `SUPABASE_KEY`, `DATABASE_URL`
3. Run: `streamlit run app.py`

## Project Structure

```
transcripts-rag/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── rag/                        # RAG pipeline
│   ├── document_processor.py  # PDF → Markdown → Chunks → Embeddings
│   ├── chunking/
│   │   └── sop_chunker.py     # Header-based chunking
│   ├── parsers/
│   │   └── markitdown_cli.py  # PDF to Markdown converter
│   ├── sop_metadata.py        # Metadata extraction
│   ├── embeddings.py          # Embedding model
│   ├── vector_store.py        # Qdrant operations
│   └── qa_chain.py            # Question-answering logic
└── db/                         # Database layer
    ├── connection.py          # Supabase client
    └── conversations.py      # Conversation management
```

For detailed code walkthrough, see [explanation.md](explanation.md).
