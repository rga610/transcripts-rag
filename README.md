# Financial Call Transcript RAG Assistant

A web application that allows users to upload company earnings call transcripts (PDFs) and ask natural-language questions about management commentary, guidance, margins, risks, and other financial topics.

Built with **Streamlit**, **LangChain**, **OpenAI**, and **Supabase** (pgvector).

## Features

- 📄 **Upload Transcripts**: Upload multiple PDF earnings call transcripts
- 💬 **Chat Interface**: Ask questions in natural language
- 🔍 **Semantic Search**: Uses vector embeddings for accurate retrieval
- 📊 **Grounded Answers**: Answers are based directly on uploaded documents
- 🔒 **Conversation Isolation**: Each conversation only sees its own uploaded files
- 💾 **Conversation History**: All conversations and messages are stored in Supabase

## How It Works

1. **Upload**: Users upload PDF transcripts via the web interface
2. **Process**: System extracts text, splits into chunks, and generates embeddings
3. **Store**: Chunks and embeddings are stored in Supabase with pgvector
4. **Query**: When users ask questions, the system:
   - Generates an embedding for the question
   - Searches for similar chunks using vector similarity
   - Uses GPT-4o-mini to generate answers based on retrieved context
5. **Isolate**: Each conversation only searches within its own uploaded files

## Technical Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: Supabase Postgres with pgvector
- **Framework**: LangChain
- **PDF Processing**: PyPDF

## Project Structure

```
transcripts-rag/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── database/
│   ├── migrations/        # Database migrations
│   └── README.md          # Migration instructions
├── db/                    # Database utilities
│   ├── connection.py      # Supabase/Postgres connections
│   └── conversations.py   # Conversation management
└── rag/                   # RAG pipeline
    ├── document_processor.py  # PDF processing & chunking
    ├── embeddings.py       # Embedding generation
    ├── qa_chain.py        # Question-answering chain
    └── vector_store.py    # Vector store operations
```
