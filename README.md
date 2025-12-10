# Financial Call Transcript RAG Assistant

A lightweight web application that allows users to upload company earnings call transcripts (PDFs) and ask natural-language questions about management commentary, guidance, margins, risks, and other financial topics.

Built with **Streamlit**, **LangChain**, **OpenAI**, and **Supabase** (pgvector).

## Features

- 📄 Upload multiple PDF transcripts
- 💬 Chat interface for asking questions
- 🔍 Semantic search using vector embeddings
- 📊 Accurate answers grounded in the uploaded documents
- 💾 Conversation history stored in Supabase

## Prerequisites

- Python 3.11 or higher
- Supabase account (free tier works)
- OpenAI API key

## Setup Instructions

### 1. Clone and Navigate to Project

```bash
cd transcripts-rag
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Supabase

1. Create a new project at [supabase.com](https://supabase.com)
2. Set up the database schema (choose one method):
   - **Method A (Easiest)**: Go to **SQL Editor** in your Supabase Dashboard, open `database/migrations/20240101000000_initial_schema.sql`, copy the SQL, paste it into SQL Editor, and click **Run**
   - **Method B (CLI)**: Use Supabase CLI via `npx supabase db push` (see `database/README.md` for details)
3. Get your API keys:
   - **Option A (New Format)**: Go to **Settings > API** and use the **Publishable key** (starts with `sb_publishable_`)
   - **Option B (Legacy Format)**: Use the **API Key** shown in **Project Overview** (JWT token starting with `eyJhbGc...`)
   - Both work! The new format is recommended, but the legacy JWT token also works.
   - **Secret key** (optional): In **Settings > API > Secret keys** (starts with `sb_secret_`) - optional for MVP

### 5. Configure Environment Variables

1. Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env  # Windows
   cp .env.example .env    # macOS/Linux
   ```

2. Edit `.env` and fill in your credentials:
   ```
   OPENAI_API_KEY=sk-...
   SUPABASE_URL=https://xxxxx.supabase.co
   SUPABASE_KEY=sb_publishable_...  # OR use the JWT token from Project Overview (eyJhbGc...)
   SUPABASE_SERVICE_KEY=sb_secret_...  # Optional: Your Secret key (for admin operations)
   DATABASE_URL=postgresql://postgres.[ref]:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres
   ```
   
   **Which key to use?**
   - Use the **Publishable key** (`sb_publishable_...`) from Settings > API (recommended)
   - OR use the **JWT token** (`eyJhbGc...`) from Project Overview (legacy format, also works)

   **To get DATABASE_URL:**
   - Go to Supabase Dashboard > Settings > Database
   - Scroll to **Connection string** section
   - **IMPORTANT:** Use the **Session Pooler** connection (not Direct connection) for better compatibility
   - Select **Session Pooler** tab, then **URI** format
   - Copy the connection string (it will look like: `postgresql://postgres.[ref]:[PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres`)
   - Replace `[YOUR-PASSWORD]` or `[PASSWORD]` with your actual database password
   - **Note:** If you don't know your password, reset it in Database Settings
   
   **Note:** The `SUPABASE_SERVICE_KEY` is optional for the MVP - the app works with just the Publishable key and DATABASE_URL.

### 6. Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. **Upload Transcripts**: Use the sidebar to upload one or more PDF earnings call transcripts
2. **Ask Questions**: Type questions in the chat interface, such as:
   - "What did management say about FY2025 margins?"
   - "What are the key risks mentioned?"
   - "What guidance was provided for next quarter?"

3. **View Answers**: The assistant will provide concise, grounded answers based on the uploaded transcripts

## Project Structure

```
transcripts-rag/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .gitignore            # Git ignore file
├── database/
│   ├── migrations/       # Database migrations
│   │   └── 20240101000000_initial_schema.sql
│   └── README.md         # Migration instructions
├── db/                   # Database utilities
│   ├── __init__.py
│   ├── connection.py     # Supabase/Postgres connections
│   └── conversations.py  # Conversation management
└── rag/                  # RAG pipeline
    ├── __init__.py
    ├── embeddings.py     # Embedding generation
    ├── document_processor.py  # PDF processing & chunking
    ├── vector_store.py   # Vector store operations
    └── qa_chain.py       # Question-answering chain
```

## Technical Stack

- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI text-embedding-3-small
- **Vector Database**: Supabase Postgres with pgvector
- **Framework**: LangChain
- **PDF Processing**: PyPDF

## Troubleshooting

### "SUPABASE_URL and SUPABASE_KEY must be set"
- Make sure your `.env` file exists and contains the required variables
- Check that you've activated your virtual environment

### "Extension 'vector' does not exist"
- Run the SQL commands in `database/migrations/20240101000000_initial_schema.sql` in your Supabase SQL Editor
- Make sure pgvector extension is enabled

### "could not translate host name" or DNS resolution errors
- **Use Session Pooler instead of Direct connection** for DATABASE_URL
- Go to Supabase Dashboard > Settings > Database > Connection string
- Select **Session Pooler** tab (not Direct connection)
- Use the connection string from there - it uses `pooler.supabase.com` hostname which is IPv4-compatible

### PDF upload fails
- Check file size (max 10MB)
- Ensure the PDF is not password-protected
- Verify the PDF contains extractable text (not just images)

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions.

**Quick deploy to Streamlit Cloud:**
1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Add secrets (API keys) in Settings
5. Deploy!

## Next Steps

- Add conversation history retrieval
- Implement better error handling
- Add file management (delete uploaded files)
- Improve chunking strategy for better context

## License

This is a learning project. Feel free to use and modify as needed.

