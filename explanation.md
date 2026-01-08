# SOP RAG Assistant - Code Walkthrough

## 1. Project Introduction

### What is this?

The **SOP RAG Assistant** is a Retrieval-Augmented Generation (RAG) application specifically designed for **Standard Operating Procedures (SOPs)**. It allows users to:

- Upload SOP documents (PDFs)
- Ask natural language questions about procedures
- Get precise, cited answers based on the uploaded documents
- Maintain conversation context and document isolation

### Why SOPs Specifically?

Unlike generic RAG systems, this application is optimized for structured procedural documents:
- **Preserves document structure**: Uses Markdown header-based chunking to keep logical sections together
- **Extracts metadata**: Captures SOP ID, version, department, last updated date
- **Cites sources**: Every answer references the specific SOP and section it came from
- **Maintains context**: Related steps, warnings, and procedures stay together

---

## 2. High-Level Overview

### The RAG Pipeline

```
PDF Upload
    ↓
[MarkItDown] PDF → Markdown (preserves structure)
    ↓
[Metadata Extraction] Extract SOP ID, version, department, etc.
    ↓
[Chunking] Split by headers (# Header 1, ## Header 2, ### Header 3)
    ↓
[Embedding] Convert each chunk to vector embeddings
    ↓
[Qdrant Storage] Store chunks + embeddings + metadata in vector database
    ↓
[Query Time] User asks question
    ↓
[Embed Query] Convert question to embedding
    ↓
[Vector Search] Find similar chunks (filtered by conversation)
    ↓
[Context Formatting] Format retrieved chunks with citations
    ↓
[LLM Generation] GPT-4o-mini generates answer with citations
```

### Key Technologies

- **Frontend**: Streamlit (Python web framework)
- **PDF Processing**: MarkItDown (Microsoft's layout-aware PDF converter)
- **Chunking**: LangChain's `MarkdownHeaderTextSplitter`
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector Database**: Qdrant Cloud
- **LLM**: OpenAI `gpt-4o-mini`
- **Relational DB**: Supabase (PostgreSQL) for conversations

---

## 3. Project Structure

```
transcripts-rag/
├── app.py                          # Main Streamlit UI and orchestration
├── config.py                       # Environment variables and configuration
├── requirements.txt                # Python dependencies
│
├── rag/                            # Core RAG pipeline
│   ├── document_processor.py      # PDF → Markdown → Chunks → Embeddings
│   ├── chunking/
│   │   └── sop_chunker.py         # Structured chunking by headers
│   ├── parsers/
│   │   └── markitdown_cli.py      # PDF to Markdown converter wrapper
│   ├── sop_metadata.py            # Extract SOP metadata (ID, version, etc.)
│   ├── embeddings.py              # Embedding model initialization
│   ├── vector_store.py            # Qdrant operations (store/retrieve)
│   └── qa_chain.py                # Question-answering logic
│
└── db/                             # Database layer
    ├── connection.py              # Supabase client setup
    └── conversations.py           # Conversation/message CRUD
```

---

## 4. Critical Files Deep Dive

### 4.1 `app.py` - The Orchestrator

**Purpose**: Main Streamlit application that ties everything together.

**Key Responsibilities**:
1. UI rendering (file upload, chat interface)
2. Session state management (conversation ID, uploaded files, messages)
3. Orchestrating the document processing and query pipeline

**Important Code Sections**:

```python
# Session State Initialization (Lines 17-25)
# Streamlit maintains state between reruns using session_state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None
```

**Why this matters**: Streamlit reruns the entire script on every interaction. Without `session_state`, we'd lose our conversation context.

```python
# Document Processing Flow (Lines 72-84)
for file in new_files:
    try:
        # Step 1: Process PDF → Markdown → Chunks + Embeddings
        chunks = process_pdf_file(file)
        
        # Step 2: Store in vector database (linked to conversation)
        store_document_chunks(chunks, st.session_state.conversation_id)
        
        st.session_state.uploaded_files.append(file.name)
        st.success(f"✅ Processed: {file.name} ({len(chunks)} chunks)")
```

**What happens here**:
1. `process_pdf_file()` converts PDF to structured chunks with embeddings
2. `store_document_chunks()` stores them in Qdrant, tagged with `conversation_id`
3. The `conversation_id` ensures documents are isolated per conversation

```python
# Query Flow (Lines 101-129)
if prompt := st.chat_input("Ask a question about these SOPs..."):
    # Generate response (scoped to this conversation)
    response = answer_question(prompt, st.session_state.conversation_id)
```

**The magic**: `answer_question()` only searches chunks belonging to this `conversation_id`, ensuring conversation isolation.

---

### 4.2 `rag/document_processor.py` - The Processing Pipeline

**Purpose**: Orchestrates the entire document processing pipeline from PDF to indexed chunks.

**Key Function**: `process_pdf_file(uploaded_file)`

**Step-by-Step Breakdown**:

```python
# Step 1: Save uploaded file temporarily (Lines 39-41)
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
    tmp_file.write(uploaded_file.getvalue())
    tmp_path = tmp_file.name
```

**Why temporary file?**: MarkItDown CLI needs a file path, not an in-memory object. We clean it up in the `finally` block.

```python
# Step 2: PDF → Markdown (Line 44)
markdown_text = pdf_to_markdown(tmp_path)
```

**What `pdf_to_markdown` does**: Calls the `markitdown` CLI tool via `subprocess`. This preserves:
- Document structure (headers, sections)
- Formatting (tables, lists)
- Layout information (where text appears on page)

**Why not PyPDFLoader?**: PyPDFLoader flattens PDFs into plain text, losing structure. SOPs need structure preserved.

```python
# Step 3: Extract SOP metadata (Line 45)
sop_metadata = extract_sop_metadata(markdown_text)
```

**What gets extracted**: Uses regex patterns to find:
- SOP ID (e.g., "SOP-IT-04")
- Version (e.g., "Version 2.1")
- Department (e.g., "IT Department")
- Last Updated date

**Why this matters**: This metadata is stored with each chunk, enabling citations like "SOP-IT-04, Version 2.1, Section 3.2".

```python
# Step 4: Structured chunking (Line 46)
chunk_entries = chunk_sop_markdown(markdown_text)
```

**This is the heart of SOP-aware chunking** - explained in detail in section 4.3.

```python
# Step 5: Generate embeddings (Lines 51-53)
embedding_model = get_embedding_model()
texts = [entry["text"] for entry in chunk_entries]
embeddings = embedding_model.embed_documents(texts)
```

**What happens**: OpenAI's `text-embedding-3-small` converts each chunk's text into a 1536-dimensional vector. Similar chunks (semantically) will have similar vectors.

**Why batch embedding?**: `embed_documents()` processes all chunks in one API call, which is faster and cheaper than individual calls.

```python
# Step 6: Assemble final payload (Lines 55-59)
for idx, (entry, embedding) in enumerate(zip(chunk_entries, embeddings)):
    chunk_data = _build_chunk_payload(entry, uploaded_file.name, sop_metadata, idx)
    chunk_data["embedding"] = embedding
    results.append(chunk_data)
```

**Final structure of each chunk**:
```python
{
    "chunk_text": "Actual text content of the chunk",
    "chunk_index": 0,  # Position in document
    "filename": "sop.pdf",
    "embedding": [0.123, -0.456, ...],  # 1536-dimensional vector
    "metadata": {
        "source_filename": "sop.pdf",
        "section_path": "Section 1 > 1.1 > 1.1.1",  # Hierarchy
        "section_title": "Safety Procedures",
        "chunk_role": "parent_section",  # or "child_chunk"
        "headers": {"Header 1": "Section 1", "Header 2": "1.1"},
        "sop_metadata": {
            "sop_id": "SOP-IT-04",
            "version": "2.1",
            "department": "IT",
            "last_updated": "2024-01-15"
        }
    }
}
```

---

### 4.3 `rag/chunking/sop_chunker.py` - Structured Chunking

**Purpose**: Split Markdown documents into logical chunks based on headers, preserving document structure.

**Key Function**: `chunk_sop_markdown(markdown_text)`

**Why header-based chunking?**:
- SOPs are hierarchical (Section 1.0 → 1.1 → 1.1.1)
- Related information (e.g., a warning and the step it protects) should stay together
- Generic chunking (e.g., "every 1000 characters") can split logical units

**The Algorithm**:

```python
# Step 1: Primary splitter - by headers (Lines 36-39)
splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),      # Top-level sections
        ("##", "Header 2"),     # Subsections
        ("###", "Header 3"),    # Sub-subsections
    ],
    strip_headers=False,  # Keep headers in the text
)
```

**What `MarkdownHeaderTextSplitter` does**:
- Splits the document at each header boundary
- Preserves header hierarchy in metadata
- Example: For Markdown like:
  ```markdown
  # Section 1: Safety
  Always wear PPE.
  ## Subsection 1.1: Gloves
  Use nitrile gloves.
  ```
  It creates chunks like:
  - Chunk 1: "Section 1: Safety\nAlways wear PPE." (metadata: `{"Header 1": "Section 1: Safety"}`)
  - Chunk 2: "Subsection 1.1: Gloves\nUse nitrile gloves." (metadata: `{"Header 1": "Section 1: Safety", "Header 2": "Subsection 1.1: Gloves"}`)

```python
# Step 2: Fallback splitter for oversized sections (Lines 44-48)
fallback_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
```

**Why a fallback?**: Sometimes a single section is very long (e.g., a detailed procedure with 5000 characters). We still split it, but:
- We split at the section level first
- Only split further if a section exceeds 1200 characters (1000 + 200 overlap)
- Overlap ensures context isn't lost at boundaries

**The Chunking Logic (Lines 53-79)**:

```python
for document in splitter.split_text(markdown_text):
    text = document.page_content.strip()
    headers = dict(document.metadata or {})
    section_path = _build_section_path(headers)  # "Section 1 > 1.1 > 1.1.1"
    section_title = _get_section_title(headers)  # "1.1.1" (most specific)
    
    if len(text) > fallback_threshold:  # Section too large?
        # Split into smaller chunks (child chunks)
        for child_text in fallback_splitter.split_text(text):
            push_chunk(child_text, "child_chunk")
    else:
        # Keep entire section as one chunk (parent section)
        push_chunk(text, "parent_section")
```

**Parent vs Child Chunks**:
- **Parent section**: A complete, logically-contained section (e.g., "Section 3.2: Emergency Shutdown")
- **Child chunk**: A fragment of a very long section (still marked with parent section path)

**Why track this?**: During retrieval, if we find a child chunk, we know which parent section it belongs to, enabling better context.

**Helper Functions**:

```python
def _build_section_path(metadata: Dict[str, str]) -> str:
    # Converts {"Header 1": "Section 1", "Header 2": "1.1"} 
    # → "Section 1 > 1.1"
    parts = [metadata[level] for level in HEADER_ORDER if metadata.get(level)]
    return " > ".join(parts) if parts else "Full Document"
```

This creates a breadcrumb trail for citations: `"SOP-IT-04 § Section 1 > 1.1"`

---

### 4.4 `rag/vector_store.py` - Qdrant Operations

**Purpose**: Interface with Qdrant Cloud for storing and retrieving document chunks.

**Key Concepts**:
- **Collection**: Like a table in SQL, contains all chunks for all conversations
- **Point**: A single chunk with its embedding vector and metadata
- **Payload**: Metadata stored with each point (not the vector)
- **Filter**: Query constraint (e.g., "only chunks from conversation X")

**Initialization (Lines 35-43)**:

```python
_client: QdrantClient | None = None  # Singleton pattern

def _get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _ensure_qdrant_configured()  # Check env vars
        _client = QdrantClient(
            url=QDRANT_CLUSTER_ENDPOINT,  # e.g., "https://xxx.us-east4-0.gcp.cloud.qdrant.io"
            api_key=QDRANT_API_KEY,
        )
    return _client
```

**Why singleton?**: Reusing the client connection is more efficient than creating new connections for each operation.

**Collection Setup (Lines 51-71)**:

```python
def _ensure_collection(client: QdrantClient, vector_size: int) -> None:
    # Create collection if it doesn't exist
    if not _collection_exists(client):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=vector_size,  # 1536 for text-embedding-3-small
                distance=Distance.COSINE,  # Similarity metric
            ),
        )
    
    # Create index on conversation_id for fast filtering
    try:
        client.create_payload_index(
            collection_name=QDRANT_COLLECTION,
            field_name="conversation_id",
            field_schema=PayloadSchemaType.KEYWORD,  # Exact match index
        )
    except Exception:
        # Index already exists; ignore
        pass
```

**Why index `conversation_id`?**: Without an index, Qdrant must scan all points to filter by `conversation_id`. With an index, filtering is O(log n) instead of O(n).

**Storing Chunks (Lines 74-103)**:

```python
def store_document_chunks(chunks: List[Dict], conversation_id: str):
    vector_size = len(chunks[0]["embedding"])  # e.g., 1536
    client = _get_qdrant_client()
    _ensure_collection(client, vector_size)
    
    points = []
    for chunk_data in chunks:
        payload = dict(chunk_data["metadata"])
        payload["chunk_text"] = chunk_data["chunk_text"]
        payload["conversation_id"] = str(conversation_id)  # For filtering
        payload["chunk_index"] = chunk_data["chunk_index"]
        
        # Generate unique UUID for point ID
        point_id = str(uuid.uuid4())
        
        point = PointStruct(
            id=point_id,                    # Unique identifier (UUID)
            vector=chunk_data["embedding"], # 1536-dimensional vector
            payload=payload,                # All metadata
        )
        points.append(point)
    
    # Batch insert (more efficient than individual inserts)
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )
```

**Important Details**:

1. **Point ID as UUID**: Qdrant requires point IDs to be UUIDs or unsigned integers. We generate UUIDs to ensure uniqueness.

2. **Why `upsert` not `insert`?**: `upsert` updates if the point exists, inserts if not. If a user re-uploads the same file, we don't duplicate.

3. **Payload vs Vector**: 
   - **Vector** (1536 floats): Used for similarity search
   - **Payload** (dict): Stored metadata, used for filtering and retrieval

**Searching Chunks (Lines 106-140)**:

```python
def search_similar_chunks(
    query_embedding: List[float],  # Question converted to embedding
    conversation_id: str,          # Only search this conversation's chunks
    top_k: int = 8,                # Return top 8 most similar
) -> List[Dict]:
    client = _get_qdrant_client()
    
    # Build filter: only chunks from this conversation
    filter_cond = Filter(
        must=[
            FieldCondition(
                key="conversation_id",
                match=MatchValue(value=str(conversation_id)),
            )
        ]
    )
    
    # Query Qdrant
    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_embedding,        # Search vector
        limit=top_k,                  # Top 8 results
        with_payload=True,            # Return metadata
        query_filter=filter_cond,     # Filter by conversation_id
    )
    
    # Format results
    formatted = []
    for hit in results.points:
        formatted.append({
            "id": hit.id,
            "chunk_text": hit.payload.get("chunk_text"),
            "metadata": hit.payload,      # All metadata (section_path, sop_metadata, etc.)
            "similarity": hit.score,      # Cosine similarity score (0-1)
        })
    return formatted
```

**How Vector Search Works**:

1. **Query Embedding**: User's question "When should incremental backups be performed?" → [0.123, -0.456, ...] (1536 dimensions)

2. **Similarity Calculation**: Qdrant calculates cosine similarity between query embedding and all chunk embeddings:
   - Cosine similarity = `dot(query, chunk) / (|query| * |chunk|)`
   - Range: -1 (opposite) to 1 (identical)
   - Higher score = more similar

3. **Filtering**: Only chunks where `payload.conversation_id == conversation_id` are considered

4. **Ranking**: Results sorted by similarity score (highest first)

5. **Top-K**: Return top 8 most similar chunks

**Why top 8?**: Balance between:
- Too few (2-3): Missing context, incomplete answers
- Too many (20+): Expensive, noisy context, slower LLM processing

---

### 4.5 `rag/qa_chain.py` - Question Answering

**Purpose**: Orchestrates the RAG query pipeline: embed question → search → format → generate answer.

**Key Function**: `answer_question(question: str, conversation_id: str)`

**The Pipeline**:

```python
# Step 1: Embed the question (Lines 51-52)
embedding_model = get_embedding_model()
query_embedding = embedding_model.embed_query(question)
```

**Note**: `embed_query()` for single strings, `embed_documents()` for batches.

```python
# Step 2: Search for similar chunks (Line 54)
retrieved_chunks = search_similar_chunks(query_embedding, conversation_id)
```

This returns the top 8 most similar chunks, already filtered by `conversation_id`.

```python
# Step 3: Format context with citations (Line 58)
context = _format_context_parts(retrieved_chunks)
```

**Formatting Logic (Lines 27-46)**:

```python
def _format_context_parts(chunks: List[Dict]) -> str:
    parts = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        citation = _format_citation(metadata)  # "sop.pdf § Section 1 > 1.1"
        sop_meta = metadata.get("sop_metadata") or {}
        
        # Build metadata trail: "SOP ID: SOP-IT-04; Version: 2.1"
        meta_trail = []
        for key in ("sop_id", "version", "department", "last_updated"):
            if sop_meta.get(key):
                meta_trail.append(f"{key.replace('_', ' ').title()}: {sop_meta[key]}")
        
        # Format: "[citation] (metadata)\nchunk text"
        prefix = f"[{citation}]"
        if meta_trail:
            prefix += " (" + "; ".join(meta_trail) + ")"
        
        chunk_text = chunk.get("chunk_text", "").strip()
        if chunk_text:
            parts.append(f"{prefix}\n{chunk_text}")
    
    return "\n\n---\n\n".join(parts)  # Separate chunks with dividers
```

**Example Output**:
```
[sop.pdf § Section 3 > 3.2] (SOP ID: SOP-IT-04; Version: 2.1; Department: IT)
Incremental backups should be performed daily at 2:00 AM. Ensure the backup storage has sufficient space.

---

[sop.pdf § Section 3 > 3.3] (SOP ID: SOP-IT-04; Version: 2.1)
Full backups are scheduled weekly on Sundays. Incremental backups run between full backups.
```

**Why format like this?**: The LLM sees:
1. The source citation (`[sop.pdf § Section 3 > 3.2]`)
2. SOP metadata (for verification)
3. The actual content

This enables the LLM to cite sources accurately.

**Step 4: LLM Generation (Lines 60-85)**:

```python
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an SOP specialist assistant. Use only the provided SOP context to answer questions.
- Be precise and honest; do not fabricate details that are not in the context.
- Always include a Citations line that references the source filename and section path or title for the statements you use.
- If the context does not contain enough information, explain that you are unable to answer rather than guessing."""),
    ("human", """Context from SOPs:
{context}

Question: {question}

Answer clearly, cite the SOP section(s) you rely on, and mention if the answer is not available from the context."""),
])

messages = prompt_template.format_messages(context=context, question=question)
llm = get_llm()  # ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke(messages)
return response.content
```

**Prompt Engineering**:

1. **System Prompt**: Sets the LLM's role and constraints:
   - Only use provided context (no hallucination)
   - Must cite sources
   - Be honest about missing information

2. **Human Prompt**: Provides:
   - Formatted context (with citations)
   - User's question

3. **Temperature=0**: Deterministic output (same question → same answer, helpful for debugging)

**Why this prompt structure?**: 
- System prompt: "What kind of assistant are you?"
- Human prompt: "Here's the data, answer this question"

This is OpenAI's chat format and works well with GPT-4o-mini.

---

### 4.6 `rag/parsers/markitdown_cli.py` - PDF Conversion

**Purpose**: Wrapper around MarkItDown CLI for PDF → Markdown conversion.

**Why MarkItDown?**:
- Preserves document structure (headers, tables, lists)
- Layout-aware (understands where text is on the page)
- Handles complex PDFs better than PyPDFLoader
- Microsoft-maintained, actively developed

**Implementation**:

```python
def pdf_to_markdown(file_path: Union[str, Path]) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    
    # MarkItDown CLI outputs to stdout by default
    command = ["markitdown", str(path)]
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,  # Capture stdout and stderr
            text=True,            # Return as string, not bytes
            check=True,           # Raise exception if command fails
            encoding="utf-8",
        )
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.strip() or exc.stdout.strip()
        raise MarkItDownError(f"MarkItDown failed for {path.name}: {error_message}") from exc
    
    markdown_text = result.stdout.strip()
    if not markdown_text:
        raise MarkItDownError(f"MarkItDown returned empty content for {path.name}")
    
    return markdown_text
```

**Why subprocess instead of Python library?**:
- MarkItDown is primarily a CLI tool
- Using subprocess is simpler than trying to use internal Python APIs
- More reliable (CLI is the "official" interface)

**Error Handling**:
- `FileNotFoundError`: PDF file doesn't exist
- `CalledProcessError`: MarkItDown CLI failed (invalid PDF, missing dependencies, etc.)
- `MarkItDownError`: Custom exception for better error messages

---

### 4.7 `rag/sop_metadata.py` - Metadata Extraction

**Purpose**: Extract structured metadata from SOP Markdown (SOP ID, version, department, last updated).

**Why Regex?**: 
- SOPs aren't structured data (like JSON)
- They're semi-structured text documents
- Regex patterns are flexible enough to catch variations

**Patterns**:

```python
_PATTERN_MAP: Dict[str, str] = {
    "sop_id": r"\b(SOP[- ]?[A-Z0-9]+)\b",
    # Matches: "SOP-IT-04", "SOP IT-04", "SOPHR001"
    
    "version": r"\b(?:Version|Rev(?:ision)?|v)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)*)\b",
    # Matches: "Version 2.1", "Rev 1.0", "v 3.2.1", "Version: 2.0"
    
    "last_updated": r"(?:Last Updated|Last Revision)\s*[:\-]\s*([A-Za-z0-9 ,./-]+)",
    # Matches: "Last Updated: 2024-01-15", "Last Revision - Jan 15, 2024"
    
    "department": r"\bDepartment\s*[:\-]\s*([A-Za-z &/]+)\b",
    # Matches: "Department: IT", "Department - Human Resources"
}
```

**Extraction Logic**:

```python
def extract_sop_metadata(markdown_text: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    if not markdown_text:
        return metadata
    
    for key, pattern in _PATTERN_MAP.items():
        match = re.search(pattern, markdown_text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()  # Extract captured group
            if value:
                metadata[key] = value
    
    return metadata
```

**How it works**:
1. Searches entire Markdown text for each pattern
2. Uses `re.IGNORECASE` to match "Version", "version", "VERSION"
3. Captures the value (group 1) after the label
4. Returns dictionary of found metadata

**Limitations**:
- If SOPs use non-standard formats, patterns may not match
- Can be improved with LLM-based extraction for complex cases
- For now, regex is fast and sufficient for most SOPs

---

### 4.8 `db/conversations.py` - Conversation Management

**Purpose**: CRUD operations for conversations and messages in Supabase.

**Why Supabase?**:
- PostgreSQL database (familiar SQL)
- Real-time capabilities (if needed later)
- Easy authentication (if needed later)
- Simple REST API via Python client

**Key Functions**:

```python
def create_conversation(title: Optional[str] = None, metadata: Optional[Dict] = None) -> str:
    """Create a new conversation and return its ID (UUID)."""
    supabase: Client = get_supabase_client()
    
    data = {"title": title, "metadata": metadata or {}}
    result = supabase.table("conversations").insert(data).execute()
    return result.data[0]["id"]  # Return UUID
```

**What happens**: Inserts a new row into `conversations` table, Supabase auto-generates UUID, returns it.

```python
def add_message(conversation_id: str, role: str, content: str) -> Dict:
    """Add a message to a conversation."""
    supabase: Client = get_supabase_client()
    
    data = {
        "conversation_id": conversation_id,
        "role": role,      # "user" or "assistant"
        "content": content
    }
    result = supabase.table("messages").insert(data).execute()
    return result.data[0]
```

**Why store messages?**:
- Conversation history (can load past conversations)
- Analytics (what questions are asked most?)
- Debugging (what did the system actually say?)

**Database Schema** (from migrations):
```sql
-- conversations table
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- messages table
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID REFERENCES conversations(id),
    role TEXT,  -- "user" or "assistant"
    content TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

### 4.9 `config.py` - Configuration Management

**Purpose**: Centralized configuration from environment variables.

**Why environment variables?**:
- Security: API keys not in code
- Flexibility: Different configs for dev/staging/prod
- Best practice: 12-factor app methodology

```python
load_dotenv()  # Load .env file if it exists

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Default value
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Qdrant Configuration
QDRANT_CLUSTER_ENDPOINT = os.getenv("QDRANT_CLUSTER_ENDPOINT")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "sop_chunks")  # Default

# RAG Configuration
TOP_K_RETRIEVAL = 8  # Number of chunks to retrieve
```

**Required `.env` file**:
```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

QDRANT_CLUSTER_ENDPOINT=https://xxx.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=xxx
QDRANT_COLLECTION=sop_chunks

SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=xxx
DATABASE_URL=postgresql://...
```

**Why `.env` file?**: 
- Local development: Store secrets locally (not in git)
- Production: Set environment variables in hosting platform (Streamlit Cloud, etc.)

---

## 5. Data Flow Examples

### Example 1: Uploading a PDF

**User Action**: Uploads `sop-it-backup.pdf`

**Step 1 - UI (`app.py:76`)**:
```python
chunks = process_pdf_file(file)
```
Triggers document processing pipeline.

**Step 2 - PDF Conversion (`document_processor.py:44`)**:
```python
markdown_text = pdf_to_markdown(tmp_path)
```
Calls MarkItDown CLI:
- Input: `/tmp/sop-it-backup.pdf`
- Output: Markdown string preserving structure
```markdown
# SOP-IT-04: Backup Procedures
Version: 2.1
Department: IT
Last Updated: 2024-01-15

## Section 1: Daily Backups
Incremental backups should be performed daily...
```

**Step 3 - Metadata Extraction (`document_processor.py:45`)**:
```python
sop_metadata = extract_sop_metadata(markdown_text)
```
Extracts:
```python
{
    "sop_id": "SOP-IT-04",
    "version": "2.1",
    "department": "IT",
    "last_updated": "2024-01-15"
}
```

**Step 4 - Chunking (`document_processor.py:46`)**:
```python
chunk_entries = chunk_sop_markdown(markdown_text)
```
Creates chunks:
```python
[
    {
        "text": "# SOP-IT-04: Backup Procedures\nVersion: 2.1\n...",
        "section_path": "SOP-IT-04: Backup Procedures",
        "section_title": "SOP-IT-04: Backup Procedures",
        "headers": {"Header 1": "SOP-IT-04: Backup Procedures"},
        "chunk_role": "parent_section"
    },
    {
        "text": "## Section 1: Daily Backups\nIncremental backups should be performed daily...",
        "section_path": "SOP-IT-04: Backup Procedures > Section 1: Daily Backups",
        "section_title": "Section 1: Daily Backups",
        "headers": {"Header 1": "...", "Header 2": "Section 1: Daily Backups"},
        "chunk_role": "parent_section"
    }
]
```

**Step 5 - Embedding (`document_processor.py:51-53`)**:
```python
embeddings = embedding_model.embed_documents(texts)
```
Converts each chunk text to 1536-dimensional vector:
- Chunk 1: `[0.123, -0.456, 0.789, ...]` (1536 floats)
- Chunk 2: `[0.234, -0.567, 0.890, ...]`

**Step 6 - Storage (`app.py:79`)**:
```python
store_document_chunks(chunks, conversation_id="abc-123-def")
```
Stores in Qdrant:
- Point 1: ID=`uuid1`, vector=`[0.123, ...]`, payload={chunk_text, metadata, conversation_id: "abc-123-def"}
- Point 2: ID=`uuid2`, vector=`[0.234, ...]`, payload={chunk_text, metadata, conversation_id: "abc-123-def"}

**Result**: Document is indexed and searchable within conversation "abc-123-def".

---

### Example 2: Asking a Question

**User Action**: Asks "When should incremental backups be performed?"

**Step 1 - UI (`app.py:118`)**:
```python
response = answer_question(prompt, conversation_id)
```

**Step 2 - Query Embedding (`qa_chain.py:51-52`)**:
```python
query_embedding = embedding_model.embed_query(question)
```
Converts question to vector: `[0.345, -0.678, 0.901, ...]`

**Step 3 - Vector Search (`qa_chain.py:54`)**:
```python
retrieved_chunks = search_similar_chunks(query_embedding, conversation_id)
```
Qdrant:
1. Filters: `payload.conversation_id == "abc-123-def"`
2. Calculates cosine similarity for all matching chunks
3. Returns top 8 most similar:
   ```python
   [
       {
           "chunk_text": "## Section 1: Daily Backups\nIncremental backups should be performed daily at 2:00 AM...",
           "metadata": {
               "section_path": "SOP-IT-04: Backup Procedures > Section 1: Daily Backups",
               "sop_metadata": {"sop_id": "SOP-IT-04", "version": "2.1"}
           },
           "similarity": 0.89  # Very similar!
       },
       # ... 7 more chunks
   ]
   ```

**Step 4 - Context Formatting (`qa_chain.py:58`)**:
```python
context = _format_context_parts(retrieved_chunks)
```
Formats as:
```
[sop-it-backup.pdf § SOP-IT-04: Backup Procedures > Section 1: Daily Backups] (SOP ID: SOP-IT-04; Version: 2.1; Department: IT)
Incremental backups should be performed daily at 2:00 AM. Ensure the backup storage has sufficient space.

---

[... more chunks ...]
```

**Step 5 - LLM Generation (`qa_chain.py:83-85`)**:
```python
response = llm.invoke(messages)
```
LLM receives:
- System: "You are an SOP specialist assistant. Always cite sources..."
- Human: "Context: [...formatted chunks...]\n\nQuestion: When should incremental backups be performed?"

LLM generates:
```
Incremental backups should be performed daily at 2:00 AM.

**Citations:**
- sop-it-backup.pdf § SOP-IT-04: Backup Procedures > Section 1: Daily Backups (SOP-IT-04, Version 2.1)
```

**Step 6 - Display (`app.py:119`)**:
```python
st.markdown(response)
```
Shows answer in UI.

---

## 6. Key Design Decisions

### Why Qdrant instead of Supabase pgvector?

**Qdrant Advantages**:
- Purpose-built for vector search (faster, optimized)
- Better API for filtering and hybrid search
- Scales better for large document collections
- Cloud-hosted (less infrastructure management)

**Trade-off**: 
- Additional service to manage
- Separate from conversation storage (Supabase)

### Why Header-Based Chunking?

**Problem**: Generic chunking (every N characters) splits logical units.

**Example**:
- Generic: Splits "Step 5: Press button" from "Step 6: Verify light"
- Header-based: Keeps entire "Section 3: Procedures" together

**Benefit**: Related information stays together, improving retrieval quality.

### Why Separate Conversations?

**Problem**: If all documents are in one pool, user A's questions might return chunks from user B's documents.

**Solution**: `conversation_id` filter ensures:
- Each conversation only sees its own documents
- Multi-tenant safe (if extended)
- Better isolation for testing/debugging

### Why MarkItDown instead of PyPDFLoader?

**PyPDFLoader Issues**:
- Flattens structure (loses headers, sections)
- Poor handling of tables and formatted text
- No layout awareness

**MarkItDown Benefits**:
- Preserves Markdown structure
- Layout-aware parsing
- Better table extraction
- Microsoft-maintained

---

## 7. Common Questions

### Q: What if a PDF has no headers?

**A**: The `MarkdownHeaderTextSplitter` will create one chunk for the entire document (marked as "Full Document"). The fallback splitter will still split if it's too large (>1200 chars).

### Q: How are duplicate chunks handled?

**A**: Each chunk gets a unique UUID. If the same document is uploaded twice:
- New UUIDs are generated
- Chunks are stored again (duplicates exist)
- Filtering by `conversation_id` ensures only relevant chunks are returned
- Future improvement: Add deduplication logic (hash chunk text, check before insert)

### Q: What happens if Qdrant is down?

**A**: 
- Storage fails: User sees error message
- Search fails: User sees "no relevant information found"
- No graceful degradation (future improvement: cache embeddings locally)

### Q: Can this handle images in PDFs?

**A**: 
- MarkItDown can extract images, but current implementation only uses text
- Images are not embedded or indexed
- Future improvement: Use vision models to describe images, embed descriptions

### Q: How are citations generated?

**A**: 
1. Metadata extraction finds SOP ID, version
2. Header chunking creates section paths
3. Context formatting combines: `[filename § section_path] (SOP ID: X; Version: Y)`
4. LLM is instructed to cite sources
5. LLM includes citations in response

### Q: What's the maximum document size?

**A**:
- Streamlit file uploader: 200MB (configurable)
- Practical limit: Depends on MarkItDown processing time
- Chunking: Handles any size (splits large sections)
- Embedding: Batch processing handles large documents
- No hard limit, but very large PDFs (>100 pages) may be slow

---

## 8. Future Improvements

### Phase 3: Hybrid Search
- **Current**: Pure vector search (semantic similarity)
- **Future**: Add keyword search (BM25) + re-ranking (cross-encoder)
- **Benefit**: Better handling of exact matches (error codes, acronyms)

### Phase 4: Re-ranking
- **Current**: Top-K retrieval by cosine similarity
- **Future**: Re-rank with cross-encoder model (more accurate relevance)
- **Benefit**: Pushes most relevant chunks to top, filters noise

### Other Improvements:
- **Deduplication**: Hash chunk text, avoid storing duplicates
- **Parent Document Retriever**: Return full parent section when child chunk matches
- **Conversation History**: Use previous Q&A pairs for better context
- **Multi-language**: Support non-English SOPs
- **Image Processing**: Extract and index images from PDFs

---

## 9. Troubleshooting Guide

### Error: "MarkItDown CLI not found"
**Cause**: `markitdown` not installed or not in PATH
**Fix**: `pip install markitdown[all]` and add to PATH

### Error: "QdrantClient object has no attribute 'search'"
**Cause**: Using wrong Qdrant client API version
**Fix**: Use `query_points()` instead of `search()` (already fixed)

### Error: "Index required but not found for conversation_id"
**Cause**: Payload index not created
**Fix**: `_ensure_collection()` now creates index automatically

### Error: "Failed to extract Markdown from PDF"
**Cause**: Invalid PDF, corrupted file, or MarkItDown error
**Fix**: Check PDF is valid, try different PDF, check MarkItDown logs

### Error: "No relevant information found"
**Causes**:
1. No documents uploaded
2. Question too different from document content
3. Filter too restrictive (conversation_id mismatch)
**Fix**: Verify documents uploaded, check conversation_id, try rephrasing question

---

## 10. Conclusion

This SOP RAG Assistant is built with a focus on:
- **Structure preservation**: Header-based chunking keeps logical sections together
- **Citation accuracy**: Metadata extraction and formatting enable precise citations
- **Conversation isolation**: Each conversation only sees its own documents
- **Extensibility**: Clear separation of concerns enables future improvements

The codebase is designed to be:
- **Understandable**: Clear function names, docstrings, single responsibility
- **Maintainable**: Modular structure, easy to extend
- **Debuggable**: Error messages, logging, structured data

**No black boxes** - every component is explained, every decision is justified, every line of code has a purpose.

---

*End of Code Walkthrough*
