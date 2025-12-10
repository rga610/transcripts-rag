"""Document processing: PDF loading, chunking, and embedding."""
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.embeddings import get_embedding_model
from config import CHUNK_SIZE, CHUNK_OVERLAP
from typing import List, Dict
import tempfile
import os


def load_pdf(file_path: str) -> List[str]:
    """Load and extract text from a PDF file."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return [doc.page_content for doc in documents]


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def process_pdf_file(uploaded_file) -> List[Dict]:
    """Process an uploaded PDF file: extract text, chunk, and generate embeddings.
    
    Returns a list of dictionaries with chunk_text, embedding, and metadata.
    """
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Load PDF
        pages = load_pdf(tmp_path)
        full_text = "\n\n".join(pages)
        
        # Chunk text
        chunks = chunk_text(full_text)
        
        # Generate embeddings
        embedding_model = get_embedding_model()
        embeddings = embedding_model.embed_documents(chunks)
        
        # Prepare results
        results = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            results.append({
                "chunk_text": chunk,
                "chunk_index": idx,
                "embedding": embedding,
                "filename": uploaded_file.name
            })
        
        return results
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

