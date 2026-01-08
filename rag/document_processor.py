"""Document processing: PDF loading, chunking, and embedding."""

import os
import tempfile
from typing import Dict, List

from rag.chunking.sop_chunker import chunk_sop_markdown
from rag.embeddings import get_embedding_model
from rag.parsers.markitdown_cli import MarkItDownError, pdf_to_markdown
from rag.sop_metadata import extract_sop_metadata


def _cleanup_temp_file(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def _build_chunk_payload(
    entry: Dict[str, str], filename: str, sop_metadata: Dict[str, str], index: int
) -> Dict:
    metadata = {
        "source_filename": filename,
        "section_path": entry["section_path"],
        "section_title": entry["section_title"],
        "chunk_role": entry["chunk_role"],
        "headers": entry["headers"],
        "sop_metadata": sop_metadata,
    }
    return {
        "chunk_text": entry["text"],
        "chunk_index": index,
        "filename": filename,
        "metadata": metadata,
    }


def process_pdf_file(uploaded_file) -> List[Dict]:
    """Convert SOP PDF → Markdown → structured chunks + embeddings."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        markdown_text = pdf_to_markdown(tmp_path)
        sop_metadata = extract_sop_metadata(markdown_text)
        chunk_entries = chunk_sop_markdown(markdown_text)

        if not chunk_entries:
            return []

        embedding_model = get_embedding_model()
        texts = [entry["text"] for entry in chunk_entries]
        embeddings = embedding_model.embed_documents(texts)

        results = []
        for idx, (entry, embedding) in enumerate(zip(chunk_entries, embeddings)):
            chunk_data = _build_chunk_payload(entry, uploaded_file.name, sop_metadata, idx)
            chunk_data["embedding"] = embedding
            results.append(chunk_data)

        return results
    except MarkItDownError as exc:
        raise RuntimeError("Failed to extract Markdown from PDF") from exc
    finally:
        _cleanup_temp_file(tmp_path)

