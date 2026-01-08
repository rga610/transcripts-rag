"""Chunk SOP Markdown text into logical sections using header metadata."""

from typing import Dict, List

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

HEADER_SPLITS = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

HEADER_ORDER = ["Header 1", "Header 2", "Header 3"]


def _build_section_path(metadata: Dict[str, str]) -> str:
    parts = [metadata[level] for level in HEADER_ORDER if metadata.get(level)]
    return " > ".join(parts) if parts else "Full Document"


def _get_section_title(metadata: Dict[str, str]) -> str:
    for level in reversed(HEADER_ORDER):
        if metadata.get(level):
            return metadata[level]
    return metadata.get(HEADER_ORDER[0], "Full Document")


def chunk_sop_markdown(markdown_text: str) -> List[Dict[str, str]]:
    """Produce structured chunks from Markdown while keeping header context."""
    if not markdown_text or not markdown_text.strip():
        return []

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADER_SPLITS,
        strip_headers=False,
    )

    chunk_size = 1000
    chunk_overlap = 200

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    aggregated: List[Dict[str, str]] = []
    fallback_threshold = chunk_size + chunk_overlap

    for document in splitter.split_text(markdown_text):
        text = document.page_content.strip()
        if not text:
            continue

        headers = dict(document.metadata or {})
        section_path = _build_section_path(headers)
        section_title = _get_section_title(headers)

        def push_chunk(content: str, role: str) -> None:
            aggregated.append(
                {
                    "text": content.strip(),
                    "section_path": section_path,
                    "section_title": section_title,
                    "headers": headers,
                    "chunk_role": role,
                }
            )

        if len(text) > fallback_threshold:
            for child_text in fallback_splitter.split_text(text):
                trimmed_child = child_text.strip()
                if trimmed_child:
                    push_chunk(trimmed_child, "child_chunk")
        else:
            push_chunk(text, "parent_section")

    return aggregated
