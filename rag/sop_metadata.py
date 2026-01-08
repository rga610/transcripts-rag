"""Utilities for extracting SOP metadata from Markdown text."""

import re
from typing import Dict

_PATTERN_MAP: Dict[str, str] = {
    "sop_id": r"\b(SOP[- ]?[A-Z0-9]+)\b",
    "version": r"\b(?:Version|Rev(?:ision)?|v)\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)*)\b",
    "last_updated": r"(?:Last Updated|Last Revision)\s*[:\-]\s*([A-Za-z0-9 ,./-]+)",
    "department": r"\bDepartment\s*[:\-]\s*([A-Za-z &/]+)\b",
}


def extract_sop_metadata(markdown_text: str) -> Dict[str, str]:
    """Extract known SOP metadata fields from the Markdown string."""
    metadata: Dict[str, str] = {}
    if not markdown_text:
        return metadata

    for key, pattern in _PATTERN_MAP.items():
        match = re.search(pattern, markdown_text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                metadata[key] = value

    return metadata
