"""Simple wrapper around the MarkItDown CLI for PDF → Markdown conversion."""

from pathlib import Path
import subprocess
from typing import Union


class MarkItDownError(Exception):
    """Raised when MarkItDown fails to parse or is unavailable."""


def pdf_to_markdown(file_path: Union[str, Path]) -> str:
    """Convert the given PDF file to Markdown via the MarkItDown CLI."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    command = ["markitdown", str(path)]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        error_message = exc.stderr.strip() or exc.stdout.strip()
        raise MarkItDownError(
            f"MarkItDown failed for {path.name}: {error_message}"
        ) from exc

    markdown_text = result.stdout.strip()
    if not markdown_text:
        raise MarkItDownError(f"MarkItDown returned empty content for {path.name}")

    return markdown_text
