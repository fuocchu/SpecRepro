"""
Paper text utilities for SpecRepro.

Supports reading:
  - Plain text (.txt)
  - PDF via pdfminer or pdfplumber (optional)
"""

import os
import re


def read_paper(path: str) -> str:
    """
    Read paper content from a .txt or .pdf file.
    Returns the full text as a string.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return _read_txt(path)
    elif ext == ".pdf":
        return _read_pdf(path)
    else:
        # Try reading as plain text anyway
        return _read_txt(path)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    """
    Extract text from PDF using pdfminer.six (preferred) or pdfplumber (fallback).
    Falls back to a clear error message if neither is installed.
    """
    try:
        from pdfminer.high_level import extract_text
        return extract_text(path)
    except ImportError:
        pass

    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n\n".join(text_parts)
    except ImportError:
        pass

    raise ImportError(
        "Cannot read PDF: install 'pdfminer.six' or 'pdfplumber'.\n"
        "  pip install pdfminer.six\n"
        "Alternatively, convert the PDF to .txt first using MinerU or similar."
    )


def truncate_paper(text: str, max_chars: int = 80_000) -> str:
    """
    Truncate paper text to fit within LLM context limits.
    Keeps the beginning and end (abstract + conclusion most relevant).
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... truncated ...]\n\n" + text[-half:]


def clean_paper_text(text: str) -> str:
    """Basic cleanup: remove excess whitespace, fix common OCR artifacts."""
    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove page headers/footers (lines with only numbers)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()
