"""Format layer — turns raw bytes into LLM-friendly markdown/text."""

from finsage.preprocess.format.html import html_to_markdown
from finsage.preprocess.format.pdf import (
    BoundingBox,
    ParsedDocument,
    ParsedPage,
    PdfParseError,
    TextBlock,
    parse_pdf,
    pdf_to_markdown,
)

__all__ = [
    "BoundingBox",
    "ParsedDocument",
    "ParsedPage",
    "PdfParseError",
    "TextBlock",
    "html_to_markdown",
    "parse_pdf",
    "pdf_to_markdown",
]
