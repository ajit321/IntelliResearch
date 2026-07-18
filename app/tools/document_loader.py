"""
app/tools/document_loader.py
Document loader for user-uploaded files (PDF, DOCX, TXT).
Extracts raw text for RAG indexing.
"""

from fastapi import UploadFile

from app.utils.logging import get_logger

logger = get_logger(__name__)


async def extract_text_from_upload(file: UploadFile) -> str:
    """
    Extract plain text from an uploaded file.

    Supports: PDF, DOCX, TXT, MD files.
    Returns empty string on failure (graceful degradation).

    Args:
        file: FastAPI UploadFile object from multipart form.

    Returns:
        Extracted text content as a string.
    """
    content = await file.read()
    filename = file.filename or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    try:
        if ext == "txt" or ext == "md":
            return content.decode("utf-8", errors="ignore")

        elif ext == "pdf":
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(content))
            text = "\n\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )
            logger.info("PDF extracted", filename=filename, chars=len(text))
            return text

        elif ext == "docx":
            import docx
            import io
            doc = docx.Document(io.BytesIO(content))
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            logger.info("DOCX extracted", filename=filename, chars=len(text))
            return text

        else:
            # Try UTF-8 decode as last resort
            text = content.decode("utf-8", errors="ignore")
            logger.warning("Unknown file type — decoded as UTF-8", filename=filename)
            return text

    except Exception as exc:
        logger.error("Document extraction failed", filename=filename, error=str(exc))
        return ""
