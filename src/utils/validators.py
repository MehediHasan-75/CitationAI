from fastapi import UploadFile, HTTPException, status
from pathlib import Path
from src.core.config import settings


def validate_pdf_file(file: UploadFile) -> None:
    """
    Validate uploaded file is a PDF and within size limits.
    
    Raises:
        HTTPException: If validation fails
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are allowed. You uploaded: {file.content_type}"
        )
    
    if file.size and file.size > settings.MAX_UPLOAD_SIZE * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {settings.MAX_UPLOAD_SIZE}MB)"
        )
"""
extract_filename_stem("document.pdf")       # → "document"
extract_filename_stem("archive.tar.gz")     # → "archive.tar"
extract_filename_stem("/path/to/file.txt")  # → "file"
"""
def extract_filename_stem(filename: str):
    """Extract filename without extension"""
    return Path(filename).stem