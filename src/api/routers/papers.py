from fastapi import APIRouter, File, UploadFile, Depends, status
from sqlalchemy.orm import Session
from typing import Dict, Any

from src.database.session import get_db
from src.services.paper_service import PaperService
from src.repositories.paper_repository import PaperRepository
from src.repositories.chunk_repository import ChunkRepository

router = APIRouter(prefix='/api/papers', tags=['Papers'])

def get_paper_service(db: Session = Depends(get_db)) -> PaperService:
    """Dependency factory for PaperService"""
    paper_repo = PaperRepository(db)
    chunk_repo = ChunkRepository(db)
    return PaperService(db, paper_repo, chunk_repo)

@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_paper(  # ✅ Changed from async def to def
    file: UploadFile = File(..., description="PDF research paper to upload"),
    paper_service: PaperService = Depends(get_paper_service)
):
    """
    Upload and process a PDF research paper.
    
    **Process:**
    1. Validates file type and size
    2. Extracts paper name from filename
    3. Checks for duplicates
    4. Processes PDF (text extraction, metadata)
    5. Creates intelligent chunks
    6. Generates embeddings
    7. Stores vectors in Qdrant
    
    **Returns:**
    - paper_id, paper_name, title, authors, quality_score, etc.
    """
    try:
        # ✅ Removed await - just call it directly
        result = paper_service.process_and_save_paper(file)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
