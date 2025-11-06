from fastapi import APIRouter, File, UploadFile, Depends, status, HTTPException
from sqlalchemy.orm import Session
from typing import List
from src.database.session import get_db
from src.database.models import Paper, Chunk
from src.services.paper_service import PaperService
from src.repositories.paper_repository import PaperRepository
from src.repositories.chunk_repository import ChunkRepository


router = APIRouter(prefix='/api/papers', tags=['Papers'])


def get_paper_service(db: Session = Depends(get_db)) -> PaperService:
    """Dependency factory for PaperService"""
    paper_repo = PaperRepository(db)
    chunk_repo = ChunkRepository(db)
    return PaperService(db, paper_repo, chunk_repo)


# ✅ EXISTING: Upload endpoint
@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_paper(
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
        result = paper_service.process_and_save_paper(file)
        return result
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# ➕ NEW: List all papers
@router.get("", status_code=status.HTTP_200_OK)
def list_papers(
    skip: int = 0,
    limit: int = 10,
    processed_only: bool = False,
    db: Session = Depends(get_db)
):
    """
    List all papers with pagination.
    
    **Query Parameters:**
    - skip: Number of records to skip (for pagination)
    - limit: Maximum number of records to return
    - processed_only: If True, only return fully processed papers
    
    **Returns:**
    - List of papers with metadata
    - Total count
    """
    query = db.query(Paper)
    
    if processed_only:
        query = query.filter(Paper.processed == True)
    
    total = query.count()
    papers = query.order_by(Paper.created_at.desc()).offset(skip).limit(limit).all()
    
    return {
        "papers": [
            {
                "id": p.id,
                "paper_name": p.paper_name,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "total_pages": p.total_pages,
                "chunk_count": p.chunk_count,
                "quality_score": p.quality_score,
                "processed": p.processed,
                "created_at": p.created_at.isoformat()
            }
            for p in papers
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }


# ➕ NEW: Get specific paper
@router.get("/{paper_id}", status_code=status.HTTP_200_OK)
def get_paper(
    paper_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific paper.
    
    **Returns:**
    - Complete paper metadata
    - List of sections
    - Processing statistics
    """
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with id {paper_id} not found"
        )
    
    return {
        "id": paper.id,
        "paper_name": paper.paper_name,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "keywords": paper.keywords,
        "abstract": paper.abstract,
        "filename": paper.filename,
        "total_pages": paper.total_pages,
        "chunk_count": paper.chunk_count,
        "quality_score": paper.quality_score,
        "format_type": paper.format_type,
        "sections": paper.sections,
        "processed": paper.processed,
        "created_at": paper.created_at.isoformat(),
        "updated_at": paper.updated_at.isoformat()
    }


# ➕ NEW: Get paper statistics
@router.get("/{paper_id}/stats", status_code=status.HTTP_200_OK)
def get_paper_stats(
    paper_id: int,
    db: Session = Depends(get_db)
):
    """
    Get statistics for a specific paper.
    
    **Returns:**
    - Chunk distribution by section
    - Average chunk length
    - Embedding status
    """
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with id {paper_id} not found"
        )
    
    # Get chunks for this paper
    chunks = db.query(Chunk).filter(Chunk.paper_id == paper_id).all()
    
    # Calculate stats
    total_chunks = len(chunks)
    embedded_chunks = sum(1 for c in chunks if c.embedding_generated)
    
    # Section distribution
    section_distribution = {}
    for chunk in chunks:
        section = chunk.section or "Unknown"
        section_distribution[section] = section_distribution.get(section, 0) + 1
    
    # Avg chunk length
    avg_length = sum(len(c.text) for c in chunks) / total_chunks if total_chunks > 0 else 0
    
    return {
        "paper_id": paper.id,
        "paper_name": paper.paper_name,
        "title": paper.title,
        "statistics": {
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "embedding_completion": round(embedded_chunks / total_chunks * 100, 2) if total_chunks > 0 else 0,
            "average_chunk_length": round(avg_length, 0),
            "section_distribution": section_distribution,
            "quality_score": paper.quality_score
        }
    }


# ➕ NEW: Get paper chunks
@router.get("/{paper_id}/chunks", status_code=status.HTTP_200_OK)
def get_paper_chunks(
    paper_id: int,
    skip: int = 0,
    limit: int = 20,
    section: str = None,
    db: Session = Depends(get_db)
):
    """
    Get chunks for a specific paper.
    
    **Query Parameters:**
    - skip: Pagination offset
    - limit: Max chunks to return
    - section: Filter by section name (optional)
    
    **Returns:**
    - List of chunks with text, section, page info
    """
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    
    if not paper:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Paper with id {paper_id} not found"
        )
    
    query = db.query(Chunk).filter(Chunk.paper_id == paper_id)
    
    if section:
        query = query.filter(Chunk.section.ilike(f"%{section}%"))
    
    total = query.count()
    chunks = query.order_by(Chunk.chunk_index).offset(skip).limit(limit).all()
    
    return {
        "paper_id": paper.id,
        "paper_name": paper.paper_name,
        "chunks": [
            {
                "id": c.id,
                "chunk_index": c.chunk_index,
                "text": c.text[:200] + "..." if len(c.text) > 200 else c.text,  # Preview
                "section": c.section,
                "section_id": c.section_id,
                "page_number": c.page_number,
                "embedding_generated": c.embedding_generated
            }
            for c in chunks
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }


# ➕ NEW: Delete paper
@router.delete("/{paper_id}", status_code=status.HTTP_200_OK)
def delete_paper(
    paper_id: int,
    paper_service: PaperService = Depends(get_paper_service)
):
    """
    Delete a paper and all its associated data.
    
    **What gets deleted:**
    - Paper record
    - All chunks (cascade)
    - All embeddings from Qdrant
    - PDF file from disk
    
    **Returns:**
    - Success message
    """
    try:
        paper_service.delete_paper(paper_id)
        return {
            "status": "success",
            "message": f"Paper {paper_id} and all associated data deleted"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete paper: {str(e)}"
        )


# ➕ NEW: Search papers
@router.get("/search/by-title", status_code=status.HTTP_200_OK)
def search_papers_by_title(
    query: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Search papers by title or author.
    
    **Query Parameters:**
    - query: Search term
    - limit: Max results
    
    **Returns:**
    - Matching papers
    """
    papers = db.query(Paper).filter(
        (Paper.title.ilike(f"%{query}%")) | 
        (Paper.paper_name.ilike(f"%{query}%"))
    ).limit(limit).all()
    
    return {
        "query": query,
        "results": [
            {
                "id": p.id,
                "paper_name": p.paper_name,
                "title": p.title,
                "authors": p.authors,
                "year": p.year,
                "processed": p.processed
            }
            for p in papers
        ],
        "count": len(papers)
    }
