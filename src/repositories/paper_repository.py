"""
Paper Repository - Data access layer for Paper entity.

Provides CRUD operations and custom queries for papers.
"""

from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from src.database.models.paper import Paper
from src.database.repositories.base import BaseRepository


class PaperRepository(BaseRepository[Paper]):
    """Repository for Paper entity with custom queries"""
    
    def __init__(self, db: Session):
        super().__init__(db, Paper)
    
    # ==================== RETRIEVE METHODS ====================
    
    def get_by_name(self, paper_name: str) -> Optional[Paper]:
        """
        Get paper by paper_name (unique identifier).
        
        Args:
            paper_name: Unique paper name (extracted from filename)
            
        Returns:
            Paper if found, None otherwise
        """
        return self.db.query(Paper).filter(Paper.paper_name == paper_name).first()
    
    def get_by_filename(self, filename: str) -> Optional[Paper]:
        """
        Get paper by original filename.
        
        Args:
            filename: Original uploaded filename
            
        Returns:
            Paper if found, None otherwise
        """
        return self.db.query(Paper).filter(Paper.filename == filename).first()
    
    def get_all(
        self, 
        skip: int = 0, 
        limit: int = 10, 
        processed_only: bool = False
    ) -> List[Paper]:
        """
        Get papers with pagination and filtering.
        
        Args:
            skip: Number of records to skip (pagination)
            limit: Maximum records to return
            processed_only: If True, return only processed papers
            
        Returns:
            List of Paper objects
        """
        query = self.db.query(Paper)
        
        if processed_only:
            query = query.filter(Paper.processed == True)
        
        return query.order_by(Paper.created_at.desc()).offset(skip).limit(limit).all()
    
    def count(self, processed_only: bool = False) -> int:
        """
        Count total papers.
        
        Args:
            processed_only: If True, count only processed papers
            
        Returns:
            Total count
        """
        query = self.db.query(func.count(Paper.id))
        
        if processed_only:
            query = query.filter(Paper.processed == True)
        
        return query.scalar()
    
    def search_by_title(self, query: str, limit: int = 10) -> List[Paper]:
        """
        Search papers by title or paper_name (case-insensitive).
        
        Args:
            query: Search term
            limit: Maximum results
            
        Returns:
            List of matching papers
        """
        return self.db.query(Paper).filter(
            (Paper.title.ilike(f"%{query}%")) | 
            (Paper.paper_name.ilike(f"%{query}%"))
        ).limit(limit).all()
    
    def get_by_author(self, author_name: str) -> List[Paper]:
        """
        Get papers by author name (searches in JSON authors field).
        
        Args:
            author_name: Author name to search
            
        Returns:
            List of papers by that author
        """
        # PostgreSQL JSON query
        return self.db.query(Paper).filter(
            func.lower(func.cast(Paper.authors, String)).like(f"%{author_name.lower()}%")
        ).all()
    
    def get_unprocessed(self) -> List[Paper]:
        """
        Get all unprocessed papers (for retry/debugging).
        
        Returns:
            List of papers where processed=False
        """
        return self.db.query(Paper).filter(Paper.processed == False).all()
    
    # ==================== DUPLICATE CHECK ====================
    
    def check_duplicate(self, paper_name: str, filename: str) -> Optional[Paper]:
        """
        Check if paper already exists (by name or filename).
        
        Args:
            paper_name: Unique paper name
            filename: Original filename
            
        Returns:
            Existing paper if duplicate found, None otherwise
        """
        # Check by name first (primary key for duplicates)
        existing = self.get_by_name(paper_name)
        if existing:
            return existing
        
        # Check by filename as backup
        existing = self.get_by_filename(filename)
        if existing:
            return existing
        
        return None
    
    # ==================== UPDATE METHODS ====================
    
    def mark_processed(self, paper: Paper, chunk_count: int) -> None:
        """
        Mark paper as fully processed.
        
        Args:
            paper: Paper object to update
            chunk_count: Number of chunks created
        """
        paper.processed = True
        paper.chunk_count = chunk_count
        self.db.commit()
        self.db.refresh(paper)
    
    def update_metadata(
        self, 
        paper_id: int, 
        title: str = None,
        authors: List[str] = None,
        year: int = None,
        keywords: List[str] = None,
        abstract: str = None
    ) -> Optional[Paper]:
        """
        Update paper metadata.
        
        Args:
            paper_id: Paper ID
            title: New title (optional)
            authors: New authors list (optional)
            year: New year (optional)
            keywords: New keywords (optional)
            abstract: New abstract (optional)
            
        Returns:
            Updated paper or None if not found
        """
        paper = self.get_by_id(paper_id)
        if not paper:
            return None
        
        if title is not None:
            paper.title = title
        if authors is not None:
            paper.authors = authors
        if year is not None:
            paper.year = year
        if keywords is not None:
            paper.keywords = keywords
        if abstract is not None:
            paper.abstract = abstract
        
        self.db.commit()
        self.db.refresh(paper)
        return paper
    
    # ==================== STATISTICS ====================
    
    def get_stats(self) -> dict:
        """
        Get overall statistics about papers.
        
        Returns:
            Dictionary with stats
        """
        total = self.db.query(func.count(Paper.id)).scalar()
        processed = self.db.query(func.count(Paper.id)).filter(
            Paper.processed == True
        ).scalar()
        avg_chunks = self.db.query(func.avg(Paper.chunk_count)).filter(
            Paper.processed == True
        ).scalar() or 0
        avg_quality = self.db.query(func.avg(Paper.quality_score)).filter(
            Paper.processed == True
        ).scalar() or 0
        
        return {
            "total_papers": total,
            "processed_papers": processed,
            "unprocessed_papers": total - processed,
            "avg_chunks_per_paper": round(avg_chunks, 1),
            "avg_quality_score": round(avg_quality, 3)
        }
