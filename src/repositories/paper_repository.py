from sqlalchemy.orm import Session
from typing import Optional
from src.database.models import Paper


class PaperRepository:
    """Repository for Paper database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_by_name(self, paper_name: str) -> Optional[Paper]:
        """Get paper by name"""
        return self.db.query(Paper).filter(
            Paper.paper_name == paper_name
        ).first()
    
    def get_by_filename(self, filename: str) -> Optional[Paper]:
        """Get paper by filename"""
        return self.db.query(Paper).filter(
            Paper.filename == filename
        ).first()
    
    def check_duplicate(self, paper_name: str, filename: str) -> Optional[Paper]:
        """Check if paper exists by name or filename"""
        existing_by_name = self.get_by_name(paper_name)
        existing_by_filename = self.get_by_filename(filename)
        return existing_by_name or existing_by_filename
    
    def create(self, **kwargs) -> Paper:
        """Create new paper record"""
        paper = Paper(**kwargs)
        self.db.add(paper)
        self.db.commit()
        self.db.refresh(paper)
        return paper
    
    def mark_processed(self, paper: Paper, chunk_count: int) -> None:
        """Mark paper as fully processed"""
        paper.processed = True
        paper.chunk_count = chunk_count
        self.db.commit()
