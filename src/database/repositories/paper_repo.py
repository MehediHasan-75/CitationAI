# src/repositories/paper_repository.py
from sqlalchemy.orm import Session
from src.database.models.paper import Paper
from src.database.repositories.base import BaseRepository

class PaperRepository(BaseRepository[Paper]):
    
    def __init__(self, db: Session):
        super().__init__(db, Paper)
    
    def get_by_name(self, paper_name: str) -> Paper | None:
        return self.db.query(Paper).filter(Paper.paper_name == paper_name).first()
    
    def get_by_filename(self, filename: str) -> Paper | None:
        return self.db.query(Paper).filter(Paper.filename == filename).first()
    
    def check_duplicate(self, paper_name: str, filename: str) -> Paper | None:
        """✅ NEW METHOD - Check if paper already exists"""
        # Check by name first (primary key for duplicates)
        existing = self.get_by_name(paper_name)
        if existing:
            return existing
        
        # Check by filename as backup
        existing = self.get_by_filename(filename)
        if existing:
            return existing
        
        return None
    
    def mark_processed(self, paper: Paper, chunk_count: int) -> None:
        """✅ FIXED - Accept paper object, not just ID"""
        paper.processed = True
        paper.chunk_count = chunk_count
        self.db.commit()
