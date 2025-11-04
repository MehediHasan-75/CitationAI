from sqlalchemy.orm import Session
from src.db.models.paper import Paper
from src.repositories.base import BaseRepository


class PaperRepository(BaseRepository[Paper]):
    
    def __init__(self, db: Session):
        super().__init__(db, Paper)
    
    def get_by_name(self, paper_name: str) -> Paper | None:
        return self.db.query(Paper).filter(Paper.paper_name == paper_name).first()
    
    def get_by_filename(self, filename: str) -> Paper | None:
        return self.db.query(Paper).filter(Paper.filename == filename).first()
    
    def mark_processed(self, paper_id: int) -> None:
        paper = self.get_by_id(paper_id)
        if paper:
            paper.processed = True
            self.db.commit()
