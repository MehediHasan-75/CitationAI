# src/repositories/chunk_repository.py
from sqlalchemy.orm import Session
from typing import List
from src.database.models import Chunk


class ChunkRepository:
    """Repository for Chunk database operations"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_batch(self, chunks: List[Chunk]):
        """Create multiple chunks in batch"""
        self.db.add_all(chunks)
        self.db.commit()
    
    def get_by_paper_id(self, paper_id: int):
        """Get all chunks for a paper"""
        return self.db.query(Chunk).filter(
            Chunk.paper_id == paper_id
        ).all()
    
    def update_vector_ids(self, chunks: List[Chunk], vector_ids: List[str]):
        """Update chunks with vector IDs"""
        for chunk, vector_id in zip(chunks, vector_ids):
            chunk.vector_id = vector_id
            chunk.embedding_generated = True
        self.db.commit()
