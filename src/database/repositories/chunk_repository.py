from typing import List
from sqlalchemy.orm import Session
from src.database.models.chunk import Chunk
from src.database.repositories.base import BaseRepository


class ChunkRepository(BaseRepository[Chunk]):
    
    def __init__(self, db: Session):
        super().__init__(db, Chunk)
    
    def get_by_paper_id(self, paper_id: int) -> List[Chunk]:
        return self.db.query(Chunk).filter(Chunk.paper_id == paper_id).all()
    
    def get_with_embeddings(self, paper_id: int) -> List[Chunk]:
        return self.db.query(Chunk).filter(
            Chunk.paper_id == paper_id,
            Chunk.embedding_generated == True
        ).all()
    
    def update_vector_ids(self, chunks: List[Chunk], vector_ids: List[str]) -> None:
        for chunk, vector_id in zip(chunks, vector_ids):
            chunk.vector_id = vector_id
            chunk.embedding_generated = True
        self.db.commit()

