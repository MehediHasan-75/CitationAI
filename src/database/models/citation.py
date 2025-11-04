from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.orm import relationship
from src.database.models.base import BaseModel


class Citation(BaseModel):
    __tablename__ = "citations"
    query_id = Column(Integer, ForeignKey("query_history.id", ondelete="CASCADE"), nullable=False)
    paper_id = Column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False)
    chunk_id = Column(Integer, ForeignKey("chunks.id", ondelete="SET NULL"), nullable=True)
    relevance_score = Column(Float, default=0.0)
    
    query = relationship("QueryHistory", backref="citations")
    paper = relationship("Paper", backref="citations")
    chunk = relationship("Chunk")
    
    def __repr__(self):
        return f"<Citation(query_id={self.query_id}, paper_id={self.paper_id})>"
