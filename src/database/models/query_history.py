from sqlalchemy import Column, String, Integer, Float, JSON, Text
from src.database.models import BaseModel
from src.database.models.mixins import TimestampMixin


class QueryHistory(BaseModel, TimestampMixin):
    __tablename__ = "query_history"
    query_text = Column(String(1000), nullable=False)
    answer = Column(Text)
    response_time = Column(Float)
    confidence = Column(Float)
    top_k = Column(Integer, default=5)
    paper_filter = Column(JSON)
    user_rating = Column(Integer)
    
    def __repr__(self):
        return f"<QueryHistory(id={self.id}, confidence={self.confidence})>"
