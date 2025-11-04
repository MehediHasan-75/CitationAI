"""
Database base classes and mixins for the Research Paper RAG System.

This module provides reusable base classes and mixins that implement common
database patterns across all models, following DRY principles.
"""

from sqlalchemy import Column, Integer
from sqlalchemy.ext.declarative import declared_attr
from src.database.base import Base

class BaseModel(Base):
    """
    Abstract base model providing common fields for all database tables.
    
    Why this exists:
    - Every database entity needs a unique identifier (id)
    - Auto-generates consistent table names from class names
    - Provides standard __repr__ for debugging
    - Ensures consistency across all models
    
    Attributes:
        id (int): Primary key, auto-incrementing. Used for:
            - Unique identification of records
            - Foreign key relationships
            - Efficient indexing and lookups
    
    Design decisions:
        - Uses Integer over UUID for performance (faster joins)
        - Auto-increment simplifies insertion
        - Index on primary key is automatic in PostgreSQL/MySQL
    """
    
    __abstract__ = True  # Don't create a table for this base class
    
    id = Column(
        Integer, 
        primary_key=True,  # Ensures uniqueness and creates automatic index
        index=True,        # Explicit for clarity (redundant but readable)
        autoincrement=True  # Database handles ID generation
    )
    
    @declared_attr
    def __tablename__(cls):
        """
        Auto-generate table name from class name (lowercase).
        
        Why: Maintains naming consistency without manual specification.
        Example: Class "Paper" → table "paper"
        """
        return cls.__name__.lower()
    
    def __repr__(self):
        """
        Default string representation for debugging and logging.
        
        Why: Makes database query results readable in logs and debug sessions.
        Example output: <Paper(id=42)>
        """
        return f"<{self.__class__.__name__}(id={self.id})>"
