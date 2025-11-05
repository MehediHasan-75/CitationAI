"""
Chunk model representing text segments from research papers.

Chunks are the fundamental unit of retrieval in the RAG system.
Each chunk is embedded and stored in Qdrant for semantic search.
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey, Index
from sqlalchemy.orm import relationship

from src.database.models import BaseModel
from src.database.models.mixins import TimestampMixin, VectorMixin, SectionHierarchyMixin


class Chunk(BaseModel, TimestampMixin, VectorMixin, SectionHierarchyMixin):
    __tablename__ = "chunks"
    """
    Text chunk extracted from a research paper with embeddings.
    
    Why chunks exist:
    - LLMs have context limits (can't process entire paper)
    - Semantic search works better on focused text segments
    - Enables precise citation (paragraph-level, not paper-level)
    - Balances retrieval granularity vs. context preservation
    
    Chunking strategy:
    - Respect section boundaries (don't split mid-section)
    - Typical size: 500-1000 tokens (configurable)
    - Overlap: 50-100 tokens to preserve context
    
    Lifecycle:
    1. Created from PDF sections → text populated
    2. Embedded → vector_id assigned
    3. Queried → retrieved by semantic similarity
    4. Cited → included in LLM response
    
    Inherits from:
        - BaseModel: id, __tablename__
        - TimestampMixin: created_at, updated_at
        - VectorMixin: vector_id, embedding_generated
        - SectionHierarchyMixin: section, section_id, section_level, page_number
    """    
    # ========================================================================
    # RELATIONSHIP FIELDS
    # ========================================================================
    
    paper_id = Column(
        Integer,
        ForeignKey("papers.id", ondelete="CASCADE"),
        # Why CASCADE: When paper deleted, auto-delete chunks
        index=True,  # Why indexed: Frequent joins and filters by paper_id
        nullable=False,
        # Why this field:
        # - Links chunk to source paper
        # - Required for:
        #   * Paper deletion (delete all chunks)
        #   * Citation ("Found in Paper X, Section Y")
        #   * Filtering ("Only search chunks from Papers 1,3,5")
    )
    
    # ========================================================================
    # CORE CHUNK FIELDS
    # ========================================================================
    
    chunk_index = Column(
        Integer,
        nullable=False,
        # Why this field:
        # - Preserves chunk order within paper
        # - Enables reconstruction: Sort chunks by index to rebuild paper
        # - Debugging: "Chunk 42 has weird text" → examine context
        # - Context assembly: Include adjacent chunks for better LLM context
        # Example: chunk_index=0 is first chunk, chunk_index=5 is sixth
    )
    
    text = Column(
        Text,  # Unlimited length (PostgreSQL: ~1GB, MySQL: 65KB)
        nullable=False,
        # Why Text vs String:
        # - Chunks can be 500-2000 characters
        # - String(N) has fixed max length
        # - Text is more flexible
        # Why this field:
        # - THE CORE DATA: This is what gets embedded and searched
        # - Quality of this text directly impacts RAG accuracy
        # - Used for:
        #   * Embedding generation
        #   * LLM context
        #   * Citation snippets
    )
    
    # ========================================================================
    # CUSTOM INDEXES FOR PERFORMANCE
    # ========================================================================
    
    __table_args__ = (
        # Composite index: Fast lookup of specific chunk in a paper
        Index('idx_paper_chunk', 'paper_id', 'chunk_index'),
        # Query: "Get chunk 5 from paper 42"
        
        # Composite index: Fast section-based search
        Index('idx_paper_section', 'paper_id', 'section_id'),
        # Query: "Get all chunks from Section 3 of paper 42"
        
        # Composite index: Page-based retrieval
        Index('idx_paper_page', 'paper_id', 'page_number'),
        # Query: "Get chunks from pages 10-15 of paper 42"
        
        # VectorMixin constraint (auto-merged)
        # Ensures: embedding_generated=True implies vector_id IS NOT NULL
    ) + VectorMixin.__table_args__
    
    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================
    
    paper = relationship(
        "Paper",
    back_populates="chunks",
        # Why this relationship:
        # - Access paper metadata from chunk: chunk.paper.title
        # - Join queries: Get chunks with paper info
        # Example: SELECT chunk.text, paper.title FROM chunks JOIN papers
    )
    
    def __repr__(self):
        """String representation for debugging"""
        return (
            f"<Chunk(id={self.id}, paper_id={self.paper_id}, "
            f"section='{self.section_id}', embedded={self.embedding_generated})>"
        )
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    @property
    def is_ready_for_search(self) -> bool:
        """
        Check if chunk is ready for semantic search.
        
        Why this method:
        - Convenience for filtering searchable chunks
        - Self-documenting: "is_ready_for_search" vs checking embedding_generated
        
        Returns:
            bool: True if embedding exists and can be searched
        """
        return self.embedding_generated and self.vector_id is not None
    
    def get_context_window(self, db_session, window_size: int = 2):
        """
        Retrieve adjacent chunks for better context.
        
        Why this method:
        - Single chunk may lack context
        - Example: Chunk mentions "this approach" - need previous chunk to know which
        - LLMs perform better with wider context
        
        Args:
            db_session: SQLAlchemy session
            window_size: Number of chunks before/after to include
        
        Returns:
            list[Chunk]: This chunk plus surrounding chunks
            
        Example:
            If chunk_index=5, window_size=2:
            Returns chunks [3, 4, 5, 6, 7]
        """
        return db_session.query(Chunk).filter(
            Chunk.paper_id == self.paper_id,
            Chunk.chunk_index >= self.chunk_index - window_size,
            Chunk.chunk_index <= self.chunk_index + window_size
        ).order_by(Chunk.chunk_index).all()
