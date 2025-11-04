"""
Reusable SQLAlchemy mixins for common database patterns.

Mixins provide composable functionality that can be added to any model
through multiple inheritance, promoting code reuse and consistency.
"""

from datetime import datetime
from sqlalchemy import Column, DateTime, Boolean, Integer, String, CheckConstraint
from sqlalchemy.ext.declarative import declared_attr


# ============================================================================
# TIMESTAMP MIXIN
# ============================================================================

class TimestampMixin:
    """
    Adds automatic timestamp tracking to any model.
    
    Why this exists:
    - Audit trail: Know when data was created/modified
    - Debugging: Track data lifecycle
    - Analytics: Time-based queries (e.g., "papers uploaded this week")
    - Compliance: Many industries require timestamp tracking
    
    Attributes:
        created_at: Timestamp when record was first inserted
        updated_at: Timestamp when record was last modified
    
    Use cases in RAG system:
    - Track when papers were uploaded
    - Monitor query frequency over time
    - Identify stale data for cleanup
    - Performance analysis (upload processing time)
    """
    
    created_at = Column(
        DateTime,
        default=datetime.utcnow,  # Auto-set on INSERT
        nullable=False,
        index=True,  # Why indexed: Enables fast time-range queries
        # Example query: "SELECT * FROM papers WHERE created_at > '2025-01-01'"
    )
    
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,  # Auto-set on INSERT
        onupdate=datetime.utcnow,  # Auto-update on UPDATE
        nullable=False,
        # Why not indexed: Updated timestamp rarely used for filtering
    )


# ============================================================================
# VECTOR EMBEDDING MIXIN
# ============================================================================

class VectorMixin:
    """
    Adds vector embedding tracking for models stored in vector databases.
    
    Why this exists:
    - Links relational DB records to vector DB entries (Qdrant)
    - Tracks embedding generation status for pipeline monitoring
    - Ensures data consistency between databases
    - Enables selective reprocessing (only regenerate failed embeddings)
    
    Attributes:
        vector_id: UUID linking this record to Qdrant vector store
        embedding_generated: Boolean flag for processing status
    
    Use cases:
    - During upload: Mark chunks as "embedding pending"
    - After embedding: Store Qdrant UUID and set flag to True
    - On error: Query chunks where embedding_generated=False for retry
    - For deletion: Use vector_id to delete from Qdrant
    """
    
    vector_id = Column(
        String(255),  # UUID from Qdrant (format: "abc123-def456-...")
        unique=True,  # Why unique: One-to-one mapping with Qdrant
        index=True,   # Why indexed: Fast lookups when syncing with Qdrant
        nullable=True  # Why nullable: Set after embedding generation
    )
    
    embedding_generated = Column(
        Boolean,
        default=False,  # New records haven't been embedded yet
        nullable=False,
        # Why not indexed: Boolean has low cardinality (not selective)
    )
    
    @declared_attr
    def __table_args__(cls):
        """
        Add database constraint ensuring data integrity.
        
        Why this constraint exists:
        - Prevents orphaned embeddings (embedding_generated=True but no vector_id)
        - Catches pipeline bugs at database level
        - Self-documenting business rule: "Can't mark as generated without ID"
        
        Logic: IF embedding_generated=True THEN vector_id MUST NOT be NULL
        """
        return (
            CheckConstraint(
                "(embedding_generated = FALSE) OR (vector_id IS NOT NULL)",
                name=f'ck_{cls.__name__.lower()}_vector_id_when_generated'
            ),
        )


# ============================================================================
# SECTION HIERARCHY MIXIN
# ============================================================================

class SectionHierarchyMixin:
    """
    Tracks document section structure for context-aware retrieval.
    
    Why this exists:
    - Academic papers have hierarchical structure (Introduction > Background > ...)
    - Users often query by section: "What's the methodology?"
    - Section context improves LLM answer quality
    - Enables section-level citations in responses
    
    Attributes:
        section: Human-readable section name (e.g., "Introduction")
        section_id: Numeric identifier (e.g., "3.2.1" for subsection)
        section_level: Hierarchy depth (0=main, 1=sub, 2=subsub)
        page_number: Page where this content appears
    
    Use cases:
    - RAG retrieval: "Find chunks from Methodology sections"
    - Citation generation: "Answer found in Section 3.2, page 15"
    - Context assembly: Prioritize chunks from same section
    - Analytics: "Which sections are most queried?"
    """
    
    section = Column(
        String(500),  # Length 500: Accommodates long section titles
        index=True,   # Why indexed: Frequent filtering by section name
        nullable=True  # Why nullable: Some PDFs lack section headers
        # Example values: "Abstract", "3. Methodology", "References"
    )
    
    section_id = Column(
        String(50),   # Format: "1", "2.1", "3.2.1", etc.
        index=True,   # Why indexed: Enables section hierarchy queries
        nullable=True  # Why nullable: Some papers use named sections only
        # Example query: "SELECT * WHERE section_id LIKE '3.%'" (all subsections of 3)
    )
    
    section_level = Column(
        Integer,
        default=0,  # 0 = top-level section (Introduction, Methods, etc.)
        nullable=False,
        # Why not indexed: Low cardinality (typically 0-3)
        # Values: 0=main, 1=subsection, 2=subsubsection
        # Example: "3.2.1" → level=2
    )
    
    page_number = Column(
        Integer,
        index=True,  # Why indexed: Enables page-based filtering and citations
        nullable=True  # Why nullable: Some PDFs have page numbering issues
        # Use cases:
        # - Citation: "Found on page 42"
        # - Filtering: "Only search Introduction (pages 1-3)"
        # - Quality check: Ensure chunks are page-sequential
    )
