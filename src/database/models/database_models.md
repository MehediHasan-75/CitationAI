# Research Paper RAG System - Database Models Documentation

## Overview

This document describes the database schema for the Research Paper RAG (Retrieval-Augmented Generation) system. The schema is designed to support:

- **Document Management**: Upload and store research papers
- **Content Chunking**: Divide papers into searchable segments
- **Vector Storage Tracking**: Link database records to Qdrant embeddings
- **Query Analytics**: Track user questions and system performance
- **Citation Management**: Maintain relationships between queries and source papers

---

## Architecture Philosophy

### Why Modular Models?

The database schema uses **mixins** and **base classes** to promote:

- **DRY (Don't Repeat Yourself)**: Common fields (timestamps, vectors) defined once, reused everywhere
- **Consistency**: All models follow the same patterns and conventions
- **Maintainability**: Changes to timestamp logic affect all models automatically
- **Testability**: Mixins can be tested independently
- **Scalability**: Easy to add new models or functionality

### Design Principles

1. **Separation of Concerns**: Each model represents one entity
2. **Referential Integrity**: Foreign keys with cascade rules prevent orphaned data
3. **Performance Indexing**: Composite indexes for common query patterns
4. **Audit Trail**: Timestamps on all entities for compliance and debugging
5. **Data Validation**: Database constraints enforce business rules at the lowest level

---

## Core Data Flow

```
User Uploads PDF
    ↓
Paper Record Created (processed=False)
    ↓
PDF Extracted & Chunked
    ↓
Multiple Chunk Records Created
    ↓
Embeddings Generated
    ↓
Vector IDs Stored (vector_id populated, embedding_generated=True)
    ↓
Paper Marked as Processed (processed=True)
    ↓
Ready for Semantic Search
```

---

## Base Classes

### BaseModel

**Purpose**: Provides common fields and methods for all database entities

**Why it exists**:
- Every table needs a unique identifier
- Auto-generates consistent table names
- Ensures all models have standard debugging output

**Key Features**:
- `id` (Integer, Primary Key): Unique identifier with automatic increment
- `__tablename__` (auto-generated): Table name from class name
- `__repr__()`: Debugging-friendly string representation

**Used by**: All models (Paper, Chunk, QueryHistory, Citation, etc.)

**Example Usage**:
```python
class Paper(BaseModel, TimestampMixin):
    pass  # Automatically has: id, __tablename__, timestamps
```

---

## Mixins (Reusable Functionality)

### TimestampMixin

**Purpose**: Track record creation and modification times

**Fields**:
- `created_at` (DateTime): Auto-set when record inserted
- `updated_at` (DateTime): Auto-updated on any modification

**Why timestamps matter**:
- **Audit Trail**: Know when papers were uploaded and processed
- **Data Lifecycle**: Identify stale papers for archival
- **Performance Analysis**: Measure upload-to-ready time
- **Analytics**: "How many papers uploaded this week?"
- **Debugging**: Correlate events with timestamps in logs
- **Compliance**: Many regulations require audit trails

**Used by**: Paper, Chunk, QueryHistory, Citation

**Example Queries**:
```sql
-- Find papers uploaded today
SELECT * FROM papers WHERE DATE(created_at) = CURDATE()

-- Get oldest unprocessed papers
SELECT * FROM papers WHERE processed=FALSE ORDER BY created_at LIMIT 10

-- Track processing time
SELECT id, EXTRACT(EPOCH FROM (updated_at - created_at)) as processing_seconds
FROM papers WHERE processed=TRUE
```

### VectorMixin

**Purpose**: Link database records to Qdrant vector store embeddings

**Fields**:
- `vector_id` (String): UUID from Qdrant (e.g., "550e8400-e29b-41d4-a716-446655440000")
- `embedding_generated` (Boolean): Whether embedding has been created

**Constraint**:
- If `embedding_generated=True`, then `vector_id` MUST NOT be NULL
- Prevents data inconsistency bugs

**Why this exists**:
- **Dual Database Pattern**: Relational DB (PostgreSQL) + Vector DB (Qdrant)
- **Syncing**: Track which records have embeddings generated
- **Debugging**: Identify processing failures (missing embeddings)
- **Reprocessing**: Query records where `embedding_generated=False` for retry
- **Deletion**: Use `vector_id` to remove vectors from Qdrant

**Used by**: Chunk, (future: ImageEmbedding, etc.)

**Workflow**:
1. Create chunk record → `embedding_generated=False`, `vector_id=NULL`
2. Generate embedding → `embedding_generated=True`, `vector_id="abc-123-def"`
3. Query results → Retrieve by `vector_id` from Qdrant
4. Delete chunk → Delete from Qdrant using `vector_id`, then delete from DB

**Example Queries**:
```sql
-- Find chunks that failed embedding generation
SELECT id FROM chunks WHERE embedding_generated=FALSE

-- Get all embeddings for a paper (to delete from Qdrant)
SELECT vector_id FROM chunks WHERE paper_id=5 AND embedding_generated=TRUE

-- Validate consistency (should return 0 rows)
SELECT * FROM chunks WHERE embedding_generated=TRUE AND vector_id IS NULL
```

### SectionHierarchyMixin

**Purpose**: Preserve document structure for context-aware retrieval

**Fields**:
- `section` (String): Human-readable section name (e.g., "Methodology")
- `section_id` (String): Numeric identifier (e.g., "3.2.1")
- `section_level` (Integer): Hierarchy depth (0=main, 1=sub, 2=subsub)
- `page_number` (Integer): Physical page location in document

**Why sections matter**:
- **Semantic Context**: User query "What's the methodology?" finds Methodology sections
- **Citation Accuracy**: "Found in Section 3.2, page 15" instead of vague paper reference
- **Context Assembly**: Include adjacent sections for better LLM responses
- **Quality Metrics**: "Which sections have the most questions?"
- **Document Reconstruction**: Sort chunks to rebuild paper structure

**Hierarchy Examples**:
```
Level 0: "Introduction", "Methodology", "Results"
Level 1: "3.1 Background", "3.2 Data Collection"
Level 2: "3.2.1 Preprocessing", "3.2.2 Validation"
```

**Used by**: Chunk, (future: Paragraph, Section)

**Example Queries**:
```sql
-- Find all methodology sections
SELECT * FROM chunks WHERE section LIKE '%Methodology%'

-- Get specific section hierarchy
SELECT * FROM chunks WHERE section_id LIKE '3.2%'  -- Section 3.2 and all subsections

-- Chunks on specific pages
SELECT * FROM chunks WHERE paper_id=5 AND page_number BETWEEN 10 AND 15

-- Section-level analytics
SELECT section, COUNT(*) as chunk_count FROM chunks GROUP BY section
```

---

## Entity Models

### Paper Model

**Represents**: A single research document uploaded to the system

**Why separate from Chunk**:
- Papers can be large (100+ pages)
- Storing full text in Paper wastes space (replicated in chunks)
- Metadata queries don't need full text
- Normalization: One paper ↔ Many chunks

**Workflow in System**:
1. User uploads PDF → Paper created with `processed=False`
2. Text extracted → `title`, `authors`, `year` populated
3. Sections identified → `sections` JSON populated
4. Quality score calculated → `quality_score` populated
5. Chunks created → `chunk_count` updated
6. Embeddings generated → `processed=True`
7. Now ready for semantic search

**Key Fields**:

**Identification** (`paper_name`, `title`):
- `paper_name`: Extracted from filename, used for deduplication
- `title`: From PDF metadata, displayed in search results

**Metadata** (`authors`, `year`, `keywords`, `abstract`):
- Extracted automatically from PDF
- Enables filtering and search refinement
- Improves user experience with rich information

**File Management** (`filename`, `file_path`):
- Enables reprocessing with improved algorithms
- Supports deletion (clean up disk)
- Audit trail: "What did user upload?"

**Processing Status** (`processed`, `chunk_count`, `quality_score`, `format_type`):
- Track pipeline completion
- Monitor quality metrics
- Support different PDF formats (journal article vs. report vs. preprint)

**Relationships**:
- One Paper → Many Chunks (1:N)
- When paper deleted → All chunks cascade-deleted from both DB and Qdrant

**Example Usage in RAG**:
```python
# Find searchable papers
papers = db.query(Paper).filter(Paper.processed == True).all()

# Filter by quality
high_quality = db.query(Paper).filter(Paper.quality_score > 0.8).all()

# Get paper with all chunks
paper = db.query(Paper).filter_by(id=5).first()
chunks = paper.chunks  # Access all chunks via relationship
```

### Chunk Model

**Represents**: A text segment extracted from a paper with embeddings

**Why chunks instead of storing full papers**:
- **LLM Limitations**: Context windows (4K-100K tokens) can't process entire papers
- **Relevance**: Retrieve only relevant sections, not whole papers
- **Citations**: Paragraph-level precision beats paper-level vagueness
- **Efficiency**: Embed 1000 small chunks faster than 1 large document
- **Quality**: GPT performs better with focused context

**Chunking Strategy**:
- Respect section boundaries (don't split mid-thought)
- Target size: 500-1000 tokens per chunk (configurable)
- Overlap: 50-100 tokens preserve context across chunks

**Workflow**:
1. Paper chunked → Chunk records created with `text` populated
2. Embeddings generated → Text vectorized via sentence-transformers
3. Stored in Qdrant → `vector_id` assigned, `embedding_generated=True`
4. User query → Find similar chunks via vector similarity
5. Build context → Include chunk text in LLM prompt
6. Generate answer → LLM uses chunks to answer

**Key Fields**:

**Linking** (`paper_id`, `chunk_index`):
- `paper_id`: Connects to source paper (enables CASCADE deletion)
- `chunk_index`: Preserves order within paper

**Content** (`text`):
- The actual text that gets embedded
- Quality of this text directly impacts RAG accuracy
- Used for context assembly and LLM input

**Location** (inherited from SectionHierarchyMixin):
- `section`, `section_id`, `section_level`, `page_number`
- Enables precise citations
- Supports section-filtered retrieval

**Embeddings** (inherited from VectorMixin):
- `vector_id`: UUID pointing to Qdrant vector
- `embedding_generated`: Processing status

**Example Usage in RAG**:
```python
# Find chunks ready for search
searchable_chunks = db.query(Chunk).filter(
    Chunk.embedding_generated == True,
    Chunk.vector_id.isnot(None)
).all()

# Get chunks from specific section
methodology_chunks = db.query(Chunk).filter(
    Chunk.paper_id == 5,
    Chunk.section.like('%Methodology%')
).all()

# Chunks with context
chunk = db.query(Chunk).filter_by(id=100).first()
context = chunk.get_context_window(db_session, window_size=2)  # Get ±2 chunks
```

### QueryHistory Model

**Represents**: User questions asked to the RAG system

**Why track queries**:
- **Analytics**: "What topics are users interested in?"
- **Quality Monitoring**: Average confidence scores
- **Debugging**: Reproduce user issues
- **Optimization**: Identify slow queries for improvement
- **User Feedback**: Optional ratings for fine-tuning

**Workflow**:
1. User submits question
2. RAG pipeline retrieves and generates answer
3. QueryHistory record saved with response metadata
4. User optionally rates answer (feedback loop)

**Key Fields**:
- `query_text`: The user's question
- `answer`: LLM-generated response
- `response_time`: Performance metric
- `confidence`: How certain was the answer (0-1)
- `top_k`: How many chunks retrieved
- `paper_filter`: Which papers were searched (optional)
- `user_rating`: User satisfaction (1-5 stars)
- `timestamp`: When query was submitted

**Example Usage**:
```python
# Most asked questions
popular = db.query(QueryHistory).group_by(
    QueryHistory.query_text
).having(count() > 10).all()

# Average response time
avg_time = db.query(func.avg(QueryHistory.response_time)).scalar()

# Questions on specific topic
nlp_queries = db.query(QueryHistory).filter(
    QueryHistory.query_text.ilike('%NLP%')
).all()
```

### Citation Model

**Represents**: Which papers were used to answer a query

**Why track citations**:
- **Traceability**: "Where did this answer come from?"
- **Source Attribution**: Legal/ethical requirement
- **Paper Popularity**: "Which papers answer most queries?"
- **Quality Assessment**: "Do answers from high-quality papers get better ratings?"

**Workflow**:
1. RAG retrieves chunks from papers A, B, C
2. Citation records created: QueryID → PaperA, PaperB, PaperC
3. Response includes: "Based on: Paper A, Section 3; Paper B, Page 15"
4. User sees sources, can click to read full papers

**Key Fields**:
- `query_id`: Links to QueryHistory
- `paper_id`: Which paper contributed to answer
- `relevance_score`: How relevant was this paper (0-1)
- `chunk_id`: Optional: specific chunk that was used

**Example Usage**:
```python
# Find all papers that answered query #42
citations = db.query(Citation).filter_by(query_id=42).all()
for citation in citations:
    print(f"Paper: {citation.paper.title}, Relevance: {citation.relevance_score}")

# Most cited papers
popular = db.query(Paper).join(Citation).group_by(
    Paper.id
).order_by(func.count(Citation.id).desc()).limit(10).all()
```

---

## Database Indexes

### Why Indexes Matter

Indexes speed up queries by creating lookup tables. Without indexes, database scans every row (slow). With indexes, database jumps directly to relevant rows (fast).

**Trade-off**: Faster queries ↔ Slower insertions (must update indexes)

### Index Strategy

**Simple Indexes** (for WHERE clauses):
- `papers.paper_name`: Deduplication checks
- `papers.processed`: "Show unprocessed papers"
- `papers.created_at`: Date-range queries
- `chunks.paper_id`: Join papers to chunks
- `chunks.embedding_generated`: "Find chunks needing embeddings"
- `chunks.section_id`: "Get all Section 3.2 content"

**Composite Indexes** (for multi-column WHERE):
- `(papers.processed, papers.created_at)`: "Oldest unprocessed papers"
- `(chunks.paper_id, chunks.chunk_index)`: "Get chunk 5 from paper 42"
- `(chunks.paper_id, chunks.section_id)`: "Get all chunks from Section X in Paper Y"

---

## Data Consistency Strategies

### Cascade Deletion

When a Paper is deleted:
```sql
DELETE FROM papers WHERE id=5
-- Automatically: DELETE FROM chunks WHERE paper_id=5
-- Automatically: DELETE FROM Qdrant WHERE paper_id=5
```

**Why**: Prevent orphaned chunks pointing to non-existent papers

### Check Constraints

Vector constraint ensures data integrity:
```sql
-- Invalid: embedding_generated=True but no vector_id
INSERT INTO chunks (text, embedding_generated, vector_id)
VALUES ('Hello', TRUE, NULL)  -- REJECTED

-- Valid: If embedded, must have vector_id
INSERT INTO chunks (text, embedding_generated, vector_id)
VALUES ('Hello', TRUE, 'abc-123-def')  -- ACCEPTED
```

### Foreign Key Constraints

`chunks.paper_id` must reference existing `papers.id`:
```sql
-- Invalid: Paper 999 doesn't exist
INSERT INTO chunks (paper_id, text) VALUES (999, 'Hello')  -- REJECTED

-- Valid: Paper 5 exists
INSERT INTO chunks (paper_id, text) VALUES (5, 'Hello')  -- ACCEPTED
```

---

## Query Patterns (Common Use Cases)

### Upload Pipeline
```python
# Create paper
paper = Paper(paper_name="transformer_paper", title="Attention Is All You Need")

# Create chunks
for section in pdf_sections:
    chunk = Chunk(
        paper_id=paper.id,
        text=section.text,
        section_id="3.2",
        embedding_generated=False
    )

# After embedding
chunk.vector_id = "qdrant-uuid-123"
chunk.embedding_generated = True

# Mark complete
paper.processed = True
```

### RAG Retrieval
```python
# Find searchable chunks
chunks = db.query(Chunk).filter(
    Chunk.embedding_generated == True,
    Chunk.paper_id.in_([1, 3, 5])  # Optional: specific papers
).all()

# Build context for LLM
context = "\n".join([c.text for c in chunks])

# Save query
query_log = QueryHistory(
    query_text=user_question,
    answer=llm_response,
    response_time=0.5,
    confidence=0.87
)

# Track citations
for chunk in chunks:
    citation = Citation(
        query_id=query_log.id,
        paper_id=chunk.paper_id,
        relevance_score=similarity_score
    )
```

### Analytics
```python
# Most queried papers
papers = db.query(Paper).join(Citation).group_by(
    Paper.id
).order_by(func.count(Citation.id).desc()).all()

# Average answer quality by paper quality
results = db.query(
    Paper.quality_score,
    func.avg(QueryHistory.user_rating)
).join(Citation).join(QueryHistory).group_by(Paper.quality_score).all()

# Processing time trend
times = db.query(
    func.date(Paper.created_at),
    func.avg(func.extract('epoch', Paper.updated_at - Paper.created_at))
).group_by(func.date(Paper.created_at)).all()
```

---

## Performance Considerations

### Database Size Estimation
- Small deployment: 100 papers, 50K chunks → ~500MB
- Medium deployment: 1K papers, 500K chunks → ~5GB
- Large deployment: 10K papers, 5M chunks → ~50GB

### Optimization Tips

1. **Batch Operations**: Insert 1000 chunks at once, not 1-by-1
2. **Connection Pooling**: Reuse DB connections
3. **Lazy Loading**: Don't load all chunks if you only need IDs
4. **Pagination**: Limit results (LIMIT 100, not LIMIT 1000000)
5. **Archive Old Queries**: Move QueryHistory > 1 year to archive table

### Query Optimization Example

```python
# ❌ Slow: Loads entire paper into memory
paper = db.query(Paper).filter_by(id=5).first()
chunk_count = len(paper.chunks)  # Triggers full load

# ✅ Fast: Uses COUNT query, no loading
chunk_count = db.query(func.count(Chunk.id)).filter_by(paper_id=5).scalar()
```

---

## Migration and Versioning

### Using Alembic for Schema Changes

```bash
# Create migration
alembic revision --autogenerate -m "Add quality_score to papers"

# Apply migration
alembic upgrade head

# Rollback if needed
alembic downgrade -1
```

### Common Migrations

1. **Adding a column**: Add index if filtering by it
2. **Renaming a column**: Backfill data before migration
3. **Adding a constraint**: Verify existing data satisfies it first
4. **Changing types**: Ensure compatibility (e.g., INTEGER → BIGINT OK, not vice versa)

---

## Security Considerations

1. **SQL Injection Prevention**: Use ORM (SQLAlchemy), never raw SQL with user input
2. **Data Encryption**: Use TLS for database connections
3. **Access Control**: Database user should have minimal permissions
4. **Audit Logging**: Track schema changes and data modifications
5. **Backup Strategy**: Daily backups, test recovery procedures

---

## References

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [PostgreSQL Best Practices](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Database Indexing Guide](https://use-the-index-luke.com/)
- [RAG System Design](https://docs.llamaindex.ai/en/stable/optimizing/)
