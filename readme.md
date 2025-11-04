# CitationAI - Complete System Documentation

**A Production-Ready Research Paper RAG System**

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Architecture Overview](#architecture-overview)
4. [Database Models Documentation](#database-models-documentation)
5. [API Guide](#api-guide)
6. [Service Layer Guide](#service-layer-guide)
7. [Setup & Configuration](#setup--configuration)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## System Overview

### What is CitationAI?

**CitationAI** is an intelligent research paper management and analysis system that uses **Retrieval-Augmented Generation (RAG)** to help researchers:

- ✅ Upload research papers (PDFs)
- ✅ Automatically extract and analyze content
- ✅ Ask natural language questions about papers
- ✅ Get accurate answers with citations and sources
- ✅ Track research queries and analytics

### Problem It Solves

```
❌ Without CitationAI:
   - Reading 50 research papers = 100+ hours
   - Manual note-taking = Error-prone
   - Finding specific information = Time-consuming

✅ With CitationAI:
   - Ask questions = Get instant answers
   - AI reads papers = Fast & accurate
   - Citations included = Know your sources
   - Analytics included = Track research patterns
```

### Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Backend** | FastAPI | Modern, fast, async-first Python framework |
| **Database** | SQLite (dev) / PostgreSQL (prod) | Structured data storage |
| **Vector DB** | Qdrant | Semantic search via embeddings |
| **ORM** | SQLAlchemy | Type-safe database queries |
| **Embeddings** | sentence-transformers | Convert text to vectors |
| **LLM** | Ollama / DeepSeek | Answer generation |
| **Task Queue** | Celery (optional) | Background processing |

---

## Quick Start Guide

### 1️⃣ **Installation**

```bash
# Clone repository
git clone <repo-url>
cd CitationAI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install Alembic for migrations
pip install alembic
```

### 2️⃣ **Configuration**

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# DATABASE_URL=sqlite:///./citationai.db
# DEBUG=True
# QDRANT_HOST=localhost
# etc.
```

### 3️⃣ **Database Setup**

```bash
# Initialize database (creates tables)
python -c "from src.database.session import init_db; init_db()"

# Or using Alembic (migrations)
alembic revision --autogenerate -m "Initial tables"
alembic upgrade head
```

### 4️⃣ **Start Services**

```bash
# Terminal 1: Start PostgreSQL/SQLite database
# (Skip if using SQLite - it's already local)

# Terminal 2: Start Qdrant vector database
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant

# Terminal 3: Start LLM service (Ollama)
ollama serve

# Terminal 4: Start FastAPI server
uvicorn src.main:app --reload
```

### 5️⃣ **Test API**

```bash
# Open in browser
http://localhost:8000/docs

# Or via curl
curl -X POST "http://localhost:8000/api/papers/upload" \
  -F "file=@your_paper.pdf"
```

---

## Architecture Overview

### 3-Tier Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     API LAYER (Routes)                       │
│  ────────────────────────────────────────────────────────  │
│  Responsibility: HTTP request/response handling             │
│  - Receive HTTP requests                                    │
│  - Validate input (file type, size)                         │
│  - Call service layer                                       │
│  - Return HTTP responses                                    │
│  Location: src/api/routers/                                 │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                  SERVICE LAYER (Logic)                       │
│  ────────────────────────────────────────────────────────  │
│  Responsibility: Business logic orchestration                │
│  - Coordinate repositories                                  │
│  - Implement workflows                                      │
│  - Handle complex operations                                │
│  - No database queries (uses repos)                          │
│  Location: src/services/                                    │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│              REPOSITORY LAYER (Data Access)                  │
│  ────────────────────────────────────────────────────────  │
│  Responsibility: Database CRUD operations                    │
│  - Query database                                           │
│  - Insert/update/delete records                             │
│  - No business logic                                        │
│  Location: src/repositories/                                │
└──────────────┬──────────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────────┐
│                   DATABASE LAYER                             │
│  ────────────────────────────────────────────────────────  │
│  - SQLAlchemy ORM                                           │
│  - Database models                                          │
│  - Migrations (Alembic)                                     │
│  Location: src/database/                                    │
└─────────────────────────────────────────────────────────────┘
```

### Example Data Flow: Paper Upload

```
User uploads paper.pdf
       ↓
API Route receives upload
       ↓
Validates: File is PDF? Size < 50MB?
       ↓
Service layer coordinates upload
       ├─ File Service: Save to disk
       ├─ PDF Processor: Extract text
       ├─ Paper Repository: Create record
       ├─ Chunk Repository: Save chunks
       ├─ Embedding Service: Generate vectors
       └─ Qdrant Service: Store in vector DB
       ↓
Database updated
       ↓
Return success response to user
```

---

## Database Models Documentation

### Model Hierarchy

```
┌─────────────────────────────────────────┐
│          Base Classes & Mixins          │
├─────────────────────────────────────────┤
│  • BaseModel (id, __tablename__)        │
│  • TimestampMixin (created_at, updated_at) │
│  • VectorMixin (vector_id, embedding)   │
│  • SectionHierarchyMixin (section info) │
└─────────────────────────────────────────┘
       ↓ (inherited by)
┌─────────────────────────────────────────┐
│         Entity Models (Tables)          │
├─────────────────────────────────────────┤
│  • Paper (research documents)           │
│  • Chunk (text segments)                │
│  • QueryHistory (user questions)        │
│  • Citation (query↔paper links)         │
└─────────────────────────────────────────┘
```

### Paper Model

**Represents**: A research document uploaded to the system

**Fields**:
- `id` - Auto-increment primary key
- `paper_name` - Unique identifier from filename
- `title` - Document title (from PDF metadata)
- `authors` - List of authors (JSON)
- `year` - Publication year
- `keywords` - Research keywords (JSON)
- `abstract` - Paper summary
- `filename` - Original upload filename
- `file_path` - Path on disk to PDF
- `total_pages` - Number of pages
- `chunk_count` - Number of text chunks
- `quality_score` - Extraction quality (0-1)
- `processed` - Ready for search? (Boolean)
- `created_at` - When uploaded
- `updated_at` - Last modified time

**Usage Example**:
```python
# Create paper on upload
paper = Paper(
    paper_name="transformer_2017",
    title="Attention Is All You Need",
    authors=["Vaswani", "Shazeer", ...],
    year=2017,
    filename="attention_is_all_you_need.pdf",
    file_path="/uploads/abc-123-def.pdf",
    processed=False
)
db.add(paper)
db.commit()

# Mark as ready for search
paper.processed = True
db.commit()
```

### Chunk Model

**Represents**: A text segment from a paper (with embeddings)

**Why chunks?**
- LLMs can't process entire papers (context limits)
- Enables precise, section-level retrieval
- Better performance (embed smaller pieces)
- Allows paragraph-level citations

**Fields**:
- `id` - Auto-increment
- `paper_id` - Foreign key to Paper
- `chunk_index` - Order in paper
- `text` - Actual text content (to be embedded)
- `section` - Section name ("Introduction", etc.)
- `section_id` - Hierarchical ID ("3.2.1")
- `section_level` - Depth (0=main, 1=sub)
- `page_number` - Physical page in PDF
- `vector_id` - UUID in Qdrant
- `embedding_generated` - Processing status
- `created_at` / `updated_at` - Timestamps

**Usage Example**:
```python
# After PDF extraction
for section in pdf_sections:
    chunks = intelligent_chunker.chunk_section(
        section_text=section.text,
        section_name="Methodology",
        page_start=15,
        page_end=20
    )
    
    for chunk in chunks:
        db_chunk = Chunk(
            paper_id=paper.id,
            chunk_index=0,
            text=chunk.text,
            section="Methodology",
            section_id="3.2",
            page_number=15
        )
        db.add(db_chunk)

db.commit()

# Generate embeddings
chunks = db.query(Chunk).filter(
    Chunk.paper_id == paper.id,
    Chunk.embedding_generated == False
).all()

embeddings = embedding_service.encode_batch(
    [c.text for c in chunks]
)

# Store in Qdrant
vector_ids = qdrant_service.upsert_chunks(
    chunks_data, embeddings, paper.id
)

# Update database
for chunk, vector_id in zip(chunks, vector_ids):
    chunk.vector_id = vector_id
    chunk.embedding_generated = True

db.commit()
```

### QueryHistory Model

**Represents**: A user question asked to the system

**Fields**:
- `id` - Auto-increment
- `query_text` - User's question
- `answer` - LLM-generated response
- `response_time` - How long it took (seconds)
- `confidence` - Answer certainty (0-1)
- `top_k` - Number of chunks retrieved
- `user_rating` - User feedback (1-5 stars)
- `created_at` - When asked

**Usage Example**:
```python
# Save query
query_log = QueryHistory(
    query_text="What is the attention mechanism?",
    answer="The attention mechanism allows...",
    response_time=0.5,
    confidence=0.92,
    top_k=5,
    user_rating=5
)
db.add(query_log)
db.commit()

# Query analytics
popular_queries = db.query(QueryHistory).order_by(
    QueryHistory.created_at.desc()
).limit(10).all()

avg_response_time = db.query(
    func.avg(QueryHistory.response_time)
).scalar()
```

### Citation Model

**Represents**: Which papers contributed to answering a query

**Fields**:
- `id` - Auto-increment
- `query_id` - Which query (FK to QueryHistory)
- `paper_id` - Which paper (FK to Paper)
- `chunk_id` - Which chunk (FK to Chunk)
- `relevance_score` - How relevant (0-1)

**Usage Example**:
```python
# When answering a query, track which papers were used
for retrieved_chunk in top_chunks:
    citation = Citation(
        query_id=query_log.id,
        paper_id=retrieved_chunk.paper_id,
        chunk_id=retrieved_chunk.id,
        relevance_score=similarity_score
    )
    db.add(citation)

db.commit()

# Analytics: Most cited papers
popular_papers = db.query(Paper).join(
    Citation
).group_by(Paper.id).order_by(
    func.count(Citation.id).desc()
).limit(10).all()

# Citation: "Based on: Paper X (relevance: 0.92)"
response = {
    "answer": "The answer is...",
    "sources": [
        f"{citation.paper.title} (p. {citation.chunk.page_number})"
        for citation in citations
    ]
}
```

---

## API Guide

### Upload Paper Endpoint

**Endpoint**: `POST /api/papers/upload`

**Request**:
```bash
curl -X POST "http://localhost:8000/api/papers/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@paper.pdf"
```

**Response** (Success - 201):
```json
{
  "paper_id": 1,
  "paper_name": "transformer_paper",
  "title": "Attention Is All You Need",
  "authors": ["Vaswani", "Shazeer"],
  "year": 2017,
  "quality_score": 0.92,
  "chunks_created": 96,
  "upload_status": "success",
  "message": "Successfully uploaded and processed transformer_paper"
}
```

**Response** (Error - 400):
```json
{
  "detail": "Only PDF files are allowed"
}
```

### List Papers Endpoint

**Endpoint**: `GET /api/papers`

**Query Parameters**:
- `skip` - Pagination offset (default: 0)
- `limit` - Results per page (default: 10)

**Example**:
```bash
curl "http://localhost:8000/api/papers?skip=0&limit=10"
```

**Response**:
```json
{
  "papers": [
    {
      "id": 1,
      "paper_name": "transformer",
      "title": "Attention Is All You Need",
      "processed": true,
      "chunk_count": 96,
      "quality_score": 0.92,
      "created_at": "2025-11-04T10:30:00"
    }
  ],
  "total": 1,
  "skip": 0,
  "limit": 10
}
```

### Get Paper Stats

**Endpoint**: `GET /api/papers/{paper_id}/stats`

**Example**:
```bash
curl "http://localhost:8000/api/papers/1/stats"
```

**Response**:
```json
{
  "paper_id": 1,
  "paper_name": "transformer",
  "title": "Attention Is All You Need",
  "chunks_created": 96,
  "quality_score": 0.92,
  "processed": true,
  "uploaded": "2025-11-04T10:30:00"
}
```

### Delete Paper

**Endpoint**: `DELETE /api/papers/{paper_id}`

**Example**:
```bash
curl -X DELETE "http://localhost:8000/api/papers/1"
```

**Response**:
```json
{
  "status": "deleted"
}
```

---

## Service Layer Guide

### PaperService

**Purpose**: Orchestrates paper upload and processing

**Location**: `src/services/paper_service.py`

**Methods**:

```python
class PaperService:
    # Upload and process a new paper
    async def process_and_save_paper(file: UploadFile) -> dict
    
    # Get all papers (paginated)
    def get_papers(skip: int, limit: int) -> list
    
    # Get specific paper
    def get_paper_by_id(paper_id: int) -> dict
    
    # Delete paper
    async def delete_paper(paper_id: int) -> None
    
    # Get paper statistics
    def get_paper_stats(paper_id: int) -> dict
```

**Example Usage**:
```python
from src.services.paper_service import PaperService
from src.database.session import SessionLocal

db = SessionLocal()
service = PaperService(db, paper_repo, chunk_repo)

# Upload paper
result = await service.process_and_save_paper(uploaded_file)

# Get stats
stats = service.get_paper_stats(paper_id=1)
```

### FileService

**Purpose**: Handles file I/O (save, delete)

**Location**: `src/services/file_service.py`

**Methods**:

```python
class FileService:
    # Save uploaded file to disk
    async def save_upload(file: UploadFile) -> str
    
    # Delete file from disk
    def delete_file(file_path: str) -> None
```

---

## Setup & Configuration

### Environment Variables

**File**: `.env`

```bash
# Application
APP_NAME=CitationAI Research Paper System
DEBUG=True
API_VERSION=v1

# Database
DATABASE_URL=sqlite:///./citationai.db
# For PostgreSQL: postgresql://user:password@localhost:5432/citationai

# Qdrant Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=citationai_papers

# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
OLLAMA_URL=http://localhost:11434

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# PDF Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
MAX_UPLOAD_SIZE=50  # MB
UPLOAD_DIR=./uploads

# Retrieval
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7
```

### Database Setup

**Development** (SQLite - no setup needed):
```bash
# Just run
python -c "from src.database.session import init_db; init_db()"
# Creates citationai.db automatically
```

**Production** (PostgreSQL):
```sql
CREATE DATABASE citationai;
CREATE USER citationai_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE citationai TO citationai_user;
```

Then set:
```bash
DATABASE_URL=postgresql://citationai_user:secure_password@localhost:5432/citationai
```

### Running the Server

**Development**:
```bash
uvicorn src.main:app --reload
```

**Production**:
```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.main:app
```

---

## Troubleshooting

### ❌ Error: "Module not found 'src'"

**Solution**: Ensure you're running from project root:
```bash
cd /path/to/CitationAI
python -c "from src.database.session import init_db; init_db()"
```

### ❌ Error: "Cannot import BaseModel from database.base"

**Solution**: BaseModel should be imported from:
```python
from src.database.models.base import BaseModel  # Not from database.base!
```

### ❌ Error: "SQLite invalid isolation level"

**Solution**: SQLite doesn't support READ COMMITTED. Session.py auto-detects:
```python
if "sqlite" in settings.DATABASE_URL:
    isolation_level = "SERIALIZABLE"  # SQLite only supports this
else:
    isolation_level = "READ COMMITTED"  # PostgreSQL uses this
```

### ❌ Error: "Foreign key associated with column could not find table"

**Solution**: Ensure `__tablename__` matches foreign key references:
```python
# In Paper model
class Paper(BaseModel):
    __tablename__ = "papers"  # Must be explicit!

# In Chunk model
class Chunk(BaseModel):
    paper_id = ForeignKey("papers.id")  # Must match Paper.__tablename__
```

### ❌ Error: "Database is locked"

**Solution**: Multiple processes accessing SQLite. Either:
1. Use PostgreSQL for production
2. Reduce concurrency in development
3. Increase timeout: `DATABASE_URL=sqlite:///./citationai.db?timeout=20`

---

## Best Practices

### 1. Always Use Dependency Injection

```python
# ✅ Good
@router.post("/upload")
async def upload(file: UploadFile, service: PaperService = Depends(get_service)):
    return await service.process(file)

# ❌ Bad
@router.post("/upload")
async def upload(file: UploadFile):
    service = PaperService()  # Can't mock for testing
```

### 2. Use Type Hints

```python
# ✅ Good
def get_paper(paper_id: int) -> Paper:
    return db.query(Paper).filter_by(id=paper_id).first()

# ❌ Bad
def get_paper(paper_id):
    return db.query(Paper).filter_by(id=paper_id).first()
```

### 3. Batch Database Operations

```python
# ✅ Good - Fast
chunks_batch = []
for chunk in chunks:
    chunks_batch.append(chunk)
    if len(chunks_batch) >= 10:
        db.add_all(chunks_batch)
        db.commit()
        chunks_batch = []

# ❌ Bad - Slow
for chunk in chunks:
    db.add(chunk)
    db.commit()  # Commits after EVERY chunk!
```

### 4. Handle Errors Explicitly

```python
# ✅ Good
try:
    paper = service.get_paper(id)
except ValueError:
    raise HTTPException(status_code=404, detail="Paper not found")
except DatabaseError:
    logger.error(f"Database error: {e}")
    raise HTTPException(status_code=500)

# ❌ Bad
try:
    paper = service.get_paper(id)
except:  # Catches everything!
    pass
```

### 5. Log Important Events

```python
# ✅ Good
logger.info(f"Paper uploaded: {paper_name}")
logger.warning(f"Low quality score: {quality_score}")
logger.error(f"Processing failed: {error}", exc_info=True)

# ❌ Bad
print("Done")  # Can't control level, format, or filter
```

### 6. Use Connection Pooling

```python
# Already configured in session.py
engine = create_engine(
    DATABASE_URL,
    pool_size=10,        # Keep 10 connections ready
    max_overflow=20,     # Allow up to 20 more on demand
    pool_pre_ping=True,  # Test before using
    pool_recycle=3600    # Recycle after 1 hour
)
```

---

## Summary

**Key Files to Understand**:

1. **`src/main.py`** - FastAPI application entry point
2. **`src/api/routers/papers.py`** - HTTP endpoints
3. **`src/services/paper_service.py`** - Business logic
4. **`src/repositories/`** - Database operations
5. **`src/database/models/`** - Data models
6. **`src/core/config.py`** - Configuration

**Typical Development Flow**:

1. Define model in `src/database/models/`
2. Create repository in `src/repositories/`
3. Create service in `src/services/`
4. Create API routes in `src/api/routers/`
5. Test via `/docs` or curl

**Deployment Checklist**:

- [ ] Set `DEBUG=False` in `.env`
- [ ] Use PostgreSQL instead of SQLite
- [ ] Set strong database password
- [ ] Configure TLS for connections
- [ ] Set up backups
- [ ] Use Gunicorn for server
- [ ] Configure reverse proxy (nginx)
- [ ] Set up monitoring/logging

---

**Happy coding! 🚀**
