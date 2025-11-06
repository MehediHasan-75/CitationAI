# CitationAI: Complete Beginner's Guide to Building RAG System

**For Complete Beginners - No Prior Experience Needed!**

---

## 📋 Table of Contents

1. [What You'll Build](#what-youll-build)
2. [Prerequisites & Setup](#prerequisites--setup)
3. [Project Timeline (7 Days)](#project-timeline-7-days)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Code Files to Create](#code-files-to-create)
6. [Testing at Each Step](#testing-at-each-step)
7. [Deployment](#deployment)

---

## What You'll Build

### CitationAI: Research Paper RAG System

A system that:
1. **Takes PDF papers** → Reads them
2. **Asks questions** → Gets AI answers
3. **Shows sources** → Cites where answers came from
4. **Tracks analytics** → Knows what you asked

**Example Workflow**:
```
User: Upload "transformer.pdf"
       ↓
System: Reads paper, extracts text, creates searchable chunks
       ↓
User: "What is attention mechanism?"
       ↓
AI: "Attention mechanism allows... [Source: Page 3]"
```

---

## Prerequisites & Setup

### What You Need to Learn First

**Duration**: 2-3 hours self-study

#### 1. **Python Basics** (If new to programming)

```python
# Variables and data types
name = "Alice"
age = 25
scores = [90, 85, 95]

# Functions
def greet(name):
    return f"Hello, {name}!"

# Loops
for score in scores:
    print(score)

# Dictionaries (JSON-like)
person = {
    "name": "Alice",
    "age": 25,
    "skills": ["Python", "AI"]
}
print(person["name"])  # Output: Alice
```

**Resource**: https://www.learnpython.org/

#### 2. **REST APIs Concept** (15 minutes)

Think of an API as a waiter:
- You (client) ask for something
- Waiter (API) goes to kitchen (backend)
- Kitchen (database) prepares food
- Waiter returns with result

**HTTP Methods**:
```
GET    = Read (get data)
POST   = Create (send data)
PUT    = Update (modify data)
DELETE = Remove (delete data)
```

#### 3. **Database Basics** (30 minutes)

Tables are like Excel sheets:
```
Papers Table:
| id | title                    | author      |
|----|--------------------------|-------------|
| 1  | Attention Is All You Need| Vaswani     |
| 2  | BERT: Pre-training       | Devlin      |

Chunks Table:
| id | paper_id | text              |
|----|----------|-------------------|
| 1  | 1        | "Attention is..." |
| 2  | 1        | "The transformer...|
```

---

## Project Timeline (7 Days)

### **Week Overview**

```
Day 1 (Mon)  → Setup + First API (1-2 hours)
Day 2 (Tue)  → Database Models (2 hours)
Day 3 (Wed)  → PDF Processing (2 hours)
Day 4 (Thu)  → Vector Database (2 hours)
Day 5 (Fri)  → RAG Pipeline (3 hours)
Day 6 (Sat)  → Complete APIs (2 hours)
Day 7 (Sun)  → Testing + Deployment (2 hours)

Total: ~16 hours
```

---

## Step-by-Step Implementation

---

## DAY 1: Setup + First API Endpoint (1-2 hours)

### What You'll Learn
- Install Python packages
- Create a FastAPI project
- Write your first API endpoint
- Test it

### Part 1: Installation (30 minutes)

```bash
# 1. Create project folder
mkdir CitationAI
cd CitationAI

# 2. Create virtual environment (isolates your project)
python -m venv venv

# 3. Activate it
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# 4. Install FastAPI (web framework)
pip install fastapi uvicorn

# 5. Verify
python -c "import fastapi; print('✅ FastAPI installed')"
```

**What is virtual environment?**
- Keeps your project's packages separate
- Like a sandbox for your project
- Different projects can use different versions

### Part 2: Create First API (30 minutes)

Create file: `main.py`

```python
from fastapi import FastAPI

# Create FastAPI app
app = FastAPI(title="CitationAI")

# Health check endpoint (test if server is running)
@app.get("/")
def home():
    """Check if server is alive"""
    return {
        "message": "✅ CitationAI Server Running",
        "version": "1.0"
    }

# Simple test endpoint
@app.get("/test")
def test():
    """Test endpoint"""
    return {
        "status": "success",
        "data": "Hello from CitationAI!"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

### Part 3: Run and Test (30 minutes)

```bash
# Terminal 1: Start server
python main.py

# Should see:
# INFO:     Started server process [12345]
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://127.0.0.1:8000
```

```bash
# Terminal 2: Test in another terminal
curl http://localhost:8000

# Or open in browser
http://localhost:8000/docs

# This shows interactive API documentation!
```

**What you've done**:
- ✅ Installed FastAPI
- ✅ Created first API
- ✅ Tested it works

---

## DAY 2: Database Models (2 hours)

### What You'll Learn
- What databases are
- Create database models
- Understand relationships

### Part 1: Install Database Package (10 minutes)

```bash
pip install sqlalchemy pydantic
```

### Part 2: Create Database Models (1 hour)

Create file: `database_models.py`

```python
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

# ========== PAPER MODEL ==========
class Paper(Base):
    """Represents a research paper"""
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    paper_name = Column(String, unique=True)  # Unique identifier
    title = Column(String)  # Paper title
    authors = Column(String)  # Comma-separated authors
    year = Column(Integer)  # Publication year
    filename = Column(String)  # Original filename
    file_path = Column(String)  # Where PDF is stored
    total_pages = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)  # 0-1
    processed = Column(Boolean, default=False)  # Is it ready?
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Paper(id={self.id}, title='{self.title}')>"


# ========== CHUNK MODEL ==========
class Chunk(Base):
    """Represents a text chunk from a paper"""
    __tablename__ = "chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"))  # Link to paper
    chunk_index = Column(Integer)  # Order in paper
    text = Column(Text)  # Actual text content
    section = Column(String)  # Section name (Intro, Methods, etc)
    page_number = Column(Integer)  # Page in PDF
    vector_id = Column(Integer)  # ID in vector database
    embedding_generated = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Chunk(id={self.id}, paper_id={self.paper_id}, section='{self.section}')>"


# ========== QUERY HISTORY MODEL ==========
class QueryHistory(Base):
    """Represents a user question"""
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    query_text = Column(String)  # The question
    answer = Column(Text)  # The answer
    response_time = Column(Float)  # How long it took
    confidence = Column(Float)  # 0-1 confidence score
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Query(id={self.id}, question='{self.query_text[:30]}...')>"


# ========== CITATION MODEL ==========
class Citation(Base):
    """Represents: Query used Paper"""
    __tablename__ = "citations"
    
    id = Column(Integer, primary_key=True, index=True)
    query_id = Column(Integer, ForeignKey("queries.id"))  # Which query
    paper_id = Column(Integer, ForeignKey("papers.id"))  # Which paper
    relevance_score = Column(Float)  # 0-1 how relevant
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Citation(query_id={self.query_id}, paper_id={self.paper_id})>"
```

**What you've learned**:
- Models are like table blueprints
- Each model becomes a database table
- ForeignKey creates relationships between tables

### Part 3: Setup Database Connection (1 hour)

Create file: `database.py`

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from database_models import Base

# Create database file
DATABASE_URL = "sqlite:///./citationai.db"

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # SQLite specific
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create all tables
Base.metadata.create_all(bind=engine)
print("✅ Database initialized!")

# Function to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

Run it:
```bash
python database.py
# Creates citationai.db file
```

**What you've done**:
- ✅ Created database models
- ✅ Set up SQLite database
- ✅ Created connection manager

---

## DAY 3: PDF Processing (2 hours)

### What You'll Learn
- Extract text from PDFs
- Split text into chunks
- Save to database

### Part 1: Install PDF Libraries (5 minutes)

```bash
pip install pdfplumber pypdf
```

### Part 2: Create PDF Processor (1 hour 30 minutes)

Create file: `pdf_processor.py`

```python
import pdfplumber
from typing import List, Dict

class PDFProcessor:
    """Extract text from PDF files"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size  # Characters per chunk
        self.chunk_overlap = chunk_overlap  # Overlap between chunks
    
    def extract_text(self, pdf_path: str) -> Dict:
        """Extract text and metadata from PDF"""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Extract metadata
                metadata = pdf.metadata or {}
                
                # Extract all text
                full_text = ""
                pages_text = []
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    full_text += text + "\n"
                    pages_text.append(text)
                
                return {
                    "success": True,
                    "title": metadata.get("Title", "Unknown"),
                    "author": metadata.get("Author", "Unknown"),
                    "pages": len(pdf.pages),
                    "full_text": full_text,
                    "pages_text": pages_text
                }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def chunk_text(self, text: str, page_num: int = 1) -> List[Dict]:
        """Split text into chunks"""
        chunks = []
        chunk_index = 0
        
        # Split by chunks with overlap
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Don't cut in middle of word
            if end < len(text):
                end = text.rfind(" ", start, end)
                if end == start:
                    end = start + self.chunk_size
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only if not empty
                chunks.append({
                    "index": chunk_index,
                    "text": chunk_text,
                    "page": page_num,
                    "char_count": len(chunk_text)
                })
                chunk_index += 1
            
            # Move start with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Complete PDF processing: extract → chunk"""
        # Extract text
        extracted = self.extract_text(pdf_path)
        if not extracted["success"]:
            return extracted
        
        # Chunk text
        all_chunks = []
        for page_num, page_text in enumerate(extracted["pages_text"], 1):
            page_chunks = self.chunk_text(page_text, page_num)
            all_chunks.extend(page_chunks)
        
        return {
            "success": True,
            "title": extracted["title"],
            "author": extracted["author"],
            "pages": extracted["pages"],
            "total_chunks": len(all_chunks),
            "chunks": all_chunks
        }


# Example usage
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Process a PDF
    result = processor.process_pdf("sample.pdf")
    
    if result["success"]:
        print(f"✅ Processed: {result['title']}")
        print(f"   Pages: {result['pages']}")
        print(f"   Chunks: {result['total_chunks']}")
        
        # Show first chunk
        if result['chunks']:
            print(f"\n📄 First chunk:\n{result['chunks'][0]['text'][:100]}...")
    else:
        print(f"❌ Error: {result['error']}")
```

**What you've done**:
- ✅ Extract text from PDFs
- ✅ Split into chunks
- ✅ Preserve metadata

---

## DAY 4: Vector Database (2 hours)

### What You'll Learn
- What vectors/embeddings are
- Store searchable vectors
- Do similarity search

### Part 1: Install Qdrant (10 minutes)

```bash
# Install Qdrant client
pip install qdrant-client sentence-transformers

# Install Docker (for Qdrant server)
# Download from: https://www.docker.com/products/docker-desktop

# Start Qdrant in Docker
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### Part 2: Create Vector Store Service (1 hour 30 minutes)

Create file: `vector_store.py`

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class VectorStore:
    """Manage vector embeddings and search"""
    
    def __init__(self, collection_name: str = "papers"):
        # Connect to Qdrant
        self.client = QdrantClient("localhost", port=6333)
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2'
        )
        
        # Create collection if not exists
        self.ensure_collection()
    
    def ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            self.client.get_collection(self.collection_name)
            print(f"✅ Collection '{self.collection_name}' exists")
        except:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,  # Embedding dimension
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created collection '{self.collection_name}'")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Convert texts to embeddings (vectors)"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def store_chunks(
        self,
        chunks: List[Dict],
        paper_id: int
    ) -> List[int]:
        """Store chunk embeddings in Qdrant"""
        # Extract texts
        texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Create points for Qdrant
        points = []
        vector_ids = []
        
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = idx + 1
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "chunk_id": chunk["index"],
                        "paper_id": paper_id,
                        "text": chunk["text"],
                        "section": chunk.get("section", ""),
                        "page": chunk["page"]
                    }
                )
            )
            vector_ids.append(point_id)
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"✅ Stored {len(points)} vectors")
        return vector_ids
    
    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Search for similar chunks"""
        # Embed query
        query_embedding = self.embed_texts([query])[0]
        
        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Format results
        formatted = []
        for result in results:
            formatted.append({
                "text": result.payload["text"],
                "paper_id": result.payload["paper_id"],
                "page": result.payload["page"],
                "score": result.score
            })
        
        return formatted


# Example usage
if __name__ == "__main__":
    # Initialize vector store
    vs = VectorStore()
    
    # Sample chunks
    chunks = [
        {
            "index": 0,
            "text": "The attention mechanism allows models to focus on relevant parts",
            "page": 1
        },
        {
            "index": 1,
            "text": "Transformers use self-attention to process sequences in parallel",
            "page": 2
        }
    ]
    
    # Store
    vector_ids = vs.store_chunks(chunks, paper_id=1)
    print(f"✅ Stored with IDs: {vector_ids}")
    
    # Search
    results = vs.search("What is attention?", top_k=2)
    print(f"\n🔍 Search results:")
    for result in results:
        print(f"  - {result['text'][:50]}... (score: {result['score']:.2f})")
```

**What you've done**:
- ✅ Set up vector database
- ✅ Convert text to embeddings
- ✅ Store and search

---

## DAY 5: RAG Pipeline (3 hours)

### What You'll Learn
- What RAG (Retrieval-Augmented Generation) is
- Connect everything: chunks → LLM → answer

### Part 1: Install LLM (20 minutes)

```bash
# Install Ollama
# Download from: https://ollama.ai/

# Download DeepSeek model
ollama pull deepseek-r1:8b

# Verify
ollama list | grep deepseek
```

### Part 2: Create RAG Pipeline (2 hours 40 minutes)

Create file: `rag_pipeline.py`

```python
import ollama
import time
from vector_store import VectorStore
from database import SessionLocal
from database_models import QueryHistory, Citation, Paper
from typing import Dict, List

class RAGPipeline:
    """Retrieval-Augmented Generation Pipeline"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.ollama_model = "deepseek-r1:8b"
    
    def retrieve_chunks(self, question: str, top_k: int = 5) -> List[Dict]:
        """Step 1: Retrieve relevant chunks from vector DB"""
        chunks = self.vector_store.search(question, top_k=top_k)
        return chunks
    
    def build_context(self, chunks: List[Dict]) -> str:
        """Step 2: Build context from retrieved chunks"""
        context_lines = []
        
        for i, chunk in enumerate(chunks, 1):
            context_lines.append(f"[Source {i}, Page {chunk['page']}]")
            context_lines.append(chunk["text"])
            context_lines.append("")
        
        return "\n".join(context_lines)
    
    def generate_answer(self, question: str, context: str) -> str:
        """Step 3: Use LLM to generate answer"""
        prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        response = ollama.generate(
            model=self.ollama_model,
            prompt=prompt,
            stream=False
        )
        
        return response["response"]
    
    def query(self, question: str, top_k: int = 5) -> Dict:
        """Complete RAG pipeline"""
        start_time = time.time()
        
        # Step 1: Retrieve
        chunks = self.retrieve_chunks(question, top_k=top_k)
        
        # Step 2: Build context
        context = self.build_context(chunks)
        
        # Step 3: Generate answer
        answer = self.generate_answer(question, context)
        
        response_time = time.time() - start_time
        
        # Build response
        response = {
            "question": question,
            "answer": answer,
            "chunks_used": len(chunks),
            "sources": [f"Paper {c['paper_id']}, Page {c['page']}" 
                       for c in chunks],
            "confidence": sum(c["score"] for c in chunks) / len(chunks) if chunks else 0,
            "response_time": response_time
        }
        
        return response
    
    def save_query(self, response: Dict):
        """Save query to database"""
        db = SessionLocal()
        
        try:
            # Save query
            query_record = QueryHistory(
                query_text=response["question"],
                answer=response["answer"],
                response_time=response["response_time"],
                confidence=response["confidence"]
            )
            
            db.add(query_record)
            db.commit()
            
            print(f"✅ Query saved (ID: {query_record.id})")
        
        finally:
            db.close()


# Example usage
if __name__ == "__main__":
    rag = RAGPipeline()
    
    # Ask a question
    result = rag.query("What is attention mechanism?", top_k=3)
    
    print(f"❓ Question: {result['question']}")
    print(f"\n📝 Answer:\n{result['answer']}")
    print(f"\n✅ Confidence: {result['confidence']:.2f}")
    print(f"⏱️ Response time: {result['response_time']:.2f}s")
    print(f"📚 Sources: {', '.join(result['sources'])}")
    
    # Save to database
    rag.save_query(result)
```

**What you've done**:
- ✅ Built RAG pipeline
- ✅ Connected vector search + LLM
- ✅ Generated answers with sources

---

## DAY 6: Complete APIs (2 hours)

### What You'll Learn
- Create all REST endpoints
- Connect to database and services

### Create file: `api_endpoints.py`

```python
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import shutil
import os

from database import get_db
from database_models import Paper, Chunk, QueryHistory, Citation
from pdf_processor import PDFProcessor
from vector_store import VectorStore
from rag_pipeline import RAGPipeline

app = FastAPI(title="CitationAI API")

# Initialize services
pdf_processor = PDFProcessor()
vector_store = VectorStore()
rag_pipeline = RAGPipeline()

# Create uploads directory
os.makedirs("uploads", exist_ok=True)


# ========== PAPER ENDPOINTS ==========

@app.post("/api/papers/upload")
async def upload_paper(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload and process a PDF paper"""
    try:
        # Save file
        file_path = f"uploads/{file.filename}"
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Process PDF
        result = pdf_processor.process_pdf(file_path)
        if not result["success"]:
            raise Exception(result["error"])
        
        # Extract filename without extension
        paper_name = file.filename.replace(".pdf", "")
        
        # Create paper record
        paper = Paper(
            paper_name=paper_name,
            title=result["title"],
            authors=result["author"],
            filename=file.filename,
            file_path=file_path,
            total_pages=result["pages"],
            chunk_count=len(result["chunks"]),
            quality_score=0.9,
            processed=False
        )
        
        db.add(paper)
        db.commit()
        db.refresh(paper)
        
        # Save chunks
        for chunk_data in result["chunks"]:
            chunk = Chunk(
                paper_id=paper.id,
                chunk_index=chunk_data["index"],
                text=chunk_data["text"],
                page_number=chunk_data["page"],
                section="General",
                embedding_generated=False
            )
            db.add(chunk)
        
        db.commit()
        
        # Generate embeddings
        vector_ids = vector_store.store_chunks(result["chunks"], paper.id)
        
        # Update chunk records with vector IDs
        chunks = db.query(Chunk).filter(Chunk.paper_id == paper.id).all()
        for chunk, vector_id in zip(chunks, vector_ids):
            chunk.vector_id = vector_id
            chunk.embedding_generated = True
        
        db.commit()
        
        # Mark as processed
        paper.processed = True
        db.commit()
        
        return {
            "status": "success",
            "paper_id": paper.id,
            "title": paper.title,
            "chunks": len(result["chunks"]),
            "message": f"✅ Paper '{paper.title}' uploaded successfully"
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/papers")
def list_papers(db: Session = Depends(get_db)):
    """List all papers"""
    papers = db.query(Paper).all()
    return {
        "total": len(papers),
        "papers": [
            {
                "id": p.id,
                "title": p.title,
                "chunks": p.chunk_count,
                "processed": p.processed
            }
            for p in papers
        ]
    }


@app.get("/api/papers/{paper_id}")
def get_paper(paper_id: int, db: Session = Depends(get_db)):
    """Get paper details"""
    paper = db.query(Paper).filter(Paper.id == paper_id).first()
    
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    return {
        "id": paper.id,
        "title": paper.title,
        "authors": paper.authors,
        "pages": paper.total_pages,
        "chunks": paper.chunk_count,
        "processed": paper.processed
    }


# ========== QUERY ENDPOINTS ==========

@app.post("/api/query")
def query_papers(
    request: dict,
    db: Session = Depends(get_db)
):
    """Ask a question about papers"""
    try:
        question = request.get("question", "")
        top_k = request.get("top_k", 5)
        
        if not question:
            raise ValueError("Question is required")
        
        # Run RAG pipeline
        result = rag_pipeline.query(question, top_k=top_k)
        
        # Save to database
        query_record = QueryHistory(
            query_text=result["question"],
            answer=result["answer"],
            response_time=result["response_time"],
            confidence=result["confidence"]
        )
        db.add(query_record)
        db.commit()
        
        return {
            "status": "success",
            "answer": result["answer"],
            "confidence": result["confidence"],
            "response_time": f"{result['response_time']:.2f}s",
            "sources": result["sources"]
        }
    
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/queries/history")
def get_query_history(db: Session = Depends(get_db)):
    """Get past queries"""
    queries = db.query(QueryHistory).order_by(
        QueryHistory.created_at.desc()
    ).limit(10).all()
    
    return {
        "total": len(queries),
        "queries": [
            {
                "id": q.id,
                "question": q.query_text[:50],
                "confidence": q.confidence,
                "time": str(q.created_at)
            }
            for q in queries
        ]
    }


# ========== ANALYTICS ENDPOINTS ==========

@app.get("/api/analytics/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    total_papers = db.query(Paper).count()
    total_queries = db.query(QueryHistory).count()
    
    avg_confidence = 0
    if total_queries > 0:
        avg_confidence = db.query(
            func.avg(QueryHistory.confidence)
        ).scalar() or 0
    
    return {
        "total_papers": total_papers,
        "total_queries": total_queries,
        "avg_confidence": f"{avg_confidence:.2f}",
        "status": "✅ System Healthy"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
```

**What you've done**:
- ✅ Upload papers API
- ✅ Query RAG API
- ✅ Analytics API
- ✅ Complete REST system

---

## DAY 7: Testing + Deployment (2 hours)

### Part 1: Test All Endpoints (1 hour)

```bash
# 1. Start all services
# Terminal 1: Qdrant
docker start qdrant

# Terminal 2: Ollama
ollama serve

# Terminal 3: FastAPI
python api_endpoints.py
```

### Part 2: Test via curl

```bash
# Health check
curl http://localhost:8000/

# Upload paper
curl -X POST "http://localhost:8000/api/papers/upload" \
  -F "file=@sample.pdf"

# Query
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this about?", "top_k": 5}'

# Get stats
curl http://localhost:8000/api/analytics/stats
```

### Part 3: Complete File Structure

```
CitationAI/
├── main.py                  # Entry point
├── api_endpoints.py         # REST APIs
├── database.py              # Database setup
├── database_models.py       # SQLAlchemy models
├── pdf_processor.py         # PDF extraction
├── vector_store.py          # Vector DB
├── rag_pipeline.py          # RAG logic
├── requirements.txt         # Dependencies
├── .env                     # Configuration
├── uploads/                 # PDF storage
└── citationai.db           # SQLite database
```

### Part 4: Create requirements.txt

```bash
fastapi
uvicorn
sqlalchemy
pydantic
pdfplumber
pypdf
qdrant-client
sentence-transformers
ollama
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## Testing at Each Step

### Day 1 Test
```bash
curl http://localhost:8000/
# Returns: {"message": "✅ CitationAI Server Running"}
```

### Day 2 Test
```python
python database.py
# Creates: citationai.db ✅
```

### Day 3 Test
```python
python pdf_processor.py
# Extracts text from PDF ✅
```

### Day 4 Test
```python
python vector_store.py
# Stores vectors in Qdrant ✅
```

### Day 5 Test
```python
python rag_pipeline.py
# Generates answers from papers ✅
```

### Day 6 Test
```bash
curl -X POST http://localhost:8000/api/papers/upload -F "file=@paper.pdf"
# Returns: {"status": "success", ...} ✅
```

### Day 7 Test
```bash
# Full system test
# 1. Upload paper
# 2. Ask question
# 3. Check stats
# All working ✅
```

---

## Quick Reference Commands

### Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Start Services
```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
ollama serve
python api_endpoints.py
```

### Test APIs
```bash
# Health
curl http://localhost:8000/

# Upload
curl -X POST http://localhost:8000/api/papers/upload -F "file=@paper.pdf"

# Query
curl -X POST http://localhost:8000/api/query -H "Content-Type: application/json" -d '{"question": "What is this?", "top_k": 5}'

# Analytics
curl http://localhost:8000/api/analytics/stats
```

### Interactive Testing
```bash
http://localhost:8000/docs
# Opens interactive API documentation
```

---

## Deployment Checklist

- [ ] All tests passing
- [ ] Error handling added
- [ ] Logging configured
- [ ] Environment variables set
- [ ] Docker Qdrant running
- [ ] Ollama model downloaded
- [ ] Database backup created
- [ ] Performance tested

---

## What You've Built

A complete **RAG (Retrieval-Augmented Generation) system** that:

1. **Accepts PDFs** → Stores papers
2. **Extracts text** → Chunks content
3. **Creates embeddings** → Stores in vector DB
4. **Answers questions** → Uses LLM + retrieved context
5. **Tracks analytics** → Saves queries
6. **Provides APIs** → RESTful endpoints

**Result**: A system that lets you upload research papers and ask questions about them, getting AI-powered answers with sources!

---

## Next Steps (Optional)

### Advanced Features
- Add user authentication
- Implement caching
- Add web UI (React/Vue)
- Deploy to cloud (AWS, GCP)
- Add more LLM providers
- Multi-language support

### Performance
- Add database indexing
- Implement query caching
- Optimize embeddings
- Add background jobs

### Production
- Use PostgreSQL instead of SQLite
- Add monitoring
- Set up CI/CD pipeline
- Add Docker containerization

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError | Run: `pip install -r requirements.txt` |
| Port 8000 in use | Run: `lsof -i :8000` and kill process |
| Database locked | Delete citationai.db and restart |
| Qdrant connection error | Run: `docker start qdrant` |
| Ollama not found | Install from: https://ollama.ai/ |
| PDF won't process | Try opening in Adobe Reader first |

---

## Estimated Total Time

```
Day 1: 1-2 hours   ✅
Day 2: 2 hours     ✅
Day 3: 2 hours     ✅
Day 4: 2 hours     ✅
Day 5: 3 hours     ✅
Day 6: 2 hours     ✅
Day 7: 2 hours     ✅

Total: ~16 hours (1 week part-time)
```

**Congratulations! You've built CitationAI!** 🎉
