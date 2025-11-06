# src/api/routers/queries.py
"""
Query History & Analytics Router

Endpoints for:
- Query history management
- Analytics and statistics
- Most popular queries
- Paper citation tracking
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from datetime import datetime, timedelta

from src.database.session import get_db
from src.database.models import QueryHistory, Citation, Paper
from src.services.rag_pipeline import RAGPipeline
from src.services.vector_store import qdrant_service
from src.services.embedding_service import embedding_service
from pydantic import BaseModel

import logging

logger = logging.getLogger(__name__)


# ==================== SCHEMAS ====================

class QueryRequest(BaseModel):
    """Request model for RAG query"""
    question: str
    top_k: int = 5
    paper_ids: Optional[List[int]] = None


class QueryResponse(BaseModel):
    """Response model for RAG query"""
    answer: str
    citations: List[dict]
    sources_used: List[str]
    confidence: float
    response_time: float


# ==================== ROUTER ====================

router = APIRouter(
    prefix='/api',
    tags=['Query & Analytics']
)


# ==================== RAG QUERY ====================

@router.post('/query', response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_papers(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Ask questions to the RAG system.
    
    The system will:
    1. Encode your question as a vector
    2. Search for similar chunks in the vector database
    3. Use LLM to generate an answer based on retrieved chunks
    4. Return answer with citations
    
    **Query Parameters:**
    - question: Your question about the papers (required)
    - top_k: Number of chunks to retrieve (default: 5)
    - paper_ids: Filter search to specific papers (optional)
    
    **Returns:**
    - answer: AI-generated answer
    - citations: Source papers and sections
    - confidence: Answer confidence score (0-1)
    - response_time: Processing time in seconds
    
    **Example:**
    ```
    {
      "question": "What is the attention mechanism?",
      "top_k": 5,
      "paper_ids":[1]
    }
    ```
    """
    try:
        # ✅ Create RAG pipeline
        rag = RAGPipeline(qdrant_service, embedding_service, db)
        
        # ✅ Generate answer with RAG
        result = rag.generate_answer(
            question=request.question,
            top_k=request.top_k,
            paper_ids=request.paper_ids
        )
        
        # ✅ Save query to history
        query_record = QueryHistory(
            query_text=request.question,
            answer=result['answer'],
            response_time=result['response_time'],
            confidence=result['confidence'],
            top_k=request.top_k
        )
        db.add(query_record)
        db.commit()
        
        # ✅ Save citations
        for citation_data in result['citations']:
            citation = Citation(
                query_id=query_record.id,
                paper_id=citation_data['paper_id'],
                relevance_score=citation_data['relevance_score']
            )
            db.add(citation)
        
        db.commit()
        logger.info(f"✅ Query processed: {request.question[:50]}...")
        
        return QueryResponse(
            answer=result['answer'],
            citations=result['citations'],
            sources_used=result['sources_used'],
            confidence=result['confidence'],
            response_time=result['response_time']
        )
    
    except Exception as e:
        logger.error(f"❌ Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


# ==================== QUERY HISTORY ====================

@router.get('/queries/history', status_code=status.HTTP_200_OK)
def get_query_history(
    skip: int = 0,
    limit: int = 50,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get query history with pagination and time filtering.
    
    **Query Parameters:**
    - skip: Pagination offset (default: 0)
    - limit: Max results (default: 50)
    - days: Only queries from last N days (default: 30)
    
    **Returns:**
    - List of past queries with metadata
    - Total count
    """
    try:
        # Filter by date
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        query = db.query(QueryHistory).filter(
            QueryHistory.created_at >= cutoff_date
        ).order_by(QueryHistory.created_at.desc())
        
        total = query.count()
        queries = query.offset(skip).limit(limit).all()
        
        return {
            "queries": [
                {
                    "id": q.id,
                    "question": q.query_text,
                    "answer": q.answer[:200] + "..." if len(q.answer or "") > 200 else q.answer,
                    "confidence": q.confidence,
                    "response_time": q.response_time,
                    "created_at": q.created_at.isoformat(),
                    "rating": q.user_rating
                }
                for q in queries
            ],
            "total": total,
            "skip": skip,
            "limit": limit,
            "days_filter": days
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to get query history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


# ==================== QUERY DETAILS ====================

@router.get('/queries/{query_id}', status_code=status.HTTP_200_OK)
def get_query_details(
    query_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific query.
    
    **Returns:**
    - Full question and answer
    - All citations with paper details
    - Performance metrics
    """
    try:
        query = db.query(QueryHistory).filter(QueryHistory.id == query_id).first()
        
        if not query:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query {query_id} not found"
            )
        
        # Get citations
        citations = db.query(Citation).filter(Citation.query_id == query_id).all()
        
        return {
            "id": query.id,
            "question": query.query_text,
            "answer": query.answer,
            "confidence": query.confidence,
            "response_time": query.response_time,
            "top_k": query.top_k,
            "user_rating": query.user_rating,
            "created_at": query.created_at.isoformat(),
            "citations": [
                {
                    "paper_id": c.paper_id,
                    "paper_title": c.paper.title if c.paper else "Unknown",
                    "relevance_score": c.relevance_score
                }
                for c in citations
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get query details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve query: {str(e)}"
        )


# ==================== ANALYTICS ====================

@router.get('/analytics/popular', status_code=status.HTTP_200_OK)
def get_popular_queries(
    limit: int = 10,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get most frequently asked questions.
    
    **Query Parameters:**
    - limit: Number of queries to return (default: 10)
    - days: Time window in days (default: 30)
    
    **Returns:**
    - Most asked questions with frequency
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        popular = db.query(
            QueryHistory.query_text,
            func.count(QueryHistory.id).label('count'),
            func.avg(QueryHistory.confidence).label('avg_confidence')
        ).filter(
            QueryHistory.created_at >= cutoff_date
        ).group_by(
            QueryHistory.query_text
        ).order_by(
            func.count(QueryHistory.id).desc()
        ).limit(limit).all()
        
        return {
            "popular_queries": [
                {
                    "question": q[0],
                    "times_asked": q[1],
                    "avg_confidence": round(q[2], 3) if q[2] else 0
                }
                for q in popular
            ],
            "period_days": days,
            "total_queries": db.query(func.count(QueryHistory.id)).filter(
                QueryHistory.created_at >= cutoff_date
            ).scalar()
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to get popular queries: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


# ==================== CITATIONS ANALYTICS ====================

@router.get('/analytics/papers/most-cited', status_code=status.HTTP_200_OK)
def get_most_cited_papers(
    limit: int = 10,
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get papers cited most frequently in queries.
    
    **Returns:**
    - Papers ranked by citation count
    - Average relevance scores
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        most_cited = db.query(
            Paper.id,
            Paper.title,
            Paper.paper_name,
            func.count(Citation.id).label('citation_count'),
            func.avg(Citation.relevance_score).label('avg_relevance')
        ).join(
            Citation, Citation.paper_id == Paper.id
        ).join(
            QueryHistory, QueryHistory.id == Citation.query_id
        ).filter(
            QueryHistory.created_at >= cutoff_date
        ).group_by(
            Paper.id, Paper.title, Paper.paper_name
        ).order_by(
            func.count(Citation.id).desc()
        ).limit(limit).all()
        
        return {
            "most_cited_papers": [
                {
                    "paper_id": p[0],
                    "title": p[1],
                    "paper_name": p[2],
                    "citation_count": p[3],
                    "avg_relevance_score": round(p[4], 3) if p[4] else 0
                }
                for p in most_cited
            ],
            "period_days": days
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to get citation analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve analytics: {str(e)}"
        )


# ==================== SYSTEM ANALYTICS ====================

@router.get('/analytics/stats', status_code=status.HTTP_200_OK)
def get_system_stats(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get overall system statistics.
    
    **Returns:**
    - Total queries, papers, and citations
    - Average confidence and response time
    - Activity trends
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        total_queries = db.query(func.count(QueryHistory.id)).filter(
            QueryHistory.created_at >= cutoff_date
        ).scalar()
        
        avg_confidence = db.query(func.avg(QueryHistory.confidence)).filter(
            QueryHistory.created_at >= cutoff_date
        ).scalar()
        
        avg_response_time = db.query(func.avg(QueryHistory.response_time)).filter(
            QueryHistory.created_at >= cutoff_date
        ).scalar()
        
        total_papers = db.query(func.count(Paper.id)).filter(
            Paper.processed == True
        ).scalar()
        
        avg_rating = db.query(func.avg(QueryHistory.user_rating)).filter(
            QueryHistory.user_rating.isnot(None),
            QueryHistory.created_at >= cutoff_date
        ).scalar()
        
        return {
            "statistics": {
                "total_queries": total_queries or 0,
                "total_papers": total_papers or 0,
                "avg_confidence": round(avg_confidence, 3) if avg_confidence else 0,
                "avg_response_time_seconds": round(avg_response_time, 3) if avg_response_time else 0,
                "avg_user_rating": round(avg_rating, 2) if avg_rating else 0
            },
            "period_days": days,
            "period_start": cutoff_date.isoformat()
        }
    
    except Exception as e:
        logger.error(f"❌ Failed to get system stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve stats: {str(e)}"
        )
