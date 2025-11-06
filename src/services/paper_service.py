# src/services/paper_service.py
"""
Paper Service - Business logic for paper processing.

Orchestrates:
- File validation and storage
- PDF processing and metadata extraction
- Chunk creation
- Embedding generation
- Qdrant storage
"""

from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session
import logging

from src.database.repositories.paper_repo import PaperRepository
from src.database.repositories.chunk_repository import ChunkRepository
from src.services.file_service import file_service
from src.services.pdf_processor import pdf_processor
from src.services.chunking_service import intelligent_chunker
from src.services.embedding_service import embedding_service
from src.services.vector_store import qdrant_service
from src.utils.validators import validate_pdf_file, extract_filename_stem
from src.database.models import Paper, Chunk

logger = logging.getLogger(__name__)


class PaperService:
    """Service for paper processing business logic"""
    
    def __init__(
        self,
        db: Session,
        paper_repo: PaperRepository,
        chunk_repo: ChunkRepository
    ):
        self.db = db
        self.paper_repo = paper_repo
        self.chunk_repo = chunk_repo
    
    # ==================== CREATE (UPLOAD) ====================
    
    def process_and_save_paper(self, file: UploadFile) -> dict:
        """
        End-to-end paper processing pipeline.
        
        Steps:
        1. Validate file
        2. Check duplicates
        3. Save file to disk
        4. Extract PDF metadata
        5. Create paper record
        6. Create chunks
        7. Generate embeddings
        8. Store in Qdrant
        9. Mark as processed
        
        Args:
            file: Uploaded PDF file
            
        Returns:
            Success response with paper metadata
            
        Raises:
            HTTPException: If any step fails
        """
        # 1️⃣ Validate
        validate_pdf_file(file)
        paper_name = extract_filename_stem(file.filename)
        logger.info(f"📝 Processing paper: {paper_name}")
        
        # 2️⃣ Check duplicates
        existing = self.paper_repo.check_duplicate(paper_name, file.filename)
        if existing:
            return self._duplicate_response(existing)
        
        # 3️⃣ Save file
        file_path = file_service.save_upload(file)
        
        try:
            # 4️⃣ Process PDF
            processed_doc = self._process_pdf(file_path)
            
            # 5️⃣ Save to database
            paper = self._create_paper_record(
                processed_doc, paper_name, file.filename, file_path
            )
            
            # 6️⃣ Create chunks
            chunk_count = self._create_chunks(paper, processed_doc, paper_name)
            
            # 7️⃣ Generate embeddings
            self._generate_embeddings(paper, paper_name)
            
            # 8️⃣ Mark processed
            self.paper_repo.mark_processed(paper, chunk_count)
            
            logger.info(f"✅ Paper processed: {paper_name} ({chunk_count} chunks)")
            
            return self._success_response(paper, processed_doc, chunk_count)
        
        except Exception as e:
            file_service.delete_file(file_path)  # Cleanup
            logger.error(f"❌ Paper processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Paper processing failed: {str(e)}"
            )
    
    # ==================== READ ====================
    
    def get_papers(
        self, 
        skip: int = 0, 
        limit: int = 10, 
        processed_only: bool = False
    ) -> dict:
        """
        Get list of papers with pagination.
        
        Args:
            skip: Pagination offset
            limit: Max results
            processed_only: Filter for processed papers
            
        Returns:
            Dictionary with papers list and metadata
        """
        papers = self.paper_repo.get_all(skip, limit, processed_only)
        total = self.paper_repo.count(processed_only)
        
        return {
            "papers": [self._format_paper_summary(p) for p in papers],
            "total": total,
            "skip": skip,
            "limit": limit
        }
    
    def get_paper_by_id(self, paper_id: int) -> Paper:
        """
        Get specific paper by ID.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            Paper object
            
        Raises:
            HTTPException: If paper not found
        """
        paper = self.paper_repo.get_by_id(paper_id)
        if not paper:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Paper with id {paper_id} not found"
            )
        return paper
    
    def search_papers(self, query: str, limit: int = 10) -> dict:
        """
        Search papers by title or name.
        
        Args:
            query: Search term
            limit: Max results
            
        Returns:
            Dictionary with search results
        """
        papers = self.paper_repo.search_by_title(query, limit)
        return {
            "query": query,
            "results": [self._format_paper_summary(p) for p in papers],
            "count": len(papers)
        }
    
    def get_paper_stats(self, paper_id: int) -> dict:
        """
        Get detailed statistics for a paper.
        
        Args:
            paper_id: Paper ID
            
        Returns:
            Statistics dictionary
        """
        paper = self.get_paper_by_id(paper_id)
        chunks = self.chunk_repo.get_by_paper_id(paper_id)
        
        total_chunks = len(chunks)
        embedded_chunks = sum(1 for c in chunks if c.embedding_generated)
        
        # Section distribution
        section_distribution = {}
        for chunk in chunks:
            section = chunk.section or "Unknown"
            section_distribution[section] = section_distribution.get(section, 0) + 1
        
        # Avg chunk length
        avg_length = sum(len(c.text) for c in chunks) / total_chunks if total_chunks > 0 else 0
        
        return {
            "paper_id": paper.id,
            "paper_name": paper.paper_name,
            "title": paper.title,
            "statistics": {
                "total_chunks": total_chunks,
                "embedded_chunks": embedded_chunks,
                "embedding_completion": round(embedded_chunks / total_chunks * 100, 2) if total_chunks > 0 else 0,
                "average_chunk_length": round(avg_length, 0),
                "section_distribution": section_distribution,
                "quality_score": paper.quality_score
            }
        }
    
    def get_overall_stats(self) -> dict:
        """
        Get overall system statistics.
        
        Returns:
            System-wide statistics
        """
        return self.paper_repo.get_stats()
    
    # ==================== DELETE ====================
    
    def delete_paper(self, paper_id: int) -> None:
        """
        Delete paper and all associated data.
        
        What gets deleted:
        - Paper record from database
        - All chunks (cascade delete via foreign key)
        - All embeddings from Qdrant
        - PDF file from disk
        
        Args:
            paper_id: Paper ID
            
        Raises:
            HTTPException: If deletion fails
        """
        # 1. Get paper
        paper = self.paper_repo.get_by_id(paper_id)
        if not paper:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Paper with id {paper_id} not found"
            )
        
        logger.info(f"🗑️ Deleting paper: {paper.paper_name} (ID: {paper_id})")
        
        try:
            # 2. Get all chunks (to get vector_ids)
            chunks = self.chunk_repo.get_by_paper_id(paper_id)
            
            # 3. Delete from Qdrant
            if chunks:
                vector_ids = [c.vector_id for c in chunks if c.vector_id]
                if vector_ids:
                    qdrant_service.delete_by_ids(vector_ids)
                    logger.info(f"✅ Deleted {len(vector_ids)} vectors from Qdrant")
            
            # 4. Delete file from disk
            try:
                file_service.delete_file(paper.file_path)
                logger.info(f"✅ Deleted file: {paper.file_path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete file: {e}")
            
            # 5. Delete from database (cascades to chunks)
            self.paper_repo.delete(paper_id)
            logger.info(f"✅ Deleted paper from database: {paper.paper_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to delete paper: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to delete paper: {str(e)}"
            )
    
    # ==================== HELPER METHODS ====================
    
    def _process_pdf(self, file_path: str):
        """Process PDF and extract metadata"""
        try:
            processed_doc = pdf_processor.process_document(file_path)
            if not processed_doc:
                raise ValueError("PDF processing returned None")
            return processed_doc
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"PDF processing failed: {str(e)}"
            )
    
    def _create_paper_record(self, processed_doc, paper_name: str, 
                            filename: str, file_path: str) -> Paper:
        """Create paper database record"""
        try:
            return self.paper_repo.create(
                paper_name=paper_name,
                title=processed_doc.title or "Unknown",
                authors=processed_doc.authors or [],
                year=processed_doc.year,
                filename=filename,
                file_path=file_path,
                total_pages=processed_doc.total_pages or 0,
                abstract=processed_doc.abstract or "",
                sections=self._format_sections(processed_doc.sections),
                quality_score=processed_doc.quality_metrics.get('overall_quality', 0),
                keywords=processed_doc.keywords or [],
                format_type=processed_doc.format_type or "standard",
                processed=False
            )
        except Exception as e:
            logger.error(f"❌ Failed to save paper: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save paper: {str(e)}"
            )
    
    def _create_chunks(self, paper: Paper, processed_doc, paper_name: str) -> int:
        """Create and save chunks"""
        chunk_count = 0
        chunks_batch = []
        
        for section in processed_doc.sections:
            section_chunks = intelligent_chunker.chunk_section(
                section_text=section.text,
                section_name=section.name,
                page_start=section.page_start,
                page_end=section.page_end,
                paper_name=paper_name,
                section_id=section.section_id,
                section_level=section.level
            )
            
            for chunk in section_chunks:
                db_chunk = Chunk(
                    paper_id=paper.id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    section=chunk.section,
                    page_number=chunk.page_number,
                    section_id=chunk.section_id,
                    section_level=chunk.section_level
                )
                chunks_batch.append(db_chunk)
                chunk_count += 1
                
                # Batch commit every 10 chunks
                if len(chunks_batch) >= 10:
                    self.chunk_repo.create_batch(chunks_batch)
                    chunks_batch = []
        
        # Commit remaining
        if chunks_batch:
            self.chunk_repo.create_batch(chunks_batch)
        
        return chunk_count
    
    def _generate_embeddings(self, paper: Paper, paper_name: str) -> None:
        """Generate embeddings and store in Qdrant"""
        chunks = self.chunk_repo.get_by_paper_id(paper.id)
        
        if not chunks:
            logger.warning(f"⚠️ No chunks for paper {paper_name}")
            return
        
        # Prepare data
        chunk_dicts = [
            {
                'id': c.id,
                'text': c.text,
                'section': c.section,
                'page_number': c.page_number,
                'paper_name': paper_name,
                'section_id': c.section_id,
                'section_level': c.section_level
            }
            for c in chunks
        ]
        
        # Generate embeddings
        texts = [c['text'] for c in chunk_dicts]
        embeddings = embedding_service.encode_batch(texts)
        
        # Store in Qdrant
        vector_ids = qdrant_service.upsert_chunks(chunk_dicts, embeddings, paper.id)
        
        # Update database
        self.chunk_repo.update_vector_ids(chunks, vector_ids)
    
    def _format_sections(self, sections):
        """Format sections for database storage"""
        return [
            {
                "name": s.name,
                "page_start": s.page_start,
                "page_end": s.page_end,
                "section_id": getattr(s, "section_id", None),
                "level": getattr(s, "level", 0),
            }
            for s in sections
        ] if sections else []
    
    def _format_paper_summary(self, paper: Paper) -> dict:
        """Format paper for list view"""
        return {
            "id": paper.id,
            "paper_name": paper.paper_name,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "total_pages": paper.total_pages,
            "chunk_count": paper.chunk_count,
            "quality_score": paper.quality_score,
            "processed": paper.processed,
            "created_at": paper.created_at.isoformat()
        }
    
    def _duplicate_response(self, paper: Paper) -> dict:
        """Format duplicate paper response"""
        return {
            "paper_id": paper.id,
            "paper_name": paper.paper_name,
            "title": paper.title,
            "status": "already_exists",
            "message": "This paper was already uploaded",
            "uploaded": paper.created_at.isoformat()
        }
    
    def _success_response(self, paper: Paper, processed_doc, chunk_count: int) -> dict:
        """Format successful upload response"""
        return {
            "paper_id": paper.id,
            "paper_name": paper.paper_name,
            "title": paper.title,
            "authors": paper.authors,
            "year": paper.year,
            "keywords": paper.keywords,
            "quality_score": paper.quality_score,
            "format_type": paper.format_type,
            "sections_extracted": len(processed_doc.sections),
            "chunks_created": chunk_count,
            "pages": paper.total_pages,
            "upload_status": "success",
            "message": f"Successfully uploaded and processed {paper.paper_name}"
        }
