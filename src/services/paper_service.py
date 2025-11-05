# src/services/paper_service.py
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session
import logging

from src.repositories.paper_repository import PaperRepository
from src.repositories.chunk_repository import ChunkRepository
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
    
    def process_and_save_paper(self, file: UploadFile) -> dict:  # ✅ Changed from async def to def
        """End-to-end paper processing pipeline."""
        
        # 1️⃣ Validate
        validate_pdf_file(file)
        paper_name = extract_filename_stem(file.filename)
        logger.info(f"📝 Processing paper: {paper_name}")
        
        # 2️⃣ Check duplicates
        existing = self.paper_repo.check_duplicate(paper_name, file.filename)
        if existing:
            return self._duplicate_response(existing)
        
        # 3️⃣ Save file (remove await)
        file_path = file_service.save_upload(file)  # ✅ Removed await
        
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
    
    def _duplicate_response(self, paper: Paper) -> dict:
        """Format duplicate paper response"""
        return {
            "paper_id": paper.id,
            "paper_name": paper.paper_name,
            "title": paper.title,
            "status": "already_exists",
            "message": "This paper was already uploaded",
            "uploaded": paper.created_at.isoformat()  # ✅ Changed from upload_date to created_at
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
