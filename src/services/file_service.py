# src/services/file_service.py
import os
import uuid
import logging
from fastapi import UploadFile, HTTPException, status
from src.core.config import settings

logger = logging.getLogger(__name__)

class FileService:
    """Service for file storage and management"""
    
    def __init__(self):
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    
    def save_upload(self, file: UploadFile) -> str:  
        """
        Save uploaded file to disk with unique filename.
        
        Returns:
            str: Full path to saved file
        
        Raises:
            HTTPException: If file save fails
        """
        file_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4()}.pdf")
        
        try:
            content = file.file.read()
            
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            logger.info(f"✅ File saved to: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"❌ File save failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"File save failed: {str(e)}"
            )
    
    def delete_file(self, file_path: str) -> None:  
        """Delete file from disk"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"✅ Deleted file: {file_path}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete {file_path}: {e}")

# Singleton instance
file_service = FileService()
