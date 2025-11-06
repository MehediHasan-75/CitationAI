# src/services/vector_store.py
"""Qdrant Vector Store Service"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Optional
import logging
from src.core.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Qdrant vector database service"""
    
    COLLECTION_NAME = settings.QDRANT_COLLECTION
    
    def __init__(
        self, 
        host: str = settings.QDRANT_HOST, 
        port: int = settings.QDRANT_PORT, 
        dimension: int = settings.EMBEDDING_DIMENSION
    ):
        self.dimension = dimension
        self.collection_name = self.COLLECTION_NAME
        
        try:
            self.client = QdrantClient(host=host, port=port)
            self.ensure_collection()
            logger.info(f"✅ Connected to Qdrant at {host}:{port}")
            self.connected = True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Qdrant: {e}")
            self.connected = False
            self.client = None
    
    def ensure_collection(self):
        """Create collection if it doesn't exist"""
        try:
            # Try to get collection - if it exists, return
            try:
                self.client.get_collection(self.COLLECTION_NAME)
                logger.info(f"✅ Collection exists: {self.COLLECTION_NAME}")
                return
            except:
                pass
            
            # Create collection
            logger.info(f"🔨 Creating collection: {self.COLLECTION_NAME}")
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE
                )
            )
            logger.info(f"✅ Created collection: {self.COLLECTION_NAME}")
        except Exception as e:
            logger.error(f"❌ Collection creation failed: {e}")
    
    def upsert_chunks(
        self, 
        chunks: List[Dict], 
        embeddings: List, 
        paper_id: int
    ) -> List[int]:
        """Store chunk vectors in Qdrant"""
        if not self.connected:
            logger.warning("⚠️ Qdrant not connected - skipping storage")
            return list(range(1, len(chunks) + 1))
        
        try:
            points = []
            vector_ids = []
            
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                point_id = idx + 1
                
                # Convert embedding to list
                if hasattr(embedding, 'tolist'):
                    vector = embedding.tolist()
                else:
                    vector = list(embedding) if not isinstance(embedding, list) else embedding
                
                points.append(PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        'chunk_id': chunk['id'],
                        'paper_id': paper_id,
                        'text': chunk['text'],
                        'section': chunk.get('section', ''),
                        'page': chunk.get('page_number', 0)
                    }
                ))
                vector_ids.append(point_id)
            
            self.client.upsert(
                collection_name=self.COLLECTION_NAME, 
                points=points
            )
            logger.info(f"✅ Stored {len(points)} vectors")
            return vector_ids
        
        except Exception as e:
            logger.error(f"❌ Upsert failed: {e}")
            return []
    
    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 5,
        paper_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """Search similar vectors"""
        if not self.connected:
            logger.warning("⚠️ Qdrant not connected")
            return []
        
        try:
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=top_k
            )
            
            return [
                {
                    'chunk_id': r.payload['chunk_id'],
                    'paper_id': r.payload['paper_id'],
                    'text': r.payload['text'],
                    'section': r.payload['section'],
                    'page': r.payload['page'],
                    'score': r.score
                }
                for r in results
            ]
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def delete_by_ids(self, vector_ids: List[int]) -> None:
        """
        Delete vectors by IDs.
        
        ✅ Simplified to avoid typing issues
        """
        if not self.connected or not vector_ids:
            return
        
        try:
            # ✅ Simple deletion - delete by IDs one by one
            for vid in vector_ids:
                try:
                    self.client.delete(
                        collection_name=self.COLLECTION_NAME,
                        points_selector={"ids": [int(vid)]}
                    )
                except Exception as e:
                    logger.warning(f"Could not delete vector {vid}: {e}")
            
            logger.info(f"✅ Deletion complete")
        except Exception as e:
            logger.error(f"❌ Delete failed: {e}")
            # Don't re-raise - just log


# Global instance
qdrant_service = QdrantService()
