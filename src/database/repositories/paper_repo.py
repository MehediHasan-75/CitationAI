from sqlalchemy.orm import Session
from src.database.models.paper import Paper
from src.database.repositories.base import BaseRepository

class PaperRepository(BaseRepository[Paper]):
    """Repository class for Paper data access and CRUD operations"""

    def __init__(self, db: Session):
        super().__init__(db, Paper)

    def get_by_name(self, paper_name: str) -> Paper | None:
        """
        Retrieve a paper by its unique paper_name.

        Args:
            paper_name (str): the paper_name to search for.

        Returns:
            Paper | None: matching paper or None if not found.
        """
        return self.db.query(Paper).filter(Paper.paper_name == paper_name).first()

    def get_by_filename(self, filename: str) -> Paper | None:
        """
        Retrieve a paper by its filename.

        Args:
            filename (str): filename stored on disk.

        Returns:
            Paper | None: matching paper or None if not found.
        """
        return self.db.query(Paper).filter(Paper.filename == filename).first()

    def check_duplicate(self, paper_name: str, filename: str) -> Paper | None:
        """
        Check if a paper with the same paper_name or filename exists.

        Args:
            paper_name (str): paper's unique name.
            filename (str): filename uploaded/stored.

        Returns:
            Paper | None: existing paper if duplicate found, else None.
        """
        existing = self.get_by_name(paper_name)
        if existing:
            return existing
        existing = self.get_by_filename(filename)
        if existing:
            return existing
        return None

    def mark_processed(self, paper: Paper, chunk_count: int) -> None:
        """
        Mark a paper as fully processed and update chunk count.

        Args:
            paper (Paper): the Paper ORM instance.
            chunk_count (int): total number of chunks created.
        """
        paper.processed = True
        paper.chunk_count = chunk_count
        self.db.commit()

    def get_all(self, skip: int = 0, limit: int = 10, processed_only=False):
        """
        Retrieve all papers with optional processing status filter, with pagination.

        Args:
            skip (int): number of records to skip.
            limit (int): maximum records to return.
            processed_only (bool): if True, filter only processed papers.

        Returns:
            List[Paper]: list of Paper ORM instances.
        """
        query = self.db.query(Paper)
        if processed_only:
            query = query.filter(Paper.processed == True)
        return query.order_by(Paper.created_at.desc()).offset(skip).limit(limit).all()

    def get_count(self, processed_only=False) -> int:
        """
        Count total number of papers with optional filter.

        Args:
            processed_only (bool): if True, count only processed papers.

        Returns:
            int: total count.
        """
        query = self.db.query(Paper)
        if processed_only:
            query = query.filter(Paper.processed == True)
        return query.count()

    def get_by_id(self, paper_id: int) -> Paper | None:
        """
        Retrieve a paper by its primary key id.

        Args:
            paper_id (int): the id of the paper.

        Returns:
            Paper | None: matching paper or None if not found.
        """
        return self.db.query(Paper).filter(Paper.id == paper_id).first()

    def delete(self, paper_id: int) -> None:
        """
        Delete a paper by id, cascade deletes handle associated chunks etc.

        Args:
            paper_id (int): id of the paper to delete.
        """
        paper_obj = self.get_by_id(paper_id)
        if not paper_obj:
            return
        self.db.delete(paper_obj)
        self.db.commit()
