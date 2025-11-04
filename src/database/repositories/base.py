from typing import TypeVar, Generic, List, Optional
from sqlalchemy.orm import Session

ModelType = TypeVar("ModelType")


class BaseRepository(Generic[ModelType]):
    
    def __init__(self, db: Session, model: type):
        self.db = db
        self.model = model
    
    def create(self, **kwargs) -> ModelType:
        obj = self.model(**kwargs)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj
    
    def get_by_id(self, id: int) -> Optional[ModelType]:
        return self.db.query(self.model).filter(self.model.id == id).first()
    
    def get_all(self, skip: int = 0, limit: int = 10) -> List[ModelType]:
        return self.db.query(self.model).offset(skip).limit(limit).all()
    
    def delete(self, id: int) -> None:
        obj = self.get_by_id(id)
        if obj:
            self.db.delete(obj)
            self.db.commit()
    
    def create_batch(self, objects: List) -> None:
        self.db.add_all(objects)
        self.db.commit()
