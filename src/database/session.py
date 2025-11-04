# src/database/session.py
"""Database session management"""
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import Pool
from sqlalchemy.exc import DisconnectionError
import logging

from src.core.config import settings
from src.database.base import Base

logger = logging.getLogger(__name__)

# Connection args for SQLite
connect_args = {}
if "sqlite" in settings.DATABASE_URL:
    connect_args = {"check_same_thread": False}

# ✅ FIX: Set isolation_level based on database type
if "sqlite" in settings.DATABASE_URL:
    isolation_level = "SERIALIZABLE"  # SQLite uses SERIALIZABLE by default
else:
    isolation_level = "READ COMMITTED"  # PostgreSQL/MySQL use READ COMMITTED

# Create engine with production settings
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=settings.DEBUG,
    isolation_level=isolation_level,  # ✅ Use dynamic isolation level
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)


@event.listens_for(Pool, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign keys for SQLite"""
    if "sqlite" in settings.DATABASE_URL:
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for database sessions.
    
    Usage:
        @router.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    except DisconnectionError as e:
        logger.error(f"Database disconnection error: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Initialize database (create all tables)"""
    # Import all models to register them with Base
    from src.database.models import Paper, Chunk, QueryHistory, Citation
    
    Base.metadata.create_all(bind=engine)
    logger.info("✅ Database tables created")


def dispose_engine() -> None:
    """Dispose of engine and close all connections"""
    engine.dispose()
    logger.info("✅ Database engine disposed")
