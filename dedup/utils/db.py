from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import os
import logging

logger = logging.getLogger(__name__)

# Global engine instance (singleton with connection pooling)
_engine = None
_SessionLocal = None


def get_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables."""
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "dedup")
    db_user = os.getenv("DB_USER", "dedup")
    db_password = os.getenv("DB_PASSWORD", "dedup_local_dev")

    return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def init_db() -> None:
    """Initialize PostgreSQL database and create all tables."""
    global _engine, _SessionLocal

    conn_string = get_connection_string()

    try:
        # Create engine with connection pooling
        _engine = create_engine(
            conn_string,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,  # Test connections before use
            echo=False,
        )

        # Import after engine creation to avoid circular imports
        from models import Base

        # Create all tables
        Base.metadata.create_all(_engine)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

        logger.info(f"✓ Database initialized: {db_host}:{db_port}/{db_name}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize database: {e}")
        raise


def get_session() -> Session:
    """Get a new SQLAlchemy session connected to PostgreSQL."""
    global _engine, _SessionLocal

    if _engine is None:
        init_db()

    return _SessionLocal()


def close_session(session: Session) -> None:
    """Safely close a session."""
    if session:
        session.close()


def close_all() -> None:
    """Close all database connections (cleanup on exit)."""
    global _engine
    if _engine:
        _engine.dispose()
