import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models.schema import Base


def init_db(db_path: str) -> None:
    """Create DuckDB database and all tables if they don't exist."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    engine = create_engine(f"duckdb:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    print(f"✓ Database initialized at {db_path}")


def get_session(db_path: str) -> Session:
    """Get a new SQLAlchemy session connected to DuckDB."""
    engine = create_engine(f"duckdb:///{db_path}", echo=False)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    return SessionLocal()


def close_session(session: Session) -> None:
    """Safely close a session."""
    if session:
        session.close()
