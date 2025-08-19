from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path
import sys

try:
    from backend.database.models.base import Base
except ImportError:
    from database.models.base import Base

# Import the config from config/database.py
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.database import DATABASE_URL

# Create a SQLAlchemy engine
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Dependency to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables (add this at the end)
def create_tables():
    Base.metadata.create_all(bind=engine)