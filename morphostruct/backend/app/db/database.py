"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

from ..core.logging import get_logger

logger = get_logger(__name__)

def mask_db_url(url: str) -> str:
    """Mask password in database URL for safe logging."""
    from urllib.parse import urlparse, urlunparse
    try:
        parsed = urlparse(url)
        if parsed.password:
            # Replace password with asterisks
            netloc = f"{parsed.username}:****@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            masked = parsed._replace(netloc=netloc)
            return urlunparse(masked)
        return url
    except Exception:
        # If parsing fails, just show the scheme
        return url.split("://")[0] + "://****" if "://" in url else "****"

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./morphostruct.db")
logger.info(f"Database URL configured: {mask_db_url(DATABASE_URL)}")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database tables."""
    try:
        logger.info("Initializing database tables")
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def seed_default_user():
    """Create a default user if none exists (development convenience)."""
    db = SessionLocal()
    try:
        from ..models.user import User, UserPreferences
        from ..services.auth import get_password_hash

        existing = db.query(User).filter(User.username == "Erick").first()
        if existing:
            logger.info("Default user 'Erick' already exists, skipping seed")
            return

        user = User(
            username="Erick",
            email="ErickGross1924@gmail.com",
            hashed_password=get_password_hash("Abcd1234"),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        preferences = UserPreferences(user_id=user.id)
        db.add(preferences)
        db.commit()

        logger.info("Default user 'Erick' created successfully")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to seed default user: {e}")
    finally:
        db.close()
