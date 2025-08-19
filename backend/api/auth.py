from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from sqlalchemy.orm import Session
import logging
import hashlib

# Initialize router
router = APIRouter(tags=["Authentication"])

# Set up logging
logger = logging.getLogger(__name__)

# Security configurations
SECRET_KEY = "your-secret-key"  # Change in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

try:
    from backend.database.database import get_db
    from backend.database.models.user_model import User
except ImportError:
    from database.database import get_db
    from database.models.user_model import User
from .models import Token, LoginRequest


@router.post("/login", response_model=Token)
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username, "role": user.role},
        expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "role": user.role.capitalize()  # Changed from "user_role" to "role"
    }


# Utility functions using simple hashing (avoiding bcrypt issues)
def get_password_hash(password: str):
    """Simple password hashing - FOR DEVELOPMENT"""
    salt = "sifs_salt_2025"
    return hashlib.sha256((password + salt).encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str):
    """
    Verify a plain password against a hashed password.
    Uses simple hashing to avoid bcrypt compatibility issues.
    """
    try:
        # Try the simple hash method
        salt = "sifs_salt_2025"
        expected_hash = hashlib.sha256((plain_password + salt).encode()).hexdigest()
        if expected_hash == hashed_password:
            return True

        # If that fails, try plain text comparison (for existing data)
        if plain_password == hashed_password:
            logger.warning("Password matched as plain text - this is insecure!")
            return True

        return False
    except Exception as e:
        logger.error(f"Error verifying password: {e}")
        return False


def authenticate_user(db: Session, username: str, password: str):
    """
    Authenticate a user by username and password.
    """
    user = db.query(User).filter(User.username == username).first()
    if not user:
        logger.info(f"User not found: {username}")
        return None

    if not verify_password(password, user.hashed_password):
        logger.info(f"Password verification failed for user: {username}")
        return None

    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user