#!/usr/bin/env python3
"""
Script to update existing user with properly hashed password
"""

import sys
import os
import hashlib

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session

try:
    from backend.database.database import get_db, SessionLocal
    from backend.database.models.user_model import User
except ImportError:
    from database.database import get_db, SessionLocal
    from database.models.user_model import User


def simple_hash_password(password: str) -> str:
    """Simple password hashing - avoiding bcrypt compatibility issues"""
    salt = "sifs_salt_2025"
    return hashlib.sha256((password + salt).encode()).hexdigest()


def update_existing_user():
    """Update existing user with properly hashed password"""

    # Database session
    db = SessionLocal()

    try:
        # Find the existing user with that email
        existing_user = db.query(User).filter(User.email == "admin@sifs.com").first()

        if existing_user:
            print(f"Found existing user: {existing_user.username} ({existing_user.email})")

            # Update the user with new credentials
            existing_user.username = "admin"  # Ensure username is 'admin'
            existing_user.hashed_password = simple_hash_password("admin123")
            existing_user.role = "admin"
            existing_user.is_active = True

            db.commit()
            print("User updated successfully!")

            print("\nUpdated User Credentials:")
            print(f"Username: {existing_user.username}")
            print("Password: admin123")
            print(f"Role: {existing_user.role}")
            print(f"Email: {existing_user.email}")

        else:
            print("No user found with email 'admin@sifs.com'")

            # Let's see what users do exist
            all_users = db.query(User).all()
            print(f"\nFound {len(all_users)} users in database:")
            for user in all_users:
                print(f"- Username: {user.username}, Email: {user.email}, Role: {user.role}")

    except Exception as e:
        print(f"Error updating user: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    update_existing_user()