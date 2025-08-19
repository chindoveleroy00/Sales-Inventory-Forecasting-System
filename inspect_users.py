#!/usr/bin/env python3
"""
Script to inspect existing users in the database
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy.orm import Session

try:
    from backend.database.database import get_db, SessionLocal
    from backend.database.models.user_model import User
except ImportError:
    from database.database import get_db, SessionLocal
    from database.models.user_model import User


def inspect_users():
    """Display all users in the database"""

    # Database session
    db = SessionLocal()

    try:
        users = db.query(User).all()

        if not users:
            print("No users found in the database.")
        else:
            print(f"Found {len(users)} user(s):\n")
            print(
                f"{'ID':<5} {'Username':<15} {'Email':<25} {'Role':<10} {'Active':<8} {'Password Hash (first 20 chars)'}")
            print("-" * 90)

            for user in users:
                hash_preview = user.hashed_password[:20] + "..." if user.hashed_password else "None"
                print(
                    f"{user.id:<5} {user.username:<15} {user.email or 'N/A':<25} {user.role:<10} {user.is_active:<8} {hash_preview}")

    except Exception as e:
        print(f"Error inspecting users: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    inspect_users()