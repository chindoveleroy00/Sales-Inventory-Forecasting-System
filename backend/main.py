from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from passlib.context import CryptContext
import logging

# Import your database components - try different import styles if needed
try:
    from backend.database.database import engine, Base, get_db
    from backend.database.models.user_model import User
    from backend.api.routes import router as api_router
    from backend.api.auth import get_password_hash
except ImportError:
    # Alternative import style if running from different directory
    from database.database import engine, Base, get_db
    from database.models.user_model import User
    from api.routes import router as api_router
    from api.auth import get_password_hash

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SIFS API",
    description="Sales & Inventory Forecasting System API",
    version="1.0.0",
)

# CORS (Cross-Origin Resource Sharing) configuration
origins = [
    "http://localhost:8080",  # Your Vue.js frontend development server
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Initialization ---
@app.on_event("startup")
def on_startup():

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables checked/created.")

    with Session(engine) as db:
        users_to_create = {
            "admin": {"password": "adminpass", "full_name": "Administrator", "email": "admin@sifs.com", "role": "admin"},
            "sales": {"password": "salespass", "full_name": "Sales User", "email": "sales@sifs.com", "role": "sales"},
            "inventory": {"password": "inventorypass", "full_name": "Inventory User", "email": "inventory@sifs.com", "role": "inventory"},
            "dataentry": {"password": "datapass", "full_name": "Data Entry User", "email": "dataentry@sifs.com", "role": "dataentry"},
        }

        for username, user_data in users_to_create.items():
            if not db.query(User).filter(User.username == username).first():
                hashed_password = get_password_hash(user_data["password"])
                new_user = User(
                    username=username,
                    hashed_password=hashed_password,
                    full_name=user_data["full_name"],
                    email=user_data["email"],
                    role=user_data["role"]
                )
                db.add(new_user)
                db.commit()
                db.refresh(new_user)
                logger.info(f"Default user '{username}' created.")
            else:
                logger.info(f"User '{username}' already exists.")


# Include your API routes
app.include_router(api_router, prefix="/api")

# Add a simple root endpoint for testing API availability
@app.get("/")
async def read_root():
    return {"message": "SIFS API is running!"}

# You might remove or modify this if your initial forecast pipeline run
# is now triggered via an API endpoint or a separate scheduled task.
# @app.on_event("startup")
# async def run_initial_forecast():
#     from forecasting.run_forecast import run_forecast_pipeline
#     logger.info("Running initial forecast pipeline...")
#     forecast = run_forecast_pipeline()
#     logger.info("Initial forecast completed.")