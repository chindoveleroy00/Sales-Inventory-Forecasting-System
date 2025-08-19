from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from starlette.responses import JSONResponse
from starlette import status

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="SIFS API",
    description="Sales & Inventory Forecasting System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:3000",
    "http://192.168.65.25:8080",  # Add this line - your frontend IP
    "http://192.168.65.25:3000",  # In case you switch Vue dev server ports
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import routers
from .auth import router as auth_router
from .routes import router as api_router

# Include routers
app.include_router(auth_router, prefix="/api/auth")
app.include_router(api_router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    return {"message": "SIFS API is running"}

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )