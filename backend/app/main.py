import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.config import get_settings
from app.api.scaffolds import router as scaffolds_router
from app.api.chat import router as chat_router
from app.api.vision import router as vision_router
from app.db.database import init_db
from app.api import auth, saved_scaffolds
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    version="1.0.0",
    description="REST API for bioprinting scaffold generation with LLM-assisted design",
)

# Get allowed origins from environment, default to localhost for development
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


# Security headers middleware
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        # Only add HSTS in production (when not localhost)
        if request.url.hostname not in ("localhost", "127.0.0.1"):
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


app.add_middleware(SecurityHeadersMiddleware)

# Include routers
app.include_router(scaffolds_router)
app.include_router(chat_router)
app.include_router(vision_router)
app.include_router(auth.router)
app.include_router(saved_scaffolds.router)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting MorphoStruct backend API server")
    logger.info(f"App name: {settings.app_name}, Debug: {settings.debug}")
    init_db()
    logger.info("Database initialized successfully")


@app.get("/")
async def root():
    return {
        "message": "Bioprint Scaffold Generator API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "scaffolds": "/api/generate, /api/preview, /api/validate, /api/export/{id}, /api/presets",
            "chat": "/api/chat",
            "vision": "/api/vision/analyze, /api/vision/status",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "debug": settings.debug,
    }
