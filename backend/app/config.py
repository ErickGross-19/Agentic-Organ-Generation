from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
import os
import sys

class Settings(BaseSettings):
    app_name: str = "MorphoStruct API"
    debug: bool = False

    # LLM provider settings
    llm_provider: str = "anthropic"  # "anthropic" or "openai"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    openai_base_url: str = ""  # Custom OpenAI-compatible endpoint (e.g., Harvard proxy)
    llm_model: str = ""  # Optional: override default model for provider

    # Generation settings
    default_resolution: int = 16
    max_triangles: int = 500000
    generation_timeout_seconds: int = 60  # Timeout for scaffold generation (must be multiple of 30)

    @field_validator('generation_timeout_seconds')
    @classmethod
    def validate_timeout_multiple_of_30(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('generation_timeout_seconds must be positive')
        if v % 30 != 0:
            raise ValueError('generation_timeout_seconds must be a multiple of 30 seconds')
        return v

    # Database settings
    database_url: str = "sqlite:///./morphostruct.db"

    # JWT Authentication settings
    jwt_secret_key: str = ""
    access_token_expire_minutes: int = 1440

    # Encryption for API keys
    encryption_secret: str = ""

    class Config:
        env_file = ".env"

@lru_cache
def get_settings() -> Settings:
    settings = Settings()

    # Validate secrets on load
    ENV = os.getenv("ENV", "development")

    # Check JWT secret
    if ENV != "development":
        if not settings.jwt_secret_key:
            print("FATAL: JWT_SECRET_KEY environment variable is required in production", file=sys.stderr)
            sys.exit(1)
    else:
        if not settings.jwt_secret_key:
            settings.jwt_secret_key = "dev-only-secret-not-for-production"

    # Check encryption secret
    if ENV != "development":
        if not settings.encryption_secret:
            print("FATAL: ENCRYPTION_SECRET environment variable is required in production", file=sys.stderr)
            sys.exit(1)
    else:
        if not settings.encryption_secret:
            settings.encryption_secret = "dev-only-secret-not-for-production"

    return settings
