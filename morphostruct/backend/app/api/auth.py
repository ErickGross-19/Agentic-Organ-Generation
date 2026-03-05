"""Authentication API endpoints."""
from datetime import datetime, timedelta
from typing import Optional, Dict
import json
import os
from urllib.parse import urlparse
import ipaddress
import socket
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..services.auth import (
    authenticate_user,
    create_access_token,
    create_user,
    decode_token,
    get_user_by_username,
    get_user_by_id,
)
from ..services.encryption import encrypt_api_key, decrypt_api_key
from ..models.user import User, UserApiKey, UserPreferences

# Optional Redis for production rate limiting
try:
    import redis
    REDIS_URL = os.getenv("REDIS_URL")
    redis_client: Optional[redis.Redis] = redis.from_url(REDIS_URL) if REDIS_URL else None
except ImportError:
    redis_client = None

# Trusted proxy IPs for X-Forwarded-For validation
TRUSTED_PROXY_IPS = os.getenv("TRUSTED_PROXY_IPS", "").split(",") if os.getenv("TRUSTED_PROXY_IPS") else []

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)


# ============================================================================
# SSRF Protection
# ============================================================================

def is_safe_url(url: str) -> tuple[bool, str, str | None]:
    """
    Check if URL is safe to request (not internal/private).
    Returns (is_safe, error_message, resolved_ip).
    The resolved_ip can be used to pin the connection and prevent DNS rebinding.
    """
    try:
        parsed = urlparse(url)
        hostname = parsed.hostname

        if not hostname:
            return False, "Invalid URL: no hostname", None

        if parsed.scheme not in ("http", "https"):
            return False, "Invalid URL: must be http or https", None

        # Block obviously internal hostnames
        blocked_patterns = [
            "localhost", "127.0.0.1", "0.0.0.0",
            "169.254", "metadata", "internal",
            ".local", ".internal", ".lan"
        ]
        hostname_lower = hostname.lower()
        for pattern in blocked_patterns:
            if pattern in hostname_lower:
                return False, f"URL blocked: contains '{pattern}'", None

        # Resolve hostname and validate IP - REQUIRED (no pass-through on failure)
        try:
            resolved_ip = socket.gethostbyname(hostname)
            ip = ipaddress.ip_address(resolved_ip)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, f"URL blocked: resolves to private/internal IP {resolved_ip}", None
            return True, "", resolved_ip
        except socket.gaierror:
            return False, f"URL blocked: hostname '{hostname}' could not be resolved", None

    except Exception as e:
        return False, f"URL validation error: {str(e)}", None


# ============================================================================
# Rate Limiting
# ============================================================================

class RateLimiter:
    """
    Rate limiter with Redis support for production multi-instance deployments.
    Falls back to in-memory storage for development.
    """

    def __init__(self):
        self.use_redis = redis_client is not None
        # In-memory fallback for development
        self._memory_store: Dict[str, Dict[str, tuple[int, datetime]]] = {}

    def _get_key(self, ip: str, action: str) -> str:
        return f"ratelimit:{action}:{ip}"

    def _check_redis(self, ip: str, action: str, max_attempts: int, window_minutes: int) -> tuple[bool, int]:
        """Check rate limit using Redis."""
        key = self._get_key(ip, action)

        # Get current count
        current = redis_client.get(key)
        if current is None:
            # First attempt - set with expiry
            redis_client.setex(key, window_minutes * 60, 1)
            return True, max_attempts - 1

        count = int(current)
        if count >= max_attempts:
            ttl = redis_client.ttl(key)
            return False, 0

        # Increment
        redis_client.incr(key)
        return True, max_attempts - count - 1

    def _check_memory(self, ip: str, action: str, max_attempts: int, window_minutes: int) -> tuple[bool, int]:
        """Check rate limit using in-memory storage (original implementation)."""
        now = datetime.utcnow()
        window = timedelta(minutes=window_minutes)

        if action not in self._memory_store:
            self._memory_store[action] = {}

        if ip in self._memory_store[action]:
            count, first_attempt = self._memory_store[action][ip]
            if now - first_attempt > window:
                # Window expired, reset
                self._memory_store[action][ip] = (1, now)
                return True, max_attempts - 1
            elif count >= max_attempts:
                return False, 0
            else:
                self._memory_store[action][ip] = (count + 1, first_attempt)
                return True, max_attempts - count - 1
        else:
            self._memory_store[action][ip] = (1, now)
            return True, max_attempts - 1

    def check(self, ip: str, action: str, max_attempts: int = 5, window_minutes: int = 15) -> tuple[bool, int]:
        """
        Check if request is allowed under rate limit.
        Returns (allowed, remaining_attempts).
        """
        if self.use_redis:
            try:
                return self._check_redis(ip, action, max_attempts, window_minutes)
            except Exception:
                # Fallback to memory on Redis error
                return self._check_memory(ip, action, max_attempts, window_minutes)
        return self._check_memory(ip, action, max_attempts, window_minutes)

    def reset(self, ip: str, action: str) -> None:
        """Reset rate limit for an IP/action."""
        if self.use_redis:
            try:
                redis_client.delete(self._get_key(ip, action))
            except Exception:
                pass
        if action in self._memory_store and ip in self._memory_store[action]:
            del self._memory_store[action][ip]

    def check_rate_limit(self, ip: str, endpoint: str, max_attempts: int, window_minutes: int = 1) -> bool:
        """
        Legacy method for backwards compatibility.
        Check if request should be allowed.
        Returns True if allowed, False if rate limit exceeded.
        """
        allowed, _ = self.check(ip, endpoint, max_attempts, window_minutes)
        return allowed

    def cleanup_old_entries(self, max_age_minutes: int = 60):
        """
        Remove entries older than max_age_minutes to prevent memory bloat.
        Only applies to in-memory storage (Redis handles expiry automatically).
        """
        if self.use_redis:
            return  # Redis handles expiry automatically

        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=max_age_minutes)

        # Collect actions to remove
        actions_to_remove = []
        for action, ips in self._memory_store.items():
            ips_to_remove = []
            for ip, (_, first_attempt) in ips.items():
                if first_attempt < cutoff:
                    ips_to_remove.append(ip)

            # Remove old IPs
            for ip in ips_to_remove:
                del ips[ip]

            # If action has no more IPs, mark for removal
            if not ips:
                actions_to_remove.append(action)

        # Remove empty actions
        for action in actions_to_remove:
            del self._memory_store[action]


# Global rate limiter instance
rate_limiter = RateLimiter()


def get_client_ip(request: Request) -> str:
    """
    Get client IP address, only trusting X-Forwarded-For from configured proxies.
    """
    client_host = request.client.host if request.client else None

    # Only trust X-Forwarded-For if request comes from a trusted proxy
    if client_host and TRUSTED_PROXY_IPS and client_host in TRUSTED_PROXY_IPS:
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            # Take the first IP (original client) from the chain
            return forwarded.split(",")[0].strip()

    return client_host or "unknown"


def check_rate_limit(request: Request, endpoint: str, max_attempts: int):
    """Dependency to check rate limits."""
    ip = get_client_ip(request)

    # Periodically cleanup old entries (every request has small chance)
    import random
    if random.random() < 0.01:  # 1% chance per request
        rate_limiter.cleanup_old_entries()

    if not rate_limiter.check_rate_limit(ip, endpoint, max_attempts):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many {endpoint} attempts. Please try again later.",
            headers={"Retry-After": "60"}
        )


# ============================================================================
# Request/Response Models
# ============================================================================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(
        ...,
        min_length=8,
        description="Password must be at least 8 characters with at least one uppercase, one lowercase, and one number"
    )
    email: Optional[str] = None

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class UserResponse(BaseModel):
    id: int
    uuid: str
    username: str
    email: Optional[str]
    is_active: bool

class ApiKeyRequest(BaseModel):
    provider: str = Field(..., pattern="^(openai|anthropic)$")
    api_key: str

class ApiKeyResponse(BaseModel):
    id: int
    provider: str
    has_key: bool

class PreferencesRequest(BaseModel):
    # Column-based fields (stored in dedicated columns)
    default_scaffold_type: Optional[str] = None
    theme: Optional[str] = None
    auto_generate: Optional[bool] = None

    # Appearance preferences
    accent_color: Optional[str] = None
    sidebar_position: Optional[str] = None
    compact_mode: Optional[bool] = None
    show_tooltips: Optional[bool] = None

    # Defaults
    default_porosity: Optional[float] = None
    default_wall_thickness: Optional[float] = None
    auto_save_drafts: Optional[bool] = None
    auto_save_interval: Optional[str] = None
    generation_timeout_seconds: Optional[int] = None  # Must be multiple of 30

    # Viewer preferences
    camera_type: Optional[str] = None
    show_grid: Optional[bool] = None
    show_axis_helpers: Optional[bool] = None
    grid_snap: Optional[bool] = None
    grid_snap_size: Optional[float] = None
    background_color: Optional[str] = None
    ambient_occlusion: Optional[bool] = None
    anti_aliasing: Optional[bool] = None

    # Export preferences
    default_export_format: Optional[str] = None
    include_textures: Optional[bool] = None
    auto_download_after_generation: Optional[bool] = None
    default_filename_pattern: Optional[str] = None
    coordinate_system: Optional[str] = None

    # Units preferences
    measurement_units: Optional[str] = None
    default_bbox_x: Optional[float] = None
    default_bbox_y: Optional[float] = None
    default_bbox_z: Optional[float] = None
    default_resolution: Optional[str] = None
    show_dimensions_in_viewport: Optional[bool] = None

    # Accessibility preferences
    reduced_motion: Optional[bool] = None
    high_contrast_mode: Optional[bool] = None
    larger_text: Optional[bool] = None
    screen_reader_descriptions: Optional[bool] = None
    keyboard_shortcuts_enabled: Optional[bool] = None
    show_keyboard_shortcut_hints: Optional[bool] = None

    # LLM provider selection
    default_provider: Optional[str] = None
    custom_endpoint: Optional[str] = None

class PreferencesResponse(BaseModel):
    # Column-based fields
    default_scaffold_type: str
    theme: str
    auto_generate: bool

    # Appearance preferences
    accent_color: Optional[str] = None
    sidebar_position: Optional[str] = None
    compact_mode: Optional[bool] = None
    show_tooltips: Optional[bool] = None

    # Defaults
    default_porosity: Optional[float] = None
    default_wall_thickness: Optional[float] = None
    auto_save_drafts: Optional[bool] = None
    auto_save_interval: Optional[str] = None
    generation_timeout_seconds: Optional[int] = None  # Must be multiple of 30

    # Viewer preferences
    camera_type: Optional[str] = None
    show_grid: Optional[bool] = None
    show_axis_helpers: Optional[bool] = None
    grid_snap: Optional[bool] = None
    grid_snap_size: Optional[float] = None
    background_color: Optional[str] = None
    ambient_occlusion: Optional[bool] = None
    anti_aliasing: Optional[bool] = None

    # Export preferences
    default_export_format: Optional[str] = None
    include_textures: Optional[bool] = None
    auto_download_after_generation: Optional[bool] = None
    default_filename_pattern: Optional[str] = None
    coordinate_system: Optional[str] = None

    # Units preferences
    measurement_units: Optional[str] = None
    default_bbox_x: Optional[float] = None
    default_bbox_y: Optional[float] = None
    default_bbox_z: Optional[float] = None
    default_resolution: Optional[str] = None
    show_dimensions_in_viewport: Optional[bool] = None

    # Accessibility preferences
    reduced_motion: Optional[bool] = None
    high_contrast_mode: Optional[bool] = None
    larger_text: Optional[bool] = None
    screen_reader_descriptions: Optional[bool] = None
    keyboard_shortcuts_enabled: Optional[bool] = None
    show_keyboard_shortcut_hints: Optional[bool] = None

    # LLM provider selection
    default_provider: Optional[str] = None
    custom_endpoint: Optional[str] = None

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(
        ...,
        min_length=8,
        description="Password must be at least 8 characters with at least one uppercase, one lowercase, and one number"
    )

    @field_validator('new_password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v

class ChangePasswordResponse(BaseModel):
    message: str

class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None

class UpdateProfileResponse(BaseModel):
    id: int
    uuid: str
    username: str
    email: Optional[str]
    display_name: Optional[str]

class TestApiKeyRequest(BaseModel):
    provider: str = Field(..., pattern="^(openai|anthropic|custom)$")
    api_key: str
    base_url: Optional[str] = Field(default=None, alias="custom_endpoint")

    class Config:
        populate_by_name = True

class TestApiKeyResponse(BaseModel):
    success: bool
    message: str
    model_info: Optional[str] = None

class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    last_active: str
    ip_address: Optional[str] = None

class DeleteSessionResponse(BaseModel):
    message: str


# ============================================================================
# Auth Dependencies
# ============================================================================

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """Get the current authenticated user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )

    user = get_user_by_id(db, int(user_id))
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
        )

    return user

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    if not credentials:
        return None
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


# ============================================================================
# Endpoints
# ============================================================================

@router.post("/register", response_model=TokenResponse)
async def register(
    request_body: RegisterRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    # Check rate limit: 3 attempts per minute
    check_rate_limit(request, "register", max_attempts=3)

    # Check if username exists
    if get_user_by_username(db, request_body.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )

    # Create user
    user = create_user(db, request_body.username, request_body.password, request_body.email)

    # Generate token
    token = create_access_token({"sub": str(user.id), "username": user.username})

    return TokenResponse(
        access_token=token,
        user={"id": user.id, "uuid": user.uuid, "username": user.username}
    )

@router.post("/login", response_model=TokenResponse)
async def login(
    request_body: LoginRequest,
    request: Request,
    db: Session = Depends(get_db)
):
    """Login and get access token."""
    # Check rate limit: 5 attempts per minute
    check_rate_limit(request, "login", max_attempts=5)

    user = authenticate_user(db, request_body.username, request_body.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = create_access_token({"sub": str(user.id), "username": user.username})

    return TokenResponse(
        access_token=token,
        user={"id": user.id, "uuid": user.uuid, "username": user.username}
    )

@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info."""
    return UserResponse(
        id=current_user.id,
        uuid=current_user.uuid,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active
    )

@router.get("/profile", response_model=UpdateProfileResponse)
async def get_profile(
    current_user: User = Depends(get_current_user)
):
    """Get user profile with extended information."""
    return UpdateProfileResponse(
        id=current_user.id,
        uuid=current_user.uuid,
        username=current_user.username,
        email=current_user.email,
        display_name=current_user.display_name
    )

@router.post("/api-keys", response_model=ApiKeyResponse)
async def save_api_key(
    request: ApiKeyRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save or update an API key for a provider."""
    # Check for existing key
    existing = db.query(UserApiKey).filter(
        UserApiKey.user_id == current_user.id,
        UserApiKey.provider == request.provider
    ).first()

    encrypted = encrypt_api_key(request.api_key)

    if existing:
        existing.encrypted_key = encrypted
        db.commit()
        return ApiKeyResponse(id=existing.id, provider=existing.provider, has_key=True)

    # Create new
    api_key = UserApiKey(
        user_id=current_user.id,
        provider=request.provider,
        encrypted_key=encrypted
    )
    db.add(api_key)
    db.commit()
    db.refresh(api_key)

    return ApiKeyResponse(id=api_key.id, provider=api_key.provider, has_key=True)

@router.get("/api-keys", response_model=list[ApiKeyResponse])
async def get_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's API keys (without revealing the actual keys)."""
    keys = db.query(UserApiKey).filter(UserApiKey.user_id == current_user.id).all()
    return [ApiKeyResponse(id=k.id, provider=k.provider, has_key=True) for k in keys]

@router.delete("/api-keys/{provider}")
async def delete_api_key(
    provider: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an API key."""
    key = db.query(UserApiKey).filter(
        UserApiKey.user_id == current_user.id,
        UserApiKey.provider == provider
    ).first()

    if not key:
        raise HTTPException(status_code=404, detail="API key not found")

    db.delete(key)
    db.commit()
    return {"message": "API key deleted"}

@router.get("/preferences", response_model=PreferencesResponse)
async def get_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user preferences."""
    prefs = db.query(UserPreferences).filter(
        UserPreferences.user_id == current_user.id
    ).first()

    if not prefs:
        # Create default preferences
        prefs = UserPreferences(user_id=current_user.id)
        db.add(prefs)
        db.commit()
        db.refresh(prefs)

    # Parse extended preferences from JSON
    extended_prefs = {}
    if prefs.preferences_json:
        try:
            extended_prefs = json.loads(prefs.preferences_json)
        except json.JSONDecodeError:
            extended_prefs = {}

    # Return response with column-based fields and extended preferences
    return PreferencesResponse(
        # Column-based fields
        default_scaffold_type=prefs.default_scaffold_type,
        theme=prefs.theme,
        auto_generate=prefs.auto_generate,

        # Extended preferences from JSON (with None defaults if not present)
        accent_color=extended_prefs.get("accent_color"),
        sidebar_position=extended_prefs.get("sidebar_position"),
        compact_mode=extended_prefs.get("compact_mode"),
        show_tooltips=extended_prefs.get("show_tooltips"),

        default_porosity=extended_prefs.get("default_porosity"),
        default_wall_thickness=extended_prefs.get("default_wall_thickness"),
        auto_save_drafts=extended_prefs.get("auto_save_drafts"),
        auto_save_interval=extended_prefs.get("auto_save_interval"),

        camera_type=extended_prefs.get("camera_type"),
        show_grid=extended_prefs.get("show_grid"),
        show_axis_helpers=extended_prefs.get("show_axis_helpers"),
        grid_snap=extended_prefs.get("grid_snap"),
        grid_snap_size=extended_prefs.get("grid_snap_size"),
        background_color=extended_prefs.get("background_color"),
        ambient_occlusion=extended_prefs.get("ambient_occlusion"),
        anti_aliasing=extended_prefs.get("anti_aliasing"),

        default_export_format=extended_prefs.get("default_export_format"),
        include_textures=extended_prefs.get("include_textures"),
        auto_download_after_generation=extended_prefs.get("auto_download_after_generation"),
        default_filename_pattern=extended_prefs.get("default_filename_pattern"),
        coordinate_system=extended_prefs.get("coordinate_system"),

        measurement_units=extended_prefs.get("measurement_units"),
        default_bbox_x=extended_prefs.get("default_bbox_x"),
        default_bbox_y=extended_prefs.get("default_bbox_y"),
        default_bbox_z=extended_prefs.get("default_bbox_z"),
        default_resolution=extended_prefs.get("default_resolution"),
        show_dimensions_in_viewport=extended_prefs.get("show_dimensions_in_viewport"),

        reduced_motion=extended_prefs.get("reduced_motion"),
        high_contrast_mode=extended_prefs.get("high_contrast_mode"),
        larger_text=extended_prefs.get("larger_text"),
        screen_reader_descriptions=extended_prefs.get("screen_reader_descriptions"),
        keyboard_shortcuts_enabled=extended_prefs.get("keyboard_shortcuts_enabled"),
        show_keyboard_shortcut_hints=extended_prefs.get("show_keyboard_shortcut_hints"),

        default_provider=extended_prefs.get("default_provider"),
        custom_endpoint=extended_prefs.get("custom_endpoint")
    )

@router.put("/preferences", response_model=PreferencesResponse)
async def update_preferences(
    request: PreferencesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user preferences."""
    prefs = db.query(UserPreferences).filter(
        UserPreferences.user_id == current_user.id
    ).first()

    if not prefs:
        prefs = UserPreferences(user_id=current_user.id)
        db.add(prefs)

    # Update column-based fields
    if request.default_scaffold_type is not None:
        prefs.default_scaffold_type = request.default_scaffold_type
    if request.theme is not None:
        prefs.theme = request.theme
    if request.auto_generate is not None:
        prefs.auto_generate = request.auto_generate

    # Load existing extended preferences
    extended_prefs = {}
    if prefs.preferences_json:
        try:
            extended_prefs = json.loads(prefs.preferences_json)
        except json.JSONDecodeError:
            extended_prefs = {}

    # Update extended preferences (only fields that are not None in the request)
    request_dict = request.model_dump(exclude_none=True)

    # List of column-based fields to exclude from JSON storage
    column_fields = {"default_scaffold_type", "theme", "auto_generate"}

    # Update extended_prefs with fields from request (except column fields)
    for field, value in request_dict.items():
        if field not in column_fields:
            extended_prefs[field] = value

    # Save updated extended preferences as JSON
    prefs.preferences_json = json.dumps(extended_prefs)

    db.commit()
    db.refresh(prefs)

    # Return full response
    return PreferencesResponse(
        # Column-based fields
        default_scaffold_type=prefs.default_scaffold_type,
        theme=prefs.theme,
        auto_generate=prefs.auto_generate,

        # Extended preferences from JSON
        accent_color=extended_prefs.get("accent_color"),
        sidebar_position=extended_prefs.get("sidebar_position"),
        compact_mode=extended_prefs.get("compact_mode"),
        show_tooltips=extended_prefs.get("show_tooltips"),

        default_porosity=extended_prefs.get("default_porosity"),
        default_wall_thickness=extended_prefs.get("default_wall_thickness"),
        auto_save_drafts=extended_prefs.get("auto_save_drafts"),
        auto_save_interval=extended_prefs.get("auto_save_interval"),

        camera_type=extended_prefs.get("camera_type"),
        show_grid=extended_prefs.get("show_grid"),
        show_axis_helpers=extended_prefs.get("show_axis_helpers"),
        grid_snap=extended_prefs.get("grid_snap"),
        grid_snap_size=extended_prefs.get("grid_snap_size"),
        background_color=extended_prefs.get("background_color"),
        ambient_occlusion=extended_prefs.get("ambient_occlusion"),
        anti_aliasing=extended_prefs.get("anti_aliasing"),

        default_export_format=extended_prefs.get("default_export_format"),
        include_textures=extended_prefs.get("include_textures"),
        auto_download_after_generation=extended_prefs.get("auto_download_after_generation"),
        default_filename_pattern=extended_prefs.get("default_filename_pattern"),
        coordinate_system=extended_prefs.get("coordinate_system"),

        measurement_units=extended_prefs.get("measurement_units"),
        default_bbox_x=extended_prefs.get("default_bbox_x"),
        default_bbox_y=extended_prefs.get("default_bbox_y"),
        default_bbox_z=extended_prefs.get("default_bbox_z"),
        default_resolution=extended_prefs.get("default_resolution"),
        show_dimensions_in_viewport=extended_prefs.get("show_dimensions_in_viewport"),

        reduced_motion=extended_prefs.get("reduced_motion"),
        high_contrast_mode=extended_prefs.get("high_contrast_mode"),
        larger_text=extended_prefs.get("larger_text"),
        screen_reader_descriptions=extended_prefs.get("screen_reader_descriptions"),
        keyboard_shortcuts_enabled=extended_prefs.get("keyboard_shortcuts_enabled"),
        show_keyboard_shortcut_hints=extended_prefs.get("show_keyboard_shortcut_hints"),

        default_provider=extended_prefs.get("default_provider"),
        custom_endpoint=extended_prefs.get("custom_endpoint")
    )

@router.post("/change-password", response_model=ChangePasswordResponse)
async def change_password(
    request: ChangePasswordRequest,
    request_obj: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change user's password."""
    from ..services.auth import verify_password, get_password_hash

    # Rate limiting: 5 attempts per 15 minutes
    client_ip = get_client_ip(request_obj)
    allowed, remaining = rate_limiter.check(client_ip, "change_password", max_attempts=5, window_minutes=15)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please try again later."
        )

    # Verify current password
    if not verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )

    # Update to new password
    current_user.hashed_password = get_password_hash(request.new_password)
    db.commit()

    return ChangePasswordResponse(message="Password changed successfully")

@router.put("/profile", response_model=UpdateProfileResponse)
async def update_profile(
    request: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user's profile information."""
    if request.display_name is not None:
        current_user.display_name = request.display_name

    if request.email is not None:
        # Check if email is already taken by another user
        existing = db.query(User).filter(
            User.email == request.email,
            User.id != current_user.id
        ).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already in use"
            )
        current_user.email = request.email

    db.commit()
    db.refresh(current_user)

    return UpdateProfileResponse(
        id=current_user.id,
        uuid=current_user.uuid,
        username=current_user.username,
        email=current_user.email,
        display_name=current_user.display_name
    )

@router.post("/test-api-key", response_model=TestApiKeyResponse)
async def test_api_key(
    request: TestApiKeyRequest,
    request_obj: Request,
    current_user: User = Depends(get_current_user)
):
    """Test an API key by making a simple request to the provider."""
    import httpx

    # Rate limiting: 10 attempts per 5 minutes (more generous for testing multiple providers)
    client_ip = get_client_ip(request_obj)
    allowed, remaining = rate_limiter.check(client_ip, "test_api_key", max_attempts=10, window_minutes=5)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please try again later."
        )

    try:
        if request.provider == "openai":
            base_url = request.base_url or "https://api.openai.com/v1"
            headers = {
                "Authorization": f"Bearer {request.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    model_count = len(data.get("data", []))
                    return TestApiKeyResponse(
                        success=True,
                        message="OpenAI API key is valid",
                        model_info=f"Found {model_count} available models"
                    )
                else:
                    return TestApiKeyResponse(
                        success=False,
                        message=f"API key test failed: {response.status_code}"
                    )

        elif request.provider == "anthropic":
            base_url = request.base_url or "https://api.anthropic.com/v1"
            headers = {
                "x-api-key": request.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            }

            # Anthropic doesn't have a models endpoint, so we'll test with a minimal message
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{base_url}/messages",
                    headers=headers,
                    json={
                        "model": "claude-3-haiku-20240307",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "Hi"}]
                    }
                )

                if response.status_code == 200:
                    return TestApiKeyResponse(
                        success=True,
                        message="Anthropic API key is valid",
                        model_info="Successfully authenticated with Claude API"
                    )
                else:
                    return TestApiKeyResponse(
                        success=False,
                        message=f"API key test failed: {response.status_code}"
                    )

        elif request.provider == "custom":
            # Custom OpenAI-compatible endpoint
            if not request.base_url:
                return TestApiKeyResponse(
                    success=False,
                    message="Custom provider requires a base URL"
                )

            # SSRF Protection: validate URL before making request
            is_safe, error_msg, resolved_ip = is_safe_url(request.base_url)
            if not is_safe:
                return TestApiKeyResponse(
                    success=False,
                    message=f"Security error: {error_msg}"
                )

            # Normalize the base URL - remove /chat/completions if present
            base_url = request.base_url.rstrip('/')
            if base_url.endswith('/chat/completions'):
                base_url = base_url[:-len('/chat/completions')]
            if not base_url.endswith('/v1'):
                # If the URL doesn't end with /v1, check if it's included elsewhere
                if '/v1' not in base_url:
                    base_url = base_url + '/v1'

            headers = {
                "Authorization": f"Bearer {request.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(timeout=15.0) as client:
                # First try /models endpoint (standard OpenAI compatible)
                try:
                    response = await client.get(
                        f"{base_url}/models",
                        headers=headers
                    )

                    if response.status_code == 200:
                        data = response.json()
                        model_count = len(data.get("data", []))
                        return TestApiKeyResponse(
                            success=True,
                            message="Custom endpoint connection successful",
                            model_info=f"Found {model_count} available models"
                        )
                except Exception:
                    pass

                # Fallback: test with a minimal chat completion
                try:
                    response = await client.post(
                        f"{base_url}/chat/completions",
                        headers=headers,
                        json={
                            "model": "gpt-4",  # Common model name
                            "max_tokens": 1,
                            "messages": [{"role": "user", "content": "Hi"}]
                        }
                    )

                    if response.status_code == 200:
                        return TestApiKeyResponse(
                            success=True,
                            message="Custom endpoint connection successful",
                            model_info="Successfully tested chat completions"
                        )
                    elif response.status_code == 401:
                        return TestApiKeyResponse(
                            success=False,
                            message="Authentication failed - check your API key"
                        )
                    elif response.status_code == 404:
                        return TestApiKeyResponse(
                            success=False,
                            message="Endpoint not found - check your base URL"
                        )
                    else:
                        error_detail = ""
                        try:
                            error_data = response.json()
                            error_detail = error_data.get("error", {}).get("message", "")
                        except Exception:
                            pass
                        return TestApiKeyResponse(
                            success=False,
                            message=f"API test failed ({response.status_code}): {error_detail or 'Unknown error'}"
                        )
                except Exception as e:
                    return TestApiKeyResponse(
                        success=False,
                        message=f"Connection failed: {str(e)}"
                    )

        return TestApiKeyResponse(
            success=False,
            message="Unknown provider"
        )

    except httpx.TimeoutException:
        return TestApiKeyResponse(
            success=False,
            message="Request timed out - check your network connection"
        )
    except Exception as e:
        return TestApiKeyResponse(
            success=False,
            message=f"Error testing API key: {str(e)}"
        )

@router.get("/sessions", response_model=list[SessionResponse])
async def get_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's active sessions."""
    from ..models.user import UserSession

    # Get all active sessions for the user
    sessions = db.query(UserSession).filter(
        UserSession.user_id == current_user.id,
        UserSession.is_active == True
    ).all()

    return [
        SessionResponse(
            session_id=str(session.id),
            created_at=session.created_at.isoformat(),
            last_active=session.last_active.isoformat(),
            ip_address=session.ip_address
        )
        for session in sessions
    ]

@router.delete("/sessions/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke a specific session."""
    from ..models.user import UserSession

    # Find the session by ID and verify it belongs to the current user
    try:
        session_id_int = int(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session ID"
        )

    session = db.query(UserSession).filter(
        UserSession.id == session_id_int,
        UserSession.user_id == current_user.id
    ).first()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Mark session as inactive instead of deleting it (for audit trail)
    session.is_active = False
    db.commit()

    return DeleteSessionResponse(message=f"Session {session_id} revoked successfully")

@router.post("/sessions/revoke-all", response_model=DeleteSessionResponse)
async def revoke_all_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke all sessions for the current user."""
    from ..models.user import UserSession

    # Mark all user's sessions as inactive
    updated_count = db.query(UserSession).filter(
        UserSession.user_id == current_user.id,
        UserSession.is_active == True
    ).update({"is_active": False})

    db.commit()

    return DeleteSessionResponse(
        message=f"All sessions revoked successfully ({updated_count} session(s) revoked)"
    )

@router.post("/send-verification", response_model=dict)
async def send_verification_email(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send email verification link to user."""
    # Note: This is a placeholder implementation
    # In production, this would:
    # 1. Generate a verification token
    # 2. Send email with verification link
    # 3. Mark user as verified when they click the link
    return {"message": "Verification email sent"}

class DeleteAccountRequest(BaseModel):
    password: str

@router.delete("/account", response_model=DeleteSessionResponse)
async def delete_account(
    request: DeleteAccountRequest,
    request_obj: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete user account permanently."""
    from ..services.auth import verify_password

    # Rate limiting: 5 attempts per 15 minutes
    client_ip = get_client_ip(request_obj)
    allowed, remaining = rate_limiter.check(client_ip, "delete_account", max_attempts=5, window_minutes=15)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please try again later."
        )

    # Verify password
    if not verify_password(request.password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect password"
        )

    # Delete user (cascade will delete related data)
    db.delete(current_user)
    db.commit()

    return DeleteSessionResponse(message="Account deleted successfully")
