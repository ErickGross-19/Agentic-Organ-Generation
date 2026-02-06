"""Saved scaffolds API endpoints."""
import os
import json
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from datetime import datetime

from ..db.database import get_db
from ..models.user import SavedScaffold, User
from ..api.auth import get_current_user

router = APIRouter(prefix="/api/scaffolds", tags=["scaffolds"])

# Storage directory for user files
STORAGE_DIR = os.getenv("STORAGE_DIR", "./user_data")


def safe_join(base_dir: str, filename: str) -> str:
    """Safely join paths, preventing directory traversal attacks."""
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Resolve to absolute paths
    base = os.path.realpath(base_dir)
    full_path = os.path.realpath(os.path.join(base, filename))

    # Ensure the result is still within base_dir
    if not full_path.startswith(base + os.sep) and full_path != base:
        raise ValueError("Path traversal attempt detected")

    return full_path


def get_user_storage_dir(user_uuid: str) -> str:
    """Get storage directory for a user."""
    path = os.path.join(STORAGE_DIR, user_uuid)
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# Request/Response Models
# ============================================================================

class SaveScaffoldRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    scaffold_type: str
    parameters: dict

class ScaffoldResponse(BaseModel):
    id: int
    uuid: str
    name: str
    type: str = Field(alias='scaffold_type')  # Accept scaffold_type from DB, output as 'type'
    parameters: dict
    thumbnail_path: Optional[str]
    thumbnail_url: Optional[str] = None  # Alias for thumbnail_path for frontend compatibility
    stl_path: Optional[str]
    created_at: datetime
    updated_at: datetime
    user_id: Optional[int] = None

    class Config:
        populate_by_name = True  # Allow both 'type' and 'scaffold_type'

class ScaffoldListResponse(BaseModel):
    scaffolds: List[ScaffoldResponse]
    total: int


# ============================================================================
# Endpoints
# ============================================================================

@router.post("", response_model=ScaffoldResponse)
async def save_scaffold(
    request: SaveScaffoldRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Save a scaffold design."""
    scaffold = SavedScaffold(
        user_id=current_user.id,
        name=request.name,
        scaffold_type=request.scaffold_type,
        parameters_json=json.dumps(request.parameters)
    )
    db.add(scaffold)
    db.commit()
    db.refresh(scaffold)

    return ScaffoldResponse(
        id=scaffold.id,
        uuid=scaffold.uuid,
        name=scaffold.name,
        type=scaffold.scaffold_type,
        parameters=json.loads(scaffold.parameters_json),
        thumbnail_path=scaffold.thumbnail_path,
        thumbnail_url=scaffold.thumbnail_path,
        stl_path=scaffold.stl_path,
        created_at=scaffold.created_at,
        updated_at=scaffold.updated_at,
        user_id=scaffold.user_id
    )

@router.get("", response_model=ScaffoldListResponse)
async def list_scaffolds(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List user's saved scaffolds."""
    scaffolds = db.query(SavedScaffold).filter(
        SavedScaffold.user_id == current_user.id
    ).order_by(SavedScaffold.updated_at.desc()).all()

    return ScaffoldListResponse(
        scaffolds=[
            ScaffoldResponse(
                id=s.id,
                uuid=s.uuid,
                name=s.name,
                type=s.scaffold_type,
                parameters=json.loads(s.parameters_json),
                thumbnail_path=s.thumbnail_path,
                thumbnail_url=s.thumbnail_path,
                stl_path=s.stl_path,
                created_at=s.created_at,
                updated_at=s.updated_at,
                user_id=s.user_id
            )
            for s in scaffolds
        ],
        total=len(scaffolds)
    )

@router.get("/{scaffold_uuid}", response_model=ScaffoldResponse)
async def get_scaffold(
    scaffold_uuid: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific scaffold."""
    scaffold = db.query(SavedScaffold).filter(
        SavedScaffold.uuid == scaffold_uuid,
        SavedScaffold.user_id == current_user.id
    ).first()

    if not scaffold:
        raise HTTPException(status_code=404, detail="Scaffold not found")

    return ScaffoldResponse(
        id=scaffold.id,
        uuid=scaffold.uuid,
        name=scaffold.name,
        type=scaffold.scaffold_type,
        parameters=json.loads(scaffold.parameters_json),
        thumbnail_path=scaffold.thumbnail_path,
        thumbnail_url=scaffold.thumbnail_path,
        stl_path=scaffold.stl_path,
        created_at=scaffold.created_at,
        updated_at=scaffold.updated_at,
        user_id=scaffold.user_id
    )

@router.put("/{scaffold_uuid}", response_model=ScaffoldResponse)
async def update_scaffold(
    scaffold_uuid: str,
    request: SaveScaffoldRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a scaffold."""
    scaffold = db.query(SavedScaffold).filter(
        SavedScaffold.uuid == scaffold_uuid,
        SavedScaffold.user_id == current_user.id
    ).first()

    if not scaffold:
        raise HTTPException(status_code=404, detail="Scaffold not found")

    scaffold.name = request.name
    scaffold.scaffold_type = request.scaffold_type
    scaffold.parameters_json = json.dumps(request.parameters)
    db.commit()
    db.refresh(scaffold)

    return ScaffoldResponse(
        id=scaffold.id,
        uuid=scaffold.uuid,
        name=scaffold.name,
        type=scaffold.scaffold_type,
        parameters=json.loads(scaffold.parameters_json),
        thumbnail_path=scaffold.thumbnail_path,
        thumbnail_url=scaffold.thumbnail_path,
        stl_path=scaffold.stl_path,
        created_at=scaffold.created_at,
        updated_at=scaffold.updated_at,
        user_id=scaffold.user_id
    )

@router.post("/{scaffold_uuid}/duplicate", response_model=ScaffoldResponse)
async def duplicate_scaffold(
    scaffold_uuid: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Duplicate a scaffold."""
    # Find original scaffold
    original = db.query(SavedScaffold).filter(
        SavedScaffold.uuid == scaffold_uuid,
        SavedScaffold.user_id == current_user.id
    ).first()

    if not original:
        raise HTTPException(status_code=404, detail="Scaffold not found")

    # Create duplicate
    duplicate = SavedScaffold(
        user_id=current_user.id,
        name=f"{original.name} (copy)",
        scaffold_type=original.scaffold_type,
        parameters_json=original.parameters_json
    )
    db.add(duplicate)
    db.commit()
    db.refresh(duplicate)

    return ScaffoldResponse(
        id=duplicate.id,
        uuid=duplicate.uuid,
        name=duplicate.name,
        scaffold_type=duplicate.scaffold_type,
        parameters=json.loads(duplicate.parameters_json),
        thumbnail_path=duplicate.thumbnail_path,
        stl_path=duplicate.stl_path,
        created_at=duplicate.created_at,
        updated_at=duplicate.updated_at
    )

@router.delete("/{scaffold_uuid}")
async def delete_scaffold(
    scaffold_uuid: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a scaffold."""
    scaffold = db.query(SavedScaffold).filter(
        SavedScaffold.uuid == scaffold_uuid,
        SavedScaffold.user_id == current_user.id
    ).first()

    if not scaffold:
        raise HTTPException(status_code=404, detail="Scaffold not found")

    # Delete associated files
    user_dir = get_user_storage_dir(current_user.uuid)
    if scaffold.stl_path:
        try:
            stl_file = safe_join(user_dir, scaffold.stl_path)
            if os.path.exists(stl_file):
                os.remove(stl_file)
        except ValueError:
            # Path traversal attempt detected, skip file deletion
            pass
    if scaffold.thumbnail_path:
        try:
            thumb_file = safe_join(user_dir, scaffold.thumbnail_path)
            if os.path.exists(thumb_file):
                os.remove(thumb_file)
        except ValueError:
            # Path traversal attempt detected, skip file deletion
            pass

    db.delete(scaffold)
    db.commit()

    return {"message": "Scaffold deleted"}
