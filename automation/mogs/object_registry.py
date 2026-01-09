"""
MOGS Object Registry

Manages the registry of all MOGS objects and provides lookup functionality.
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from .models import ObjectManifest, generate_object_uuid, get_timestamp
from .folder_manager import FolderManager, create_new_object, load_object


@dataclass
class ObjectRegistryEntry:
    """Entry in the object registry."""
    object_uuid: str
    object_name: str
    created_at: str
    last_modified_at: str
    active_spec_version: int
    path: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_uuid": self.object_uuid,
            "object_name": self.object_name,
            "created_at": self.created_at,
            "last_modified_at": self.last_modified_at,
            "active_spec_version": self.active_spec_version,
            "path": self.path,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ObjectRegistryEntry":
        return cls(**d)


class ObjectRegistry:
    """
    Registry for all MOGS objects.
    
    Provides functionality to:
    - Create new objects
    - List existing objects
    - Look up objects by UUID or name
    - Track object metadata
    """
    
    def __init__(self, objects_base_dir: str):
        """
        Initialize the object registry.
        
        Parameters
        ----------
        objects_base_dir : str
            Base directory for all objects
        """
        self.objects_base_dir = os.path.abspath(objects_base_dir)
        self.registry_path = os.path.join(self.objects_base_dir, "registry.json")
        
        # Ensure base directory exists
        os.makedirs(self.objects_base_dir, exist_ok=True)
        
        # Load or create registry
        self._registry: Dict[str, ObjectRegistryEntry] = {}
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load the registry from disk."""
        if os.path.exists(self.registry_path):
            with open(self.registry_path, 'r') as f:
                data = json.load(f)
                self._registry = {
                    uuid: ObjectRegistryEntry.from_dict(entry)
                    for uuid, entry in data.get("objects", {}).items()
                }
        else:
            self._registry = {}
            self._save_registry()
    
    def _save_registry(self) -> None:
        """Save the registry to disk."""
        data = {
            "objects": {
                uuid: entry.to_dict()
                for uuid, entry in self._registry.items()
            },
            "last_updated": get_timestamp(),
        }
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_object(self, object_name: str) -> FolderManager:
        """
        Create a new object with the given name.
        
        Parameters
        ----------
        object_name : str
            Human-readable name for the object
            
        Returns
        -------
        FolderManager
            Folder manager for the new object
        """
        # Generate UUID
        object_uuid = generate_object_uuid()
        
        # Create folder structure
        manager = create_new_object(self.objects_base_dir, object_name, object_uuid)
        
        # Add to registry
        entry = ObjectRegistryEntry(
            object_uuid=object_uuid,
            object_name=object_name,
            created_at=get_timestamp(),
            last_modified_at=get_timestamp(),
            active_spec_version=0,
            path=manager.object_dir,
        )
        self._registry[object_uuid] = entry
        self._save_registry()
        
        return manager
    
    def get_object(self, object_uuid: str) -> Optional[FolderManager]:
        """
        Get an object by UUID.
        
        Parameters
        ----------
        object_uuid : str
            UUID of the object
            
        Returns
        -------
        FolderManager or None
            Folder manager for the object, or None if not found
        """
        if object_uuid not in self._registry:
            return None
        
        try:
            return load_object(self.objects_base_dir, object_uuid)
        except FileNotFoundError:
            return None
    
    def get_object_by_name(self, object_name: str) -> Optional[FolderManager]:
        """
        Get an object by name (returns first match).
        
        Parameters
        ----------
        object_name : str
            Name of the object
            
        Returns
        -------
        FolderManager or None
            Folder manager for the object, or None if not found
        """
        for entry in self._registry.values():
            if entry.object_name == object_name:
                return self.get_object(entry.object_uuid)
        return None
    
    def list_objects(self) -> List[ObjectRegistryEntry]:
        """
        List all objects in the registry.
        
        Returns
        -------
        List[ObjectRegistryEntry]
            List of registry entries
        """
        return list(self._registry.values())
    
    def search_objects(self, query: str) -> List[ObjectRegistryEntry]:
        """
        Search objects by name (case-insensitive partial match).
        
        Parameters
        ----------
        query : str
            Search query
            
        Returns
        -------
        List[ObjectRegistryEntry]
            Matching registry entries
        """
        query_lower = query.lower()
        return [
            entry for entry in self._registry.values()
            if query_lower in entry.object_name.lower()
        ]
    
    def update_object_metadata(self, object_uuid: str) -> None:
        """
        Update registry entry from object manifest.
        
        Parameters
        ----------
        object_uuid : str
            UUID of the object to update
        """
        if object_uuid not in self._registry:
            return
        
        manager = self.get_object(object_uuid)
        if manager is None:
            return
        
        manifest = manager.load_manifest()
        entry = self._registry[object_uuid]
        entry.object_name = manifest.object_name
        entry.last_modified_at = manifest.last_modified_at
        entry.active_spec_version = manifest.active_spec_version
        
        self._save_registry()
    
    def delete_object(self, object_uuid: str, confirm: bool = False) -> bool:
        """
        Delete an object from the registry.
        
        Note: This only removes from registry, does not delete files.
        
        Parameters
        ----------
        object_uuid : str
            UUID of the object to delete
        confirm : bool
            Must be True to actually delete
            
        Returns
        -------
        bool
            True if deleted, False otherwise
        """
        if not confirm:
            return False
        
        if object_uuid in self._registry:
            del self._registry[object_uuid]
            self._save_registry()
            return True
        
        return False
    
    def refresh_registry(self) -> int:
        """
        Refresh registry by scanning the objects directory.
        
        Adds any objects that exist on disk but not in registry.
        
        Returns
        -------
        int
            Number of objects added
        """
        added = 0
        
        if not os.path.exists(self.objects_base_dir):
            return 0
        
        for item in os.listdir(self.objects_base_dir):
            item_path = os.path.join(self.objects_base_dir, item)
            
            # Skip non-directories and registry file
            if not os.path.isdir(item_path) or item == "registry.json":
                continue
            
            # Check if it's a valid object (has manifest)
            manifest_path = os.path.join(item_path, "project", "00_admin", "object_manifest.json")
            if not os.path.exists(manifest_path):
                continue
            
            # Skip if already in registry
            if item in self._registry:
                continue
            
            # Load manifest and add to registry
            try:
                manifest = ObjectManifest.load(manifest_path)
                entry = ObjectRegistryEntry(
                    object_uuid=manifest.object_uuid,
                    object_name=manifest.object_name,
                    created_at=manifest.created_at,
                    last_modified_at=manifest.last_modified_at,
                    active_spec_version=manifest.active_spec_version,
                    path=item_path,
                )
                self._registry[manifest.object_uuid] = entry
                added += 1
            except Exception:
                pass
        
        if added > 0:
            self._save_registry()
        
        return added
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the registry.
        
        Returns
        -------
        Dict[str, Any]
            Registry statistics
        """
        return {
            "total_objects": len(self._registry),
            "objects_base_dir": self.objects_base_dir,
            "registry_path": self.registry_path,
        }
