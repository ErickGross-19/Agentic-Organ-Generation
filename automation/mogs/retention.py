"""
MOGS Retention Policy

Implements version retention and safe garbage collection:
- Keep only the last 5 spec versions (configurable)
- Safe cleanup of old versions as a unit
- Audit trail of all deletions
- Support for pinning versions
"""

import os
import json
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from datetime import datetime

from .models import (
    RetentionPolicy,
    RunIndex,
    RunEntry,
    get_timestamp,
)
from .folder_manager import FolderManager


@dataclass
class RetentionAction:
    """
    Record of a retention action (deletion, pin, unpin).
    """
    action_type: str  # "delete", "pin", "unpin"
    version: int
    timestamp: str
    reason: str
    files_affected: List[str] = field(default_factory=list)
    success: bool = True
    error: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "version": self.version,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "files_affected": self.files_affected,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class RetentionReport:
    """
    Report from a retention cleanup operation.
    """
    timestamp: str
    versions_before: List[int]
    versions_after: List[int]
    versions_deleted: List[int]
    actions: List[RetentionAction]
    total_files_deleted: int = 0
    total_bytes_freed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "versions_before": self.versions_before,
            "versions_after": self.versions_after,
            "versions_deleted": self.versions_deleted,
            "actions": [a.to_dict() for a in self.actions],
            "total_files_deleted": self.total_files_deleted,
            "total_bytes_freed": self.total_bytes_freed,
        }


class RetentionManager:
    """
    Manages version retention for a MOGS object.
    
    Implements:
    - Keep last N versions (default 5)
    - Safe deletion of old versions as a unit
    - Audit trail in retention_log.md
    - Version pinning to preserve milestones
    """
    
    def __init__(self, folder_manager: FolderManager):
        """
        Initialize the retention manager.
        
        Parameters
        ----------
        folder_manager : FolderManager
            Folder manager for the object
        """
        self.folder_manager = folder_manager
    
    def get_retention_policy(self) -> RetentionPolicy:
        """Get the current retention policy from the manifest."""
        manifest = self.folder_manager.load_manifest()
        return manifest.retention_policy
    
    def set_retention_policy(self, policy: RetentionPolicy) -> None:
        """
        Set the retention policy.
        
        Parameters
        ----------
        policy : RetentionPolicy
            New retention policy
        """
        manifest = self.folder_manager.load_manifest()
        manifest.retention_policy = policy
        self.folder_manager.save_manifest(manifest)
        self.folder_manager.log_event(f"Retention policy updated: keep_last={policy.keep_last_versions}")
    
    def pin_version(self, version: int, reason: str = "") -> bool:
        """
        Pin a version to prevent deletion.
        
        Parameters
        ----------
        version : int
            Version to pin
        reason : str
            Reason for pinning
            
        Returns
        -------
        bool
            True if pinned successfully
        """
        if not self.folder_manager.version_exists(version):
            return False
        
        manifest = self.folder_manager.load_manifest()
        if version not in manifest.retention_policy.pinned_versions:
            manifest.retention_policy.pinned_versions.append(version)
            self.folder_manager.save_manifest(manifest)
        
        self.folder_manager.log_retention_action("PINNED", version, reason)
        self.folder_manager.log_event(f"Version {version} pinned: {reason}")
        
        return True
    
    def unpin_version(self, version: int, reason: str = "") -> bool:
        """
        Unpin a version to allow deletion.
        
        Parameters
        ----------
        version : int
            Version to unpin
        reason : str
            Reason for unpinning
            
        Returns
        -------
        bool
            True if unpinned successfully
        """
        manifest = self.folder_manager.load_manifest()
        if version in manifest.retention_policy.pinned_versions:
            manifest.retention_policy.pinned_versions.remove(version)
            self.folder_manager.save_manifest(manifest)
            
            self.folder_manager.log_retention_action("UNPINNED", version, reason)
            self.folder_manager.log_event(f"Version {version} unpinned: {reason}")
            return True
        
        return False
    
    def get_versions_to_delete(self) -> List[int]:
        """
        Get list of versions that should be deleted based on retention policy.
        
        Returns
        -------
        List[int]
            Versions to delete (oldest first)
        """
        policy = self.get_retention_policy()
        existing_versions = self.folder_manager.get_existing_versions()
        
        if len(existing_versions) <= policy.keep_last_versions:
            return []
        
        # Load manifest to get accepted versions
        manifest = self.folder_manager.load_manifest()
        accepted_versions = set(manifest.accepted_versions)
        most_recent_accepted = max(accepted_versions) if accepted_versions else None
        
        # Determine which versions to keep
        versions_to_keep: Set[int] = set()
        
        # Keep last N versions
        versions_to_keep.update(existing_versions[-policy.keep_last_versions:])
        
        # Keep pinned versions
        versions_to_keep.update(policy.pinned_versions)
        
        # Keep most recent accepted version
        if most_recent_accepted is not None:
            versions_to_keep.add(most_recent_accepted)
        
        # Return versions to delete (oldest first)
        versions_to_delete = [v for v in existing_versions if v not in versions_to_keep]
        return sorted(versions_to_delete)
    
    def delete_version(self, version: int, dry_run: bool = False) -> RetentionAction:
        """
        Delete a single version and all its associated files.
        
        Parameters
        ----------
        version : int
            Version to delete
        dry_run : bool
            If True, don't actually delete, just report what would be deleted
            
        Returns
        -------
        RetentionAction
            Record of the deletion action
        """
        policy = self.get_retention_policy()
        
        # Check if version is pinned
        if version in policy.pinned_versions:
            return RetentionAction(
                action_type="delete",
                version=version,
                timestamp=get_timestamp(),
                reason="Retention cleanup",
                success=False,
                error="Version is pinned and cannot be deleted",
            )
        
        # Get all paths for this version
        paths = self.folder_manager.get_all_version_paths(version)
        files_to_delete = []
        
        for path_name, path in paths.items():
            if os.path.exists(path):
                if os.path.isdir(path):
                    # Recursively list all files in directory
                    for root, dirs, files in os.walk(path):
                        for f in files:
                            files_to_delete.append(os.path.join(root, f))
                else:
                    files_to_delete.append(path)
        
        # Also check for agent docs
        for agent in ["CSA", "CBA", "VQA"]:
            agent_dir = self.folder_manager.get_agent_docs_dir(agent)
            if os.path.exists(agent_dir):
                for filename in os.listdir(agent_dir):
                    if f"v{version:03d}" in filename:
                        files_to_delete.append(os.path.join(agent_dir, filename))
        
        # Check for prompts
        prompts_dir = self.folder_manager.prompts_dir
        if os.path.exists(prompts_dir):
            for filename in os.listdir(prompts_dir):
                if f"v{version:03d}" in filename:
                    files_to_delete.append(os.path.join(prompts_dir, filename))
        
        if dry_run:
            return RetentionAction(
                action_type="delete_dry_run",
                version=version,
                timestamp=get_timestamp(),
                reason="Retention cleanup (dry run)",
                files_affected=files_to_delete,
                success=True,
            )
        
        # Actually delete files
        try:
            for path_name, path in paths.items():
                if os.path.exists(path):
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
            
            # Delete agent docs
            for agent in ["CSA", "CBA", "VQA"]:
                agent_dir = self.folder_manager.get_agent_docs_dir(agent)
                if os.path.exists(agent_dir):
                    for filename in os.listdir(agent_dir):
                        if f"v{version:03d}" in filename:
                            os.remove(os.path.join(agent_dir, filename))
            
            # Delete prompts
            if os.path.exists(prompts_dir):
                for filename in os.listdir(prompts_dir):
                    if f"v{version:03d}" in filename:
                        os.remove(os.path.join(prompts_dir, filename))
            
            action = RetentionAction(
                action_type="delete",
                version=version,
                timestamp=get_timestamp(),
                reason="Retention cleanup",
                files_affected=files_to_delete,
                success=True,
            )
            
            self.folder_manager.log_retention_action(
                "DELETED",
                version,
                f"Deleted {len(files_to_delete)} files",
            )
            self.folder_manager.log_event(f"Version {version} deleted: {len(files_to_delete)} files")
            
            return action
            
        except Exception as e:
            return RetentionAction(
                action_type="delete",
                version=version,
                timestamp=get_timestamp(),
                reason="Retention cleanup",
                files_affected=files_to_delete,
                success=False,
                error=str(e),
            )
    
    def run_cleanup(self, dry_run: bool = False) -> RetentionReport:
        """
        Run retention cleanup.
        
        Deletes versions that exceed the retention policy.
        
        Parameters
        ----------
        dry_run : bool
            If True, don't actually delete, just report what would be deleted
            
        Returns
        -------
        RetentionReport
            Report of the cleanup operation
        """
        versions_before = self.folder_manager.get_existing_versions()
        versions_to_delete = self.get_versions_to_delete()
        
        actions = []
        total_files = 0
        
        for version in versions_to_delete:
            action = self.delete_version(version, dry_run=dry_run)
            actions.append(action)
            if action.success:
                total_files += len(action.files_affected)
        
        versions_after = self.folder_manager.get_existing_versions() if not dry_run else [
            v for v in versions_before if v not in versions_to_delete
        ]
        
        report = RetentionReport(
            timestamp=get_timestamp(),
            versions_before=versions_before,
            versions_after=versions_after,
            versions_deleted=versions_to_delete if not dry_run else [],
            actions=actions,
            total_files_deleted=total_files,
        )
        
        if not dry_run and versions_to_delete:
            self._save_cleanup_report(report)
        
        return report
    
    def _save_cleanup_report(self, report: RetentionReport) -> None:
        """Save a cleanup report to disk."""
        reports_dir = os.path.join(self.folder_manager.admin_dir, "retention_reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        filename = f"cleanup_{report.timestamp.replace(':', '-')}.json"
        filepath = os.path.join(reports_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
    
    def get_retention_status(self) -> Dict[str, Any]:
        """
        Get current retention status.
        
        Returns
        -------
        Dict[str, Any]
            Retention status information
        """
        policy = self.get_retention_policy()
        existing_versions = self.folder_manager.get_existing_versions()
        versions_to_delete = self.get_versions_to_delete()
        
        manifest = self.folder_manager.load_manifest()
        
        return {
            "policy": policy.to_dict(),
            "existing_versions": existing_versions,
            "total_versions": len(existing_versions),
            "versions_to_delete": versions_to_delete,
            "pinned_versions": policy.pinned_versions,
            "accepted_versions": manifest.accepted_versions,
            "cleanup_needed": len(versions_to_delete) > 0,
        }
    
    def mark_version_accepted(self, version: int) -> bool:
        """
        Mark a version as accepted.
        
        Accepted versions are protected from deletion (most recent accepted
        is always kept).
        
        Parameters
        ----------
        version : int
            Version to mark as accepted
            
        Returns
        -------
        bool
            True if marked successfully
        """
        if not self.folder_manager.version_exists(version):
            return False
        
        manifest = self.folder_manager.load_manifest()
        if version not in manifest.accepted_versions:
            manifest.accepted_versions.append(version)
            self.folder_manager.save_manifest(manifest)
        
        self.folder_manager.log_event(f"Version {version} marked as accepted")
        
        return True
    
    def unmark_version_accepted(self, version: int) -> bool:
        """
        Unmark a version as accepted.
        
        Parameters
        ----------
        version : int
            Version to unmark
            
        Returns
        -------
        bool
            True if unmarked successfully
        """
        manifest = self.folder_manager.load_manifest()
        if version in manifest.accepted_versions:
            manifest.accepted_versions.remove(version)
            self.folder_manager.save_manifest(manifest)
            self.folder_manager.log_event(f"Version {version} unmarked as accepted")
            return True
        
        return False
