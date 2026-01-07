"""
Artifact Verifier

Verifies that required artifacts exist after script execution,
validates file contents, and creates manifests for tracking.
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .script_artifacts import (
    ArtifactProfile,
    ArtifactManifest,
    ArtifactsJson,
    get_artifact_profile,
)


@dataclass
class FileCheckResult:
    """
    Result of checking a single file.
    
    Attributes
    ----------
    path : str
        Path to the file
    exists : bool
        Whether the file exists
    size_bytes : int
        File size in bytes (0 if not exists)
    is_valid : bool
        Whether the file passed validation
    validation_details : Dict[str, Any]
        Details from validation (e.g., mesh stats)
    error : str or None
        Error message if validation failed
    """
    path: str
    exists: bool
    size_bytes: int = 0
    is_valid: bool = False
    validation_details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class VerificationResult:
    """
    Result of verifying all artifacts for a stage.
    
    Attributes
    ----------
    success : bool
        Whether all required artifacts passed verification
    required_passed : int
        Number of required files that passed
    required_total : int
        Total number of required files
    optional_passed : int
        Number of optional files that passed
    optional_total : int
        Total number of optional files
    file_results : List[FileCheckResult]
        Individual results for each file
    artifacts_json : ArtifactsJson or None
        Parsed ARTIFACTS_JSON from script output
    manifest : ArtifactManifest or None
        Generated manifest for this verification
    errors : List[str]
        List of error messages
    warnings : List[str]
        List of warning messages
    """
    success: bool
    required_passed: int
    required_total: int
    optional_passed: int
    optional_total: int
    file_results: List[FileCheckResult] = field(default_factory=list)
    artifacts_json: Optional[ArtifactsJson] = None
    manifest: Optional[ArtifactManifest] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def check_file_exists(file_path: str) -> FileCheckResult:
    """
    Check if a file exists and get its size.
    
    Parameters
    ----------
    file_path : str
        Path to the file to check
        
    Returns
    -------
    FileCheckResult
        Result with existence and size information
    """
    if not os.path.exists(file_path):
        return FileCheckResult(
            path=file_path,
            exists=False,
            is_valid=False,
            error="File does not exist",
        )
    
    try:
        size = os.path.getsize(file_path)
        if size == 0:
            return FileCheckResult(
                path=file_path,
                exists=True,
                size_bytes=0,
                is_valid=False,
                error="File is empty (0 bytes)",
            )
        
        return FileCheckResult(
            path=file_path,
            exists=True,
            size_bytes=size,
            is_valid=True,
        )
    except Exception as e:
        return FileCheckResult(
            path=file_path,
            exists=True,
            is_valid=False,
            error=f"Failed to check file: {e}",
        )


def validate_json_file(file_path: str) -> FileCheckResult:
    """
    Validate a JSON file by attempting to parse it.
    
    Parameters
    ----------
    file_path : str
        Path to the JSON file
        
    Returns
    -------
    FileCheckResult
        Result with validation details
    """
    result = check_file_exists(file_path)
    if not result.exists or not result.is_valid:
        return result
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        result.validation_details = {
            "type": type(data).__name__,
            "keys": list(data.keys()) if isinstance(data, dict) else None,
            "length": len(data) if isinstance(data, (list, dict)) else None,
        }
        result.is_valid = True
        
    except json.JSONDecodeError as e:
        result.is_valid = False
        result.error = f"Invalid JSON: {e}"
    except Exception as e:
        result.is_valid = False
        result.error = f"Failed to read JSON: {e}"
    
    return result


def validate_stl_file(file_path: str, check_watertight: bool = False) -> FileCheckResult:
    """
    Validate an STL file by attempting to load it with trimesh.
    
    Parameters
    ----------
    file_path : str
        Path to the STL file
    check_watertight : bool
        Whether to check if the mesh is watertight
        
    Returns
    -------
    FileCheckResult
        Result with mesh validation details
    """
    result = check_file_exists(file_path)
    if not result.exists or not result.is_valid:
        return result
    
    try:
        import trimesh
        mesh = trimesh.load(file_path)
        
        result.validation_details = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "is_watertight": mesh.is_watertight,
            "is_volume": mesh.is_volume,
            "bounds": mesh.bounds.tolist() if hasattr(mesh.bounds, 'tolist') else None,
        }
        
        if check_watertight and not mesh.is_watertight:
            result.is_valid = False
            result.error = "Mesh is not watertight"
        else:
            result.is_valid = True
            
    except ImportError:
        # trimesh not available, just check file exists
        result.validation_details = {"note": "trimesh not available for validation"}
        result.is_valid = True
    except Exception as e:
        result.is_valid = False
        result.error = f"Failed to load STL: {e}"
    
    return result


def validate_file(file_path: str, check_watertight: bool = False) -> FileCheckResult:
    """
    Validate a file based on its extension.
    
    Parameters
    ----------
    file_path : str
        Path to the file
    check_watertight : bool
        For STL files, whether to check watertightness
        
    Returns
    -------
    FileCheckResult
        Validation result
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.json':
        return validate_json_file(file_path)
    elif ext == '.stl':
        return validate_stl_file(file_path, check_watertight)
    else:
        return check_file_exists(file_path)


def verify_artifacts(
    object_dir: str,
    profile: ArtifactProfile,
    script_output: Optional[str] = None,
    version: int = 1,
    spec_path: Optional[str] = None,
    seed: Optional[int] = None,
    check_watertight: bool = False,
) -> VerificationResult:
    """
    Verify artifacts against a profile.
    
    Parameters
    ----------
    object_dir : str
        Base directory for the object
    profile : ArtifactProfile
        Artifact profile defining expected files
    script_output : str, optional
        Script stdout to parse for ARTIFACTS_JSON
    version : int
        Version number for manifest
    spec_path : str, optional
        Path to the spec file used for generation
    seed : int, optional
        Random seed used
    check_watertight : bool
        Whether to check STL watertightness
        
    Returns
    -------
    VerificationResult
        Complete verification result
    """
    errors = []
    warnings = []
    file_results = []
    
    # Parse ARTIFACTS_JSON from script output if available
    artifacts_json = None
    if script_output:
        artifacts_json = ArtifactsJson.parse_from_output(script_output)
        if artifacts_json and artifacts_json.status == "failed":
            errors.append("Script reported failure in ARTIFACTS_JSON")
    
    # Check required files
    required_passed = 0
    for rel_path in profile.required_files:
        full_path = os.path.join(object_dir, rel_path)
        result = validate_file(full_path, check_watertight)
        file_results.append(result)
        
        if result.is_valid:
            required_passed += 1
        else:
            errors.append(f"Required file failed: {rel_path} - {result.error}")
    
    # Check optional files
    optional_passed = 0
    for rel_path in profile.optional_files:
        full_path = os.path.join(object_dir, rel_path)
        result = validate_file(full_path, check_watertight)
        file_results.append(result)
        
        if result.is_valid:
            optional_passed += 1
        elif result.exists:
            warnings.append(f"Optional file invalid: {rel_path} - {result.error}")
    
    # Check files reported by ARTIFACTS_JSON
    if artifacts_json:
        for reported_file in artifacts_json.files:
            # Check if it's an absolute path or relative
            if os.path.isabs(reported_file):
                full_path = reported_file
            else:
                full_path = os.path.join(object_dir, reported_file)
            
            # Only check if not already in our list
            already_checked = any(r.path == full_path for r in file_results)
            if not already_checked:
                result = validate_file(full_path, check_watertight)
                file_results.append(result)
                
                if not result.is_valid:
                    warnings.append(
                        f"File reported by script not valid: {reported_file} - {result.error}"
                    )
    
    # Determine overall success
    success = required_passed == len(profile.required_files)
    
    # Create manifest
    manifest = ArtifactManifest(
        version=version,
        created_files=[r.path for r in file_results if r.exists],
        spec_path=spec_path or "",
        seed=seed,
        key_metrics=artifacts_json.metrics if artifacts_json else {},
        status="completed" if success else "failed",
        errors=errors,
    )
    
    return VerificationResult(
        success=success,
        required_passed=required_passed,
        required_total=len(profile.required_files),
        optional_passed=optional_passed,
        optional_total=len(profile.optional_files),
        file_results=file_results,
        artifacts_json=artifacts_json,
        manifest=manifest,
        errors=errors,
        warnings=warnings,
    )


def verify_generation_stage(
    object_dir: str,
    version: int,
    script_output: Optional[str] = None,
    spec_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> VerificationResult:
    """
    Verify artifacts for the generation stage.
    
    Parameters
    ----------
    object_dir : str
        Base directory for the object
    version : int
        Version number
    script_output : str, optional
        Script stdout to parse
    spec_path : str, optional
        Path to spec file
    seed : int, optional
        Random seed used
        
    Returns
    -------
    VerificationResult
        Verification result for generation stage
    """
    profile = get_artifact_profile("generation", version)
    return verify_artifacts(
        object_dir=object_dir,
        profile=profile,
        script_output=script_output,
        version=version,
        spec_path=spec_path,
        seed=seed,
    )


def verify_final_stage(
    object_dir: str,
    script_output: Optional[str] = None,
    spec_path: Optional[str] = None,
    seed: Optional[int] = None,
    check_watertight: bool = True,
) -> VerificationResult:
    """
    Verify artifacts for the final stage.
    
    Parameters
    ----------
    object_dir : str
        Base directory for the object
    script_output : str, optional
        Script stdout to parse
    spec_path : str, optional
        Path to spec file
    seed : int, optional
        Random seed used
    check_watertight : bool
        Whether to verify STL watertightness
        
    Returns
    -------
    VerificationResult
        Verification result for final stage
    """
    profile = get_artifact_profile("final")
    return verify_artifacts(
        object_dir=object_dir,
        profile=profile,
        script_output=script_output,
        version=1,  # Final stage doesn't have versions
        spec_path=spec_path,
        seed=seed,
        check_watertight=check_watertight,
    )


def save_manifest(
    manifest: ArtifactManifest,
    object_dir: str,
    version: int,
) -> str:
    """
    Save a manifest to the object directory.
    
    Parameters
    ----------
    manifest : ArtifactManifest
        Manifest to save
    object_dir : str
        Object directory
    version : int
        Version number for filename
        
    Returns
    -------
    str
        Path to saved manifest
    """
    manifest_path = os.path.join(object_dir, f"manifest_v{version:03d}.json")
    manifest.save(manifest_path)
    return manifest_path


def print_verification_summary(result: VerificationResult, verbose: bool = True) -> None:
    """
    Print a summary of verification results.
    
    Parameters
    ----------
    result : VerificationResult
        Verification result to summarize
    verbose : bool
        Whether to print detailed information
    """
    print()
    print("-" * 40)
    print("Artifact Verification Summary")
    print("-" * 40)
    
    status = "PASSED" if result.success else "FAILED"
    print(f"Status: {status}")
    print(f"Required: {result.required_passed}/{result.required_total}")
    print(f"Optional: {result.optional_passed}/{result.optional_total}")
    
    if result.errors:
        print()
        print("Errors:")
        for err in result.errors:
            print(f"  - {err}")
    
    if result.warnings:
        print()
        print("Warnings:")
        for warn in result.warnings:
            print(f"  - {warn}")
    
    if verbose and result.file_results:
        print()
        print("Files:")
        for fr in result.file_results:
            status_char = "[OK]" if fr.is_valid else "[--]"
            size_str = f"({fr.size_bytes} bytes)" if fr.exists else "(missing)"
            print(f"  {status_char} {fr.path} {size_str}")
    
    if result.artifacts_json:
        print()
        print("Script-reported metrics:")
        for key, value in result.artifacts_json.metrics.items():
            print(f"  {key}: {value}")
    
    print("-" * 40)
