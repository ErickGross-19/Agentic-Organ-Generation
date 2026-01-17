"""
JSON serialization utilities for DesignSpec reports.

This module provides utilities for JSON-safe serialization and
content hashing for reproducibility.

SAFETY
------
All serialization functions ensure that numpy scalars and other
non-JSON-serializable types are converted to native Python types.
"""

from typing import Any, Dict, List, Union
import json
import hashlib
import logging

logger = logging.getLogger(__name__)


def make_json_safe(obj: Any) -> Any:
    """
    Convert an object to a JSON-safe representation.
    
    Handles:
    - numpy scalars → Python scalars
    - numpy arrays → Python lists
    - bytes → base64 strings
    - dataclasses → dicts
    - objects with to_dict() → dicts
    
    Parameters
    ----------
    obj : Any
        Object to convert
        
    Returns
    -------
    Any
        JSON-safe representation
    """
    if obj is None:
        return None
    
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode("ascii")
    
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj]
    
    try:
        import numpy as np
        
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        
        if isinstance(obj, np.bool_):
            return bool(obj)
            
    except ImportError:
        pass
    
    if hasattr(obj, "to_dict"):
        return make_json_safe(obj.to_dict())
    
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return make_json_safe(asdict(obj))
    
    return str(obj)


def to_json(obj: Any, indent: int = 2, sort_keys: bool = True) -> str:
    """
    Serialize an object to JSON string.
    
    Parameters
    ----------
    obj : Any
        Object to serialize
    indent : int
        Indentation level (default: 2)
    sort_keys : bool
        Whether to sort keys (default: True for reproducibility)
        
    Returns
    -------
    str
        JSON string
    """
    safe_obj = make_json_safe(obj)
    return json.dumps(safe_obj, indent=indent, sort_keys=sort_keys)


def from_json(json_str: str) -> Any:
    """
    Deserialize a JSON string.
    
    Parameters
    ----------
    json_str : str
        JSON string to deserialize
        
    Returns
    -------
    Any
        Deserialized object
    """
    return json.loads(json_str)


def compute_content_hash(obj: Any, algorithm: str = "sha256") -> str:
    """
    Compute a stable content hash for an object.
    
    The hash is computed from the canonical JSON representation
    (sorted keys, no whitespace) to ensure stability.
    
    Parameters
    ----------
    obj : Any
        Object to hash
    algorithm : str
        Hash algorithm (default: sha256)
        
    Returns
    -------
    str
        Hex digest of the hash
    """
    safe_obj = make_json_safe(obj)
    canonical_json = json.dumps(safe_obj, sort_keys=True, separators=(",", ":"))
    
    hasher = hashlib.new(algorithm)
    hasher.update(canonical_json.encode("utf-8"))
    
    return hasher.hexdigest()


def compute_short_hash(obj: Any, length: int = 16) -> str:
    """
    Compute a short content hash for an object.
    
    Parameters
    ----------
    obj : Any
        Object to hash
    length : int
        Length of hash to return (default: 16)
        
    Returns
    -------
    str
        Truncated hex digest
    """
    full_hash = compute_content_hash(obj)
    return full_hash[:length]


def canonical_json(obj: Any) -> str:
    """
    Generate canonical JSON representation for hashing.
    
    Canonical JSON has:
    - Sorted keys
    - No whitespace
    - Consistent formatting
    
    Parameters
    ----------
    obj : Any
        Object to serialize
        
    Returns
    -------
    str
        Canonical JSON string
    """
    safe_obj = make_json_safe(obj)
    return json.dumps(safe_obj, sort_keys=True, separators=(",", ":"))


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """
    Save an object to a JSON file.
    
    Parameters
    ----------
    obj : Any
        Object to save
    path : str
        Path to save to
    indent : int
        Indentation level
    """
    from pathlib import Path
    
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    with open(p, "w") as f:
        f.write(to_json(obj, indent=indent))


def load_json(path: str) -> Any:
    """
    Load an object from a JSON file.
    
    Parameters
    ----------
    path : str
        Path to load from
        
    Returns
    -------
    Any
        Loaded object
    """
    with open(path, "r") as f:
        return from_json(f.read())


__all__ = [
    "make_json_safe",
    "to_json",
    "from_json",
    "compute_content_hash",
    "compute_short_hash",
    "canonical_json",
    "save_json",
    "load_json",
]
