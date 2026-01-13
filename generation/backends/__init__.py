"""
Backend interfaces for vascular network generation.

This module provides a unified interface for different generation methods:
- CCO hybrid backend (Sexton-style accelerated)
- Space colonization backend (wrapper for existing ops)

Backend Registration Pattern:
- Backends are registered with their capabilities (dual_tree, closed_loop support)
- Optional backends gracefully degrade if dependencies are missing
- Use get_available_backends() to discover available backends at runtime
"""

from typing import Dict, List, Optional, Type

from .base import GenerationBackend, BackendConfig
from .cco_hybrid_backend import CCOHybridBackend, CCOConfig

_BACKEND_REGISTRY: Dict[str, Type[GenerationBackend]] = {
    "cco_hybrid": CCOHybridBackend,
}

_CONFIG_REGISTRY: Dict[str, Type[BackendConfig]] = {
    "cco_hybrid": CCOConfig,
}

_BACKEND_LOAD_ERRORS: Dict[str, str] = {}

try:
    from .space_colonization_backend import SpaceColonizationBackend, SpaceColonizationConfig
    _BACKEND_REGISTRY["space_colonization"] = SpaceColonizationBackend
    _CONFIG_REGISTRY["space_colonization"] = SpaceColonizationConfig
except ImportError as e:
    _BACKEND_LOAD_ERRORS["space_colonization"] = f"Import failed: {e}"
    SpaceColonizationBackend = None
    SpaceColonizationConfig = None
except Exception as e:
    _BACKEND_LOAD_ERRORS["space_colonization"] = f"Unexpected error: {e}"
    SpaceColonizationBackend = None
    SpaceColonizationConfig = None


def get_available_backends() -> List[str]:
    """
    Get list of available backend names.
    
    Returns
    -------
    List[str]
        Names of backends that are available for use
    """
    return list(_BACKEND_REGISTRY.keys())


def get_backend(name: str) -> Optional[Type[GenerationBackend]]:
    """
    Get a backend class by name.
    
    Parameters
    ----------
    name : str
        Backend name (e.g., "cco_hybrid", "space_colonization")
        
    Returns
    -------
    Type[GenerationBackend] or None
        Backend class if available, None otherwise
    """
    return _BACKEND_REGISTRY.get(name)


def get_backend_config(name: str) -> Optional[Type[BackendConfig]]:
    """
    Get a backend config class by name.
    
    Parameters
    ----------
    name : str
        Backend name (e.g., "cco_hybrid", "space_colonization")
        
    Returns
    -------
    Type[BackendConfig] or None
        Config class if available, None otherwise
    """
    return _CONFIG_REGISTRY.get(name)


def get_backend_capabilities(name: str) -> Dict[str, bool]:
    """
    Get capabilities of a backend.
    
    Parameters
    ----------
    name : str
        Backend name
        
    Returns
    -------
    Dict[str, bool]
        Dictionary of capability names to boolean values
    """
    backend_class = _BACKEND_REGISTRY.get(name)
    if backend_class is None:
        return {}
    
    backend = backend_class()
    return {
        "dual_tree": backend.supports_dual_tree,
        "closed_loop": backend.supports_closed_loop,
    }


def get_backend_load_error(name: str) -> Optional[str]:
    """
    Get the error message if a backend failed to load.
    
    Parameters
    ----------
    name : str
        Backend name
        
    Returns
    -------
    str or None
        Error message if backend failed to load, None if loaded successfully
    """
    return _BACKEND_LOAD_ERRORS.get(name)


__all__ = [
    "GenerationBackend",
    "BackendConfig",
    "CCOHybridBackend",
    "CCOConfig",
    "SpaceColonizationBackend",
    "SpaceColonizationConfig",
    "get_available_backends",
    "get_backend",
    "get_backend_config",
    "get_backend_capabilities",
    "get_backend_load_error",
]
