"""
Scaffold models and type definitions.
"""

from .scaffold import (
    # Enums
    ScaffoldType,
    PorePattern,
    InnerTexture,
    UnitCell,
    PrimitiveShape,
    ModificationOperation,
    # Base Parameters
    BaseParams,
    # Type-Specific Parameters
    VascularNetworkParams,
    PorousDiscParams,
    TubularConduitParams,
    LatticeParams,
    PrimitiveParams,
    # Discriminated Union
    ScaffoldParams,
    # Supporting Types
    BoundingBoxDimensions,
    PrimitiveModification,
    # Generation Results
    MeshData,
    BoundingBox,
    GenerationResult,
    # Validation Results
    ValidationCheck,
    MinWallThicknessCheck,
    OverhangCheck,
    DimensionsCheck,
    ValidationChecks,
    ValidationResult,
    # Chat Interface
    ChatMessage,
    # Default Values
    DEFAULT_RESOLUTION,
    DEFAULT_VASCULAR_NETWORK,
    DEFAULT_POROUS_DISC,
    DEFAULT_TUBULAR_CONDUIT,
    DEFAULT_LATTICE,
    DEFAULT_PRIMITIVE,
)

__all__ = [
    # Enums
    "ScaffoldType",
    "PorePattern",
    "InnerTexture",
    "UnitCell",
    "PrimitiveShape",
    "ModificationOperation",
    # Base Parameters
    "BaseParams",
    # Type-Specific Parameters
    "VascularNetworkParams",
    "PorousDiscParams",
    "TubularConduitParams",
    "LatticeParams",
    "PrimitiveParams",
    # Discriminated Union
    "ScaffoldParams",
    # Supporting Types
    "BoundingBoxDimensions",
    "PrimitiveModification",
    # Generation Results
    "MeshData",
    "BoundingBox",
    "GenerationResult",
    # Validation Results
    "ValidationCheck",
    "MinWallThicknessCheck",
    "OverhangCheck",
    "DimensionsCheck",
    "ValidationChecks",
    "ValidationResult",
    # Chat Interface
    "ChatMessage",
    # Default Values
    "DEFAULT_RESOLUTION",
    "DEFAULT_VASCULAR_NETWORK",
    "DEFAULT_POROUS_DISC",
    "DEFAULT_TUBULAR_CONDUIT",
    "DEFAULT_LATTICE",
    "DEFAULT_PRIMITIVE",
]
