"""Parameter presets for common vascular network designs.

This module provides named parameter presets for SpaceColonizationParams
to make it easier for LLMs to generate appropriate networks for different
anatomical contexts and use cases.

Units: All spatial parameters are in millimeters.
"""

from ..ops.space_colonization import SpaceColonizationParams


def liver_arterial_dense() -> SpaceColonizationParams:
    """
    Dense arterial tree for liver lobe.
    
    Characteristics:
    - Small influence radius for dense branching
    - Moderate curvature constraint for realistic vessels
    - Bifurcation encouraged for tree-like structure
    
    Units: All spatial parameters in millimeters.
    """
    return SpaceColonizationParams(
        influence_radius=10.0,  # 10mm - dense branching
        kill_radius=2.0,  # 2mm - fine perfusion
        step_size=3.0,  # 3mm - small steps
        min_radius=0.2,  # 0.2mm - capillary-like
        taper_factor=0.92,  # Moderate tapering
        vessel_type="arterial",
        max_steps=200,
        max_curvature_deg=60.0,  # Realistic vessel curvature
        min_clearance=1.0,  # 1mm minimum clearance
        encourage_bifurcation=True,
        min_attractions_for_bifurcation=2,
        max_children_per_node=2,
        bifurcation_angle_threshold_deg=35.0,
        bifurcation_probability=0.8,
    )


def liver_venous_sparse() -> SpaceColonizationParams:
    """
    Sparse venous tree for liver lobe.
    
    Characteristics:
    - Larger influence radius for sparser branching
    - Less aggressive bifurcation
    - Larger minimum radius
    
    Units: All spatial parameters in millimeters.
    """
    return SpaceColonizationParams(
        influence_radius=20.0,  # 20mm - sparse branching
        kill_radius=4.0,  # 4mm - coarser perfusion
        step_size=6.0,  # 6mm - larger steps
        min_radius=0.4,  # 0.4mm
        taper_factor=0.94,  # Gentler tapering
        vessel_type="venous",
        max_steps=150,
        max_curvature_deg=70.0,  # More flexible
        min_clearance=1.5,  # 1.5mm minimum clearance
        encourage_bifurcation=True,
        min_attractions_for_bifurcation=3,
        max_children_per_node=2,
        bifurcation_angle_threshold_deg=40.0,
        bifurcation_probability=0.6,
    )


def kidney_arterial() -> SpaceColonizationParams:
    """
    Arterial tree for kidney.
    
    Characteristics:
    - Very dense branching for high perfusion
    - Tight curvature constraints
    - Strong bifurcation encouragement
    
    Units: All spatial parameters in millimeters.
    """
    return SpaceColonizationParams(
        influence_radius=8.0,  # 8mm - very dense
        kill_radius=1.5,  # 1.5mm - fine perfusion
        step_size=2.5,  # 2.5mm - small steps
        min_radius=0.15,  # 0.15mm
        taper_factor=0.90,  # Aggressive tapering
        vessel_type="arterial",
        max_steps=250,
        max_curvature_deg=50.0,  # Tight curvature
        min_clearance=0.8,  # 0.8mm minimum clearance
        encourage_bifurcation=True,
        min_attractions_for_bifurcation=2,
        max_children_per_node=2,
        bifurcation_angle_threshold_deg=30.0,
        bifurcation_probability=0.85,
    )


def sparse_debug() -> SpaceColonizationParams:
    """
    Sparse network for quick visual debugging.
    
    Characteristics:
    - Very large influence radius
    - Large step size
    - Few steps
    - No quality constraints
    
    Units: All spatial parameters in millimeters.
    """
    return SpaceColonizationParams(
        influence_radius=40.0,  # 40mm - very sparse
        kill_radius=10.0,  # 10mm - coarse perfusion
        step_size=15.0,  # 15mm - large steps
        min_radius=1.0,  # 1mm
        taper_factor=0.98,  # Minimal tapering
        vessel_type="arterial",
        max_steps=30,  # Quick generation
        max_curvature_deg=None,  # No curvature constraint
        min_clearance=None,  # No clearance check
        encourage_bifurcation=False,
    )


def lung_arterial() -> SpaceColonizationParams:
    """
    Pulmonary arterial tree.
    
    Characteristics:
    - Moderate density
    - Directional bias toward alveoli
    - Moderate curvature
    
    Units: All spatial parameters in millimeters.
    """
    return SpaceColonizationParams(
        influence_radius=12.0,  # 12mm
        kill_radius=3.0,  # 3mm
        step_size=4.0,  # 4mm
        min_radius=0.3,  # 0.3mm
        taper_factor=0.93,
        vessel_type="arterial",
        max_steps=180,
        max_curvature_deg=65.0,
        min_clearance=1.2,  # 1.2mm
        encourage_bifurcation=True,
        min_attractions_for_bifurcation=2,
        max_children_per_node=2,
        bifurcation_angle_threshold_deg=38.0,
        bifurcation_probability=0.75,
    )


def brain_arterial() -> SpaceColonizationParams:
    """
    Cerebral arterial tree.
    
    Characteristics:
    - Very dense for high metabolic demand
    - Strict curvature constraints
    - Small vessels
    
    Units: All spatial parameters in millimeters.
    """
    return SpaceColonizationParams(
        influence_radius=6.0,  # 6mm - very dense
        kill_radius=1.2,  # 1.2mm
        step_size=2.0,  # 2mm - small steps
        min_radius=0.1,  # 0.1mm - very fine
        taper_factor=0.88,  # Aggressive tapering
        vessel_type="arterial",
        max_steps=300,
        max_curvature_deg=45.0,  # Strict curvature
        min_clearance=0.6,  # 0.6mm
        encourage_bifurcation=True,
        min_attractions_for_bifurcation=2,
        max_children_per_node=2,
        bifurcation_angle_threshold_deg=32.0,
        bifurcation_probability=0.9,
    )


def dense_bifurcation() -> SpaceColonizationParams:
    """
    Dense bifurcating tree with aggressive branching.
    
    Characteristics:
    - Large influence radius to attract many tissue points
    - Small kill radius for fine perfusion
    - Low bifurcation angle threshold for frequent splits
    - High bifurcation probability
    - Minimal directional bias for organic growth
    
    Units: All spatial parameters in millimeters.
    """
    return SpaceColonizationParams(
        influence_radius=25.0,  # 25mm - large influence for many attractions
        kill_radius=2.0,  # 2mm - small kill radius
        step_size=5.0,  # 5mm - moderate steps
        min_radius=0.3,  # 0.3mm
        taper_factor=0.95,
        vessel_type="arterial",
        max_steps=200,
        max_curvature_deg=None,  # No curvature constraint for organic growth
        min_clearance=None,  # No clearance check for denser packing
        encourage_bifurcation=True,
        min_attractions_for_bifurcation=2,  # Lower threshold
        max_children_per_node=2,
        bifurcation_angle_threshold_deg=20.0,  # Lower angle threshold
        bifurcation_probability=0.9,  # High probability
        directional_bias=0.0,  # No directional bias
        smoothing_weight=0.0,  # No smoothing for more organic branching
    )


PRESETS = {
    "liver_arterial_dense": liver_arterial_dense,
    "liver_venous_sparse": liver_venous_sparse,
    "kidney_arterial": kidney_arterial,
    "sparse_debug": sparse_debug,
    "lung_arterial": lung_arterial,
    "brain_arterial": brain_arterial,
    "dense_bifurcation": dense_bifurcation,
}


def get_preset(name: str) -> SpaceColonizationParams:
    """
    Get a parameter preset by name.
    
    Parameters
    ----------
    name : str
        Preset name (e.g., "liver_arterial_dense", "sparse_debug")
        
    Returns
    -------
    SpaceColonizationParams
        Parameter configuration
        
    Raises
    ------
    ValueError
        If preset name is not recognized
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    
    return PRESETS[name]()


def list_presets() -> list:
    """List all available preset names."""
    return list(PRESETS.keys())
