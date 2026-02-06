"""
Scaffold generation REST API endpoints.

Provides endpoints for:
- POST /generate: Full scaffold generation
- POST /preview: Fast preview without final boolean operations
- POST /validate: Parameter validation
- GET /export/{scaffold_id}: Download generated STL
- GET /presets: List available scaffold presets
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Union
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.config import get_settings

from app.models.scaffold import (
    ScaffoldType,
    ScaffoldParams,
    # Legacy 3
    VascularNetworkParams,
    PorousDiscParams,
    PrimitiveParams,
    # Tubular 7
    TubularConduitParams,
    VascularPerfusionDishParams,
    BloodVesselParams,
    NerveConduitParams,
    SpinalCordParams,
    BladderParams,
    TracheaParams,
    # Organ 6
    HepaticLobuleParams,
    CardiacPatchParams,
    KidneyTubuleParams,
    LungAlveoliParams,
    PancreaticIsletParams,
    LiverSinusoidParams,
    # Skeletal 7
    TrabecularBoneParams,
    OsteochondralParams,
    ArticularCartilageParams,
    MeniscusParams,
    TendonLigamentParams,
    IntervertebralDiscParams,
    HaversianBoneParams,
    # Soft Tissue 4
    MultilayerSkinParams,
    SkeletalMuscleParams,
    CorneaParams,
    AdiposeParams,
    # Dental 3
    DentinPulpParams,
    EarAuricleParams,
    NasalSeptumParams,
    # Lattice 6
    LatticeParams,
    GyroidParams,
    SchwarzPParams,
    OctetTrussParams,
    VoronoiParams,
    HoneycombParams,
    # Microfluidic 3
    OrganOnChipParams,
    GradientScaffoldParams,
    PerfusableNetworkParams,
    # Advanced Vascular 2
    SpaceColonizationParams,
    BifurcatingTreeParams,
    # Supporting types
    MeshData,
    BoundingBox,
    PorePattern,
    InnerTexture,
    UnitCell,
    PrimitiveShape,
    GradientType,
    ArchitectureType,
    OsteonPattern,
    CurveType,
    NetworkType,
)

# Legacy geometry imports
from app.geometry import (
    # Legacy 3
    generate_vascular_network_from_dict,
    generate_porous_disc_from_dict,
    generate_primitive_from_dict,
    check_manifold_available,
    # Tubular 7
    generate_tubular_conduit_from_dict,
    generate_vascular_perfusion_dish_from_dict,
    generate_blood_vessel_from_dict,
    generate_nerve_conduit_from_dict,
    generate_spinal_cord_from_dict,
    generate_bladder_from_dict,
    generate_trachea_from_dict,
    # Lattice 6
    generate_lattice_from_dict,
    generate_gyroid_from_dict,
    generate_schwarz_p_from_dict,
    generate_octet_truss_from_dict,
    generate_voronoi_from_dict,
    generate_honeycomb_from_dict,
)

# Skeletal tissue generators
from app.geometry.skeletal import (
    generate_trabecular_bone_from_dict,
    generate_osteochondral_from_dict,
    generate_articular_cartilage_from_dict,
    generate_meniscus_from_dict,
    generate_tendon_ligament_from_dict,
    generate_intervertebral_disc_from_dict,
    generate_haversian_bone_from_dict,
)

# Organ-specific generators
from app.geometry.organ import (
    generate_hepatic_lobule_from_dict,
    generate_cardiac_patch_from_dict,
    generate_kidney_tubule_from_dict,
    generate_lung_alveoli_from_dict,
    generate_pancreatic_islet_from_dict,
    generate_liver_sinusoid_from_dict,
)

# Soft tissue generators
from app.geometry.soft_tissue import (
    generate_multilayer_skin_from_dict,
    generate_skeletal_muscle_from_dict,
    generate_cornea_from_dict,
    generate_adipose_tissue_from_dict,
)

# Dental generators
from app.geometry.dental import (
    generate_dentin_pulp_from_dict,
    generate_ear_auricle_from_dict,
    generate_nasal_septum_from_dict,
)

# Microfluidic generators
from app.geometry.microfluidic import (
    generate_organ_on_chip_from_dict,
    generate_gradient_scaffold_from_dict,
    generate_perfusable_network_from_dict,
)

# Advanced vascular generators (AOG-powered)
from app.vascular import (
    generate_space_colonization_from_dict,
    generate_bifurcating_tree_from_dict,
)

from app.geometry.stl_export import (
    manifold_to_stl_binary,
    manifold_to_stl_ascii,
    manifold_to_mesh_dict,
    get_bounding_box,
    stl_to_base64,
)
from app.core.logging import get_logger

try:
    import manifold3d as m3d
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["scaffolds"])

# In-memory cache for generated scaffolds (scaffold_id -> (manifold, stl_bytes, metadata))
# In production, use Redis or a proper cache
_scaffold_cache: Dict[str, tuple] = {}
_CACHE_MAX_SIZE = 50


# ============================================================================
# Request/Response Models
# ============================================================================


class GenerateRequest(BaseModel):
    """Request body for scaffold generation."""

    type: ScaffoldType = Field(description="Scaffold type to generate")
    params: Dict[str, Any] = Field(default_factory=dict, description="Type-specific parameters")
    preview_only: bool = Field(default=False, description="Skip final boolean operations for speed")
    invert: bool = Field(default=False, description="Invert geometry (swap solid/void spaces)")


class MeshResponse(BaseModel):
    """Mesh data in response."""

    vertices: List[float] = Field(description="Flattened vertex array [x,y,z,...]")
    indices: List[int] = Field(description="Triangle indices")
    normals: List[float] = Field(description="Vertex normals [nx,ny,nz,...]")


class StatsResponse(BaseModel):
    """Generation statistics."""

    triangle_count: int = Field(description="Number of triangles")
    volume_mm3: float = Field(description="Volume in cubic millimeters")
    generation_time_ms: float = Field(description="Generation time in milliseconds")


class GenerateResponse(BaseModel):
    """Response for scaffold generation."""

    success: bool = Field(description="Whether generation succeeded")
    scaffold_id: str = Field(description="ID for retrieving the scaffold later")
    mesh: MeshResponse = Field(description="3D mesh data")
    stl_base64: Optional[str] = Field(default=None, description="Base64-encoded STL file")
    stats: StatsResponse = Field(description="Generation statistics")
    bounding_box: Dict[str, List[float]] = Field(description="Bounding box {min: [x,y,z], max: [x,y,z]}")
    inverted: bool = Field(default=False, description="Whether the geometry was inverted")


class ValidateRequest(BaseModel):
    """Request body for parameter validation."""

    type: ScaffoldType = Field(description="Scaffold type")
    params: Dict[str, Any] = Field(default_factory=dict, description="Parameters to validate")


class ValidateResponse(BaseModel):
    """Response for parameter validation."""

    valid: bool = Field(description="Whether parameters are valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")


class PresetInfo(BaseModel):
    """Information about a scaffold preset."""

    id: str = Field(description="Preset identifier")
    name: str = Field(description="Human-readable name")
    type: ScaffoldType = Field(description="Scaffold type")
    params: Dict[str, Any] = Field(description="Preset parameters")
    description: str = Field(default="", description="Preset description")
    category: str = Field(default="general", description="Preset category")


class PresetsResponse(BaseModel):
    """Response listing available presets."""

    presets: List[PresetInfo] = Field(description="Available presets")


# ============================================================================
# Presets Data
# ============================================================================

PRESETS: List[PresetInfo] = [
    # -------------------------------------------------------------------------
    # Legacy Presets (3 types: vascular_network, porous_disc, primitive)
    # -------------------------------------------------------------------------
    PresetInfo(
        id="vascular_4inlet",
        name="4-Inlet Vascular Network",
        type=ScaffoldType.VASCULAR_NETWORK,
        description="Standard vascular network with 4 inlet channels",
        category="legacy",
        params={
            "inlets": 4,
            "levels": 3,
            "splits": 2,
            "spread": 0.5,
            "ratio": 0.79,
            "curvature": 0.3,
            "tips_down": True,
            "outer_radius_mm": 10.0,
            "height_mm": 5.0,
            "inlet_radius_mm": 1.0,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="vascular_9inlet",
        name="9-Inlet Vascular Network",
        type=ScaffoldType.VASCULAR_NETWORK,
        description="High-density vascular network with 9 inlet channels",
        category="legacy",
        params={
            "inlets": 9,
            "levels": 2,
            "splits": 2,
            "spread": 0.4,
            "ratio": 0.79,
            "curvature": 0.4,
            "tips_down": True,
            "outer_radius_mm": 12.0,
            "height_mm": 6.0,
            "inlet_radius_mm": 0.8,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="porous_disc_standard",
        name="Standard Porous Disc",
        type=ScaffoldType.POROUS_DISC,
        description="Hexagonal pore pattern disc for general tissue engineering",
        category="legacy",
        params={
            "diameter_mm": 10.0,
            "height_mm": 2.0,
            "pore_diameter_um": 200.0,
            "pore_spacing_um": 400.0,
            "pore_pattern": "hexagonal",
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="porous_disc_fine",
        name="Fine Pore Disc",
        type=ScaffoldType.POROUS_DISC,
        description="Smaller pores for cell seeding applications",
        category="legacy",
        params={
            "diameter_mm": 8.0,
            "height_mm": 1.5,
            "pore_diameter_um": 100.0,
            "pore_spacing_um": 200.0,
            "pore_pattern": "hexagonal",
            "resolution": 24,
        },
    ),
    PresetInfo(
        id="primitive_hollow_cylinder",
        name="Hollow Cylinder",
        type=ScaffoldType.PRIMITIVE,
        description="Basic hollow cylindrical scaffold",
        category="legacy",
        params={
            "shape": "cylinder",
            "dimensions": {"radius_mm": 5.0, "height_mm": 10.0},
            "modifications": [{"operation": "shell", "params": {"wall_thickness_mm": 1.0}}],
            "resolution": 32,
        },
    ),
    # -------------------------------------------------------------------------
    # Tubular Presets (7 types: tubular_conduit, vascular_perfusion_dish, blood_vessel, nerve_conduit, spinal_cord, bladder, trachea)
    # -------------------------------------------------------------------------
    PresetInfo(
        id="tubular_nerve",
        name="Simple Nerve Conduit",
        type=ScaffoldType.TUBULAR_CONDUIT,
        description="Grooved nerve guidance conduit",
        category="tubular",
        params={
            "outer_diameter_mm": 4.0,
            "wall_thickness_mm": 0.5,
            "length_mm": 15.0,
            "inner_texture": "grooved",
            "groove_count": 8,
            "resolution": 32,
        },
    ),
    PresetInfo(
        id="tubular_vascular",
        name="Simple Vascular Graft",
        type=ScaffoldType.TUBULAR_CONDUIT,
        description="Porous vascular graft scaffold",
        category="tubular",
        params={
            "outer_diameter_mm": 6.0,
            "wall_thickness_mm": 1.0,
            "length_mm": 30.0,
            "inner_texture": "porous",
            "resolution": 32,
        },
    ),
    PresetInfo(
        id="vascular_perfusion_dish_4inlet",
        name="4-Inlet Vascular Perfusion Dish",
        type=ScaffoldType.VASCULAR_PERFUSION_DISH,
        description="Vascular network with collision detection for non-overlapping branches",
        category="tubular",
        params={
            "inlets": 4,
            "levels": 2,
            "splits": 2,
            "spread": 0.35,
            "ratio": 0.79,
            "cone_angle": 60.0,
            "curvature": 0.3,
            "bottom_height": 0.06,
            "radius_variation": 0.25,
            "flip_chance": 0.30,
            "z_variation": 0.35,
            "angle_variation": 0.40,
            "collision_buffer": 0.08,
            "even_spread": True,
            "deterministic": False,
            "tips_down": False,
            "outer_radius_mm": 4.875,
            "height_mm": 2.0,
            "inlet_radius_mm": 0.35,
            "seed": 42,
            "resolution": 12,
        },
    ),
    PresetInfo(
        id="vascular_perfusion_dish_deterministic",
        name="Deterministic Vascular Perfusion Dish",
        type=ScaffoldType.VASCULAR_PERFUSION_DISH,
        description="Grid-aligned vascular channels with no randomization",
        category="tubular",
        params={
            "inlets": 9,
            "levels": 2,
            "splits": 2,
            "spread": 0.35,
            "ratio": 0.79,
            "cone_angle": 60.0,
            "curvature": 0.0,
            "bottom_height": 0.06,
            "radius_variation": 0.0,
            "flip_chance": 0.0,
            "z_variation": 0.0,
            "angle_variation": 0.0,
            "collision_buffer": 0.08,
            "even_spread": True,
            "deterministic": True,
            "tips_down": True,
            "outer_radius_mm": 4.875,
            "height_mm": 2.0,
            "inlet_radius_mm": 0.35,
            "seed": 42,
            "resolution": 12,
        },
    ),
    # -------------------------------------------------------------------------
    # Skeletal Tissue Presets (7 types)
    # -------------------------------------------------------------------------
    PresetInfo(
        id="trabecular_bone_femur",
        name="Femoral Trabecular Bone",
        type=ScaffoldType.TRABECULAR_BONE,
        description="Trabecular bone scaffold mimicking femoral head architecture",
        category="skeletal",
        params={
            "porosity": 0.7,
            "pore_size_um": 400.0,
            "strut_thickness_um": 200.0,
            "anisotropy_ratio": 1.5,
            "bounding_box": {"x": 10, "y": 10, "z": 10},
            "seed": 42,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="osteochondral_knee",
        name="Knee Osteochondral Plug",
        type=ScaffoldType.OSTEOCHONDRAL,
        description="Gradient scaffold for osteochondral defect repair",
        category="skeletal",
        params={
            "bone_depth": 3.0,
            "cartilage_depth": 2.0,
            "transition_width": 1.0,
            "gradient_type": "linear",
            "diameter": 8.0,
            "bone_porosity": 0.7,
            "cartilage_porosity": 0.85,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="meniscus_medial",
        name="Medial Meniscus",
        type=ScaffoldType.MENISCUS,
        description="Wedge-shaped meniscus for knee repair",
        category="skeletal",
        params={
            "inner_radius": 12.0,
            "outer_radius": 20.0,
            "height": 8.0,
            "wedge_angle_deg": 20.0,
            "zone_count": 3,
            "fiber_diameter": 0.2,
            "arc_degrees": 300.0,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="intervertebral_disc_lumbar",
        name="Lumbar Intervertebral Disc",
        type=ScaffoldType.INTERVERTEBRAL_DISC,
        description="Full IVD scaffold with annulus fibrosus and nucleus pulposus",
        category="skeletal",
        params={
            "disc_diameter": 40.0,
            "disc_height": 10.0,
            "af_ring_count": 5,
            "np_diameter": 15.0,
            "af_layer_angle": 30.0,
            "fiber_diameter": 0.15,
            "np_porosity": 0.9,
            "resolution": 16,
        },
    ),
    # -------------------------------------------------------------------------
    # Organ-Specific Presets
    # -------------------------------------------------------------------------
    PresetInfo(
        id="hepatic_lobule_single",
        name="Single Hepatic Lobule",
        type=ScaffoldType.HEPATIC_LOBULE,
        description="Single hexagonal liver lobule unit",
        category="organ",
        params={
            "num_lobules": 1,
            "lobule_radius": 1.5,
            "lobule_height": 3.0,
            "wall_thickness": 0.1,
            "central_vein_radius": 0.15,
            "portal_vein_radius": 0.12,
            "show_sinusoids": False,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="hepatic_lobule_cluster",
        name="Hepatic Lobule Cluster (7)",
        type=ScaffoldType.HEPATIC_LOBULE,
        description="7-lobule cluster for liver tissue engineering",
        category="organ",
        params={
            "num_lobules": 7,
            "lobule_radius": 1.5,
            "lobule_height": 3.0,
            "wall_thickness": 0.1,
            "central_vein_radius": 0.15,
            "portal_vein_radius": 0.12,
            "show_sinusoids": False,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="cardiac_patch_standard",
        name="Cardiac Patch (10x10mm)",
        type=ScaffoldType.CARDIAC_PATCH,
        description="Aligned microfiber cardiac patch for cardiomyocyte seeding",
        category="organ",
        params={
            "fiber_spacing": 300.0,
            "fiber_diameter": 100.0,
            "patch_size": {"x": 10, "y": 10, "z": 1},
            "layer_count": 3,
            "alignment_angle": 0.0,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="lung_alveoli_terminal",
        name="Terminal Lung Unit",
        type=ScaffoldType.LUNG_ALVEOLI,
        description="Branching airways with terminal alveolar sacs",
        category="organ",
        params={
            "branch_generations": 3,
            "alveoli_diameter": 200.0,
            "airway_diameter": 1.0,
            "branch_angle": 35.0,
            "bounding_box": {"x": 10, "y": 10, "z": 10},
            "resolution": 16,
        },
    ),
    # -------------------------------------------------------------------------
    # Soft Tissue Presets
    # -------------------------------------------------------------------------
    PresetInfo(
        id="multilayer_skin_standard",
        name="Full-Thickness Skin",
        type=ScaffoldType.MULTILAYER_SKIN,
        description="3-layer skin scaffold with vascular channels",
        category="soft_tissue",
        params={
            "epidermis_thickness_mm": 0.15,
            "dermis_thickness_mm": 1.5,
            "hypodermis_thickness_mm": 3.0,
            "diameter_mm": 10.0,
            "pore_gradient": [0.3, 0.5, 0.7],
            "vascular_channel_diameter_mm": 0.2,
            "vascular_channel_count": 8,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="cornea_standard",
        name="Cornea Scaffold",
        type=ScaffoldType.CORNEA,
        description="Curved dome cornea with lamellar structure",
        category="soft_tissue",
        params={
            "diameter_mm": 11.0,
            "thickness_mm": 0.55,
            "radius_of_curvature_mm": 7.8,
            "lamella_count": 5,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="skeletal_muscle_parallel",
        name="Parallel Muscle Fibers",
        type=ScaffoldType.SKELETAL_MUSCLE,
        description="Aligned myofiber channels for muscle regeneration",
        category="soft_tissue",
        params={
            "fiber_diameter_um": 75.0,
            "fiber_spacing_um": 150.0,
            "pennation_angle_deg": 0.0,
            "fascicle_count": 3,
            "architecture_type": "parallel",
            "length_mm": 10.0,
            "width_mm": 5.0,
            "height_mm": 5.0,
            "resolution": 16,
        },
    ),
    # -------------------------------------------------------------------------
    # Tubular Presets (continued: blood_vessel, nerve_conduit, trachea, etc.)
    # -------------------------------------------------------------------------
    PresetInfo(
        id="blood_vessel_small",
        name="Small Artery (5mm)",
        type=ScaffoldType.BLOOD_VESSEL,
        description="3-layer arterial scaffold",
        category="tubular",
        params={
            "inner_diameter_mm": 5.0,
            "wall_thickness_mm": 1.0,
            "layer_ratios": [0.1, 0.5, 0.4],
            "length_mm": 30.0,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="nerve_conduit_multichannel",
        name="Multi-Channel Nerve Conduit",
        type=ScaffoldType.NERVE_CONDUIT,
        description="7-channel nerve guidance conduit",
        category="tubular",
        params={
            "outer_diameter_mm": 4.0,
            "channel_count": 7,
            "channel_diameter_um": 200.0,
            "wall_thickness_mm": 0.5,
            "length_mm": 20.0,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="trachea_segment",
        name="Trachea Segment",
        type=ScaffoldType.TRACHEA,
        description="C-shaped cartilage rings for tracheal repair",
        category="tubular",
        params={
            "outer_diameter_mm": 20.0,
            "ring_thickness_mm": 3.0,
            "ring_spacing_mm": 4.0,
            "ring_count": 10,
            "posterior_gap_angle_deg": 90.0,
            "resolution": 16,
        },
    ),
    # -------------------------------------------------------------------------
    # Dental/Craniofacial Presets
    # -------------------------------------------------------------------------
    PresetInfo(
        id="dentin_pulp_molar",
        name="Molar Tooth Scaffold",
        type=ScaffoldType.DENTIN_PULP,
        description="Tooth scaffold with dentin shell and pulp chamber",
        category="dental",
        params={
            "tooth_height": 12.0,
            "crown_diameter": 10.0,
            "root_length": 12.0,
            "root_diameter": 5.0,
            "pulp_chamber_size": 0.4,
            "dentin_thickness": 3.0,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="ear_auricle_adult",
        name="Adult Ear Framework",
        type=ScaffoldType.EAR_AURICLE,
        description="Cartilage framework for auricular reconstruction",
        category="dental",
        params={
            "scale_factor": 1.0,
            "thickness": 2.0,
            "helix_definition": 0.7,
            "antihelix_depth": 0.3,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="nasal_septum_standard",
        name="Nasal Septum",
        type=ScaffoldType.NASAL_SEPTUM,
        description="Curved cartilage sheet for nasal reconstruction",
        category="dental",
        params={
            "length": 40.0,
            "height": 30.0,
            "thickness": 3.0,
            "curvature_radius": 75.0,
            "curve_type": "single",
            "resolution": 16,
        },
    ),
    # -------------------------------------------------------------------------
    # Lattice Presets (6 types: basic, gyroid, schwarz_p, octet_truss, voronoi, honeycomb)
    # -------------------------------------------------------------------------
    PresetInfo(
        id="lattice_cubic",
        name="Cubic Lattice",
        type=ScaffoldType.LATTICE,
        description="Simple cubic lattice structure",
        category="lattice",
        params={
            "bounding_box": {"x": 10, "y": 10, "z": 10},
            "unit_cell": "cubic",
            "cell_size_mm": 2.0,
            "strut_diameter_mm": 0.5,
            "resolution": 8,
        },
    ),
    PresetInfo(
        id="lattice_bcc",
        name="BCC Lattice",
        type=ScaffoldType.LATTICE,
        description="Body-centered cubic lattice with higher mechanical strength",
        category="lattice",
        params={
            "bounding_box": {"x": 10, "y": 10, "z": 10},
            "unit_cell": "bcc",
            "cell_size_mm": 2.5,
            "strut_diameter_mm": 0.4,
            "resolution": 8,
        },
    ),
    PresetInfo(
        id="gyroid_standard",
        name="Gyroid TPMS",
        type=ScaffoldType.GYROID,
        description="Triply periodic minimal surface with excellent permeability",
        category="lattice",
        params={
            "bounding_box": {"x": 10, "y": 10, "z": 10},
            "cell_size_mm": 2.0,
            "wall_thickness_mm": 0.3,
            "isovalue": 0.0,
            "samples_per_cell": 20,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="schwarz_p_standard",
        name="Schwarz P TPMS",
        type=ScaffoldType.SCHWARZ_P,
        description="TPMS with cubic symmetry and circular openings",
        category="lattice",
        params={
            "bounding_box": {"x": 10, "y": 10, "z": 10},
            "cell_size_mm": 2.0,
            "wall_thickness_mm": 0.3,
            "isovalue": 0.0,
            "samples_per_cell": 20,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="octet_truss_standard",
        name="Octet Truss",
        type=ScaffoldType.OCTET_TRUSS,
        description="High strength-to-weight ratio lattice structure",
        category="lattice",
        params={
            "bounding_box": {"x": 10, "y": 10, "z": 10},
            "cell_size_mm": 2.0,
            "strut_diameter_mm": 0.3,
            "resolution": 8,
        },
    ),
    PresetInfo(
        id="voronoi_organic",
        name="Voronoi Organic",
        type=ScaffoldType.VORONOI,
        description="Randomized cellular structure mimicking trabecular bone",
        category="lattice",
        params={
            "bounding_box": {"x": 10, "y": 10, "z": 10},
            "cell_count": 30,
            "strut_diameter_mm": 0.3,
            "seed": 42,
            "margin_factor": 0.2,
            "resolution": 8,
        },
    ),
    PresetInfo(
        id="honeycomb_standard",
        name="Honeycomb",
        type=ScaffoldType.HONEYCOMB,
        description="Hexagonal honeycomb for load-bearing applications",
        category="lattice",
        params={
            "bounding_box": {"x": 10, "y": 10, "z": 5},
            "cell_size_mm": 2.0,
            "wall_thickness_mm": 0.3,
            "resolution": 16,
        },
    ),
    # -------------------------------------------------------------------------
    # Microfluidic Presets
    # -------------------------------------------------------------------------
    PresetInfo(
        id="organ_on_chip_dual",
        name="Dual-Chamber Organ Chip",
        type=ScaffoldType.ORGAN_ON_CHIP,
        description="Microfluidic chip with 2 tissue chambers",
        category="microfluidic",
        params={
            "channel_width_mm": 0.3,
            "channel_depth_mm": 0.1,
            "chamber_size": {"x": 3, "y": 2, "z": 0.15},
            "chamber_count": 2,
            "inlet_count": 2,
            "chip_size": {"x": 15, "y": 10, "z": 2},
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="gradient_scaffold_linear",
        name="Linear Gradient Scaffold",
        type=ScaffoldType.GRADIENT_SCAFFOLD,
        description="Z-direction porosity gradient (20% to 80%)",
        category="microfluidic",
        params={
            "dimensions": {"x": 10, "y": 10, "z": 10},
            "gradient_direction": "z",
            "start_porosity": 0.2,
            "end_porosity": 0.8,
            "gradient_type": "linear",
            "pore_base_size_mm": 0.5,
            "grid_spacing_mm": 1.5,
            "resolution": 16,
        },
    ),
    PresetInfo(
        id="perfusable_network_arterial",
        name="Arterial Perfusable Network",
        type=ScaffoldType.PERFUSABLE_NETWORK,
        description="Murray's law vascular network for thick tissue constructs",
        category="microfluidic",
        params={
            "inlet_diameter_mm": 2.0,
            "branch_generations": 3,
            "murray_ratio": 0.79,
            "network_type": "arterial",
            "bounding_box": {"x": 10, "y": 10, "z": 10},
            "branching_angle_deg": 30.0,
            "resolution": 16,
        },
    ),
    # -------------------------------------------------------------------------
    # Advanced Vascular Presets (2 types: space_colonization, bifurcating_tree)
    # -------------------------------------------------------------------------
    PresetInfo(
        id="space_colonization_single",
        name="Single Inlet Vascular Network",
        type=ScaffoldType.SPACE_COLONIZATION,
        description="Organic vascular growth from single inlet using space colonization",
        category="vascular",
        params={
            "inlets": [{
                "position": [0.0, 0.0, 0.001],
                "radius": 0.0002,
                "direction": [0.0, 0.0, -1.0]
            }],
            "domain_radius": 0.005,
            "domain_height": 0.002,
            "num_attractors": 50000,
            "influence_radius": 0.002,
            "kill_radius": 0.00025,
            "step_size": 0.00018,
            "max_iterations": 300,
            "enable_bifurcation": True,
            "bifurcation_probability": 0.35,
            "min_attractors_for_split": 8,
            "max_children_per_node": 2,
            "min_radius": 0.00003,
            "max_radius": 0.0002,
            "taper_factor": 0.95,
            "multi_inlet_mode": "blended",
            "directional_bias": 0.35,
            "max_deviation_deg": 70.0,
            "radial_resolution": 12,
            "random_seed": 42,
            "resolution": "medium",
        },
    ),
    PresetInfo(
        id="space_colonization_five_inlet",
        name="5-Inlet Vascular Network",
        type=ScaffoldType.SPACE_COLONIZATION,
        description="Multi-inlet organic vascular network with blended growth",
        category="vascular",
        params={
            "inlets": [
                {"position": [0.0, 0.0, 0.001], "radius": 0.0002, "direction": [0.0, 0.0, -1.0]},
                {"position": [0.003, 0.0, 0.001], "radius": 0.0002, "direction": [0.0, 0.0, -1.0]},
                {"position": [-0.003, 0.0, 0.001], "radius": 0.0002, "direction": [0.0, 0.0, -1.0]},
                {"position": [0.0, 0.003, 0.001], "radius": 0.0002, "direction": [0.0, 0.0, -1.0]},
                {"position": [0.0, -0.003, 0.001], "radius": 0.0002, "direction": [0.0, 0.0, -1.0]},
            ],
            "domain_radius": 0.006,
            "domain_height": 0.002,
            "num_attractors": 80000,
            "influence_radius": 0.002,
            "kill_radius": 0.00025,
            "step_size": 0.00018,
            "max_iterations": 300,
            "enable_bifurcation": True,
            "bifurcation_probability": 0.4,
            "min_attractors_for_split": 8,
            "max_children_per_node": 2,
            "min_radius": 0.00003,
            "max_radius": 0.0002,
            "taper_factor": 0.95,
            "multi_inlet_mode": "blended",
            "directional_bias": 0.35,
            "max_deviation_deg": 70.0,
            "radial_resolution": 12,
            "random_seed": 42,
            "resolution": "medium",
        },
    ),
    PresetInfo(
        id="bifurcating_tree_standard",
        name="Standard Bifurcating Tree",
        type=ScaffoldType.BIFURCATING_TREE,
        description="Regular binary tree with Murray's law radius tapering",
        category="vascular",
        params={
            "root_position": [0.0, 0.0, 0.001],
            "root_direction": [0.0, 0.0, -1.0],
            "root_radius": 0.0002,
            "branching_levels": 5,
            "branches_per_node": 2,
            "branching_angle_deg": 35.0,
            "segment_length": 0.0003,
            "taper_segment_length": True,
            "length_taper_factor": 0.85,
            "radius_mode": "murray",
            "murray_exponent": 3.0,
            "min_terminal_radius": 0.00003,
            "add_variation": False,
            "angle_variation_deg": 10.0,
            "length_variation_pct": 15.0,
            "radial_resolution": 12,
            "random_seed": 42,
            "resolution": "medium",
        },
    ),
    PresetInfo(
        id="bifurcating_tree_ternary",
        name="Ternary Bifurcating Tree",
        type=ScaffoldType.BIFURCATING_TREE,
        description="Three-way branching tree for dense vascular networks",
        category="vascular",
        params={
            "root_position": [0.0, 0.0, 0.001],
            "root_direction": [0.0, 0.0, -1.0],
            "root_radius": 0.00025,
            "branching_levels": 4,
            "branches_per_node": 3,
            "branching_angle_deg": 40.0,
            "segment_length": 0.00025,
            "taper_segment_length": True,
            "length_taper_factor": 0.85,
            "radius_mode": "murray",
            "murray_exponent": 3.0,
            "min_terminal_radius": 0.00003,
            "add_variation": True,
            "angle_variation_deg": 15.0,
            "length_variation_pct": 20.0,
            "radial_resolution": 12,
            "random_seed": 42,
            "resolution": "medium",
        },
    ),
]


# ============================================================================
# Helper Functions
# ============================================================================


def _invert_manifold(manifold, padding_mm: float = 1.0):
    """
    Invert a manifold by subtracting it from its bounding box.

    This swaps positive and negative spaces - what was solid becomes void
    and vice versa.

    Args:
        manifold: manifold3d Manifold object to invert
        padding_mm: Padding to add around the bounding box (default: 1.0mm)

    Returns:
        Inverted manifold (bounding_box - original)
    """
    if not MANIFOLD_AVAILABLE:
        raise RuntimeError("manifold3d library not available for inversion")

    bbox_min, bbox_max = get_bounding_box(manifold)
    logger.debug(f"Inversion: bbox_min={bbox_min}, bbox_max={bbox_max}")

    # Calculate dimensions with padding
    dx = bbox_max[0] - bbox_min[0] + 2 * padding_mm
    dy = bbox_max[1] - bbox_min[1] + 2 * padding_mm
    dz = bbox_max[2] - bbox_min[2] + 2 * padding_mm

    # Validate dimensions
    if dx <= 0 or dy <= 0 or dz <= 0:
        logger.warning(f"Invalid bounding box dimensions: dx={dx}, dy={dy}, dz={dz}")
        return manifold  # Return original if dimensions are invalid

    logger.debug(f"Inversion: creating cube with dimensions [{dx}, {dy}, {dz}]")

    # Create bounding box cube
    bbox_cube = m3d.Manifold.cube([dx, dy, dz])

    # Translate to match original position (with padding offset)
    translate_vec = [
        bbox_min[0] - padding_mm,
        bbox_min[1] - padding_mm,
        bbox_min[2] - padding_mm
    ]
    logger.debug(f"Inversion: translating cube by {translate_vec}")
    bbox_cube = bbox_cube.translate(translate_vec)

    # Subtract original from bounding box to invert
    result = bbox_cube - manifold
    logger.debug(f"Inversion complete: result is_empty={result.is_empty() if hasattr(result, 'is_empty') else 'unknown'}")

    return result


def _convert_params_for_generator(scaffold_type: ScaffoldType, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert API params to format expected by geometry generators.

    Handles field name differences and type conversions.
    """
    converted = dict(params)

    if scaffold_type == ScaffoldType.VASCULAR_NETWORK:
        # Map outer_radius_mm to outer_radius for the generator
        if "outer_radius_mm" in converted:
            converted["outer_radius"] = converted.pop("outer_radius_mm")
        if "height_mm" in converted:
            converted["height"] = converted.pop("height_mm")
            converted["scaffold_height"] = converted["height"] * 0.96  # Default scaffold height ratio
        if "inlet_radius_mm" in converted:
            converted["inlet_radius"] = converted.pop("inlet_radius_mm")
        # Map inner_radius if not specified
        if "inner_radius" not in converted and "outer_radius" in converted:
            converted["inner_radius"] = converted["outer_radius"] * 0.915

    elif scaffold_type == ScaffoldType.VASCULAR_PERFUSION_DISH:
        # Map outer_radius_mm to outer_radius for the generator
        if "outer_radius_mm" in converted:
            converted["outer_radius"] = converted.pop("outer_radius_mm")
        if "height_mm" in converted:
            converted["height"] = converted.pop("height_mm")
            converted["scaffold_height"] = converted["height"] * 0.96
        if "inlet_radius_mm" in converted:
            converted["inlet_radius"] = converted.pop("inlet_radius_mm")
        if "rim_height_mm" in converted:
            converted["rim_height"] = converted.pop("rim_height_mm")
        # Map inner_radius if not specified
        if "inner_radius" not in converted and "outer_radius" in converted:
            converted["inner_radius"] = converted["outer_radius"] * 0.94

    elif scaffold_type == ScaffoldType.TUBULAR_CONDUIT:
        # Handle groove_count for grooved texture
        if converted.get("inner_texture") == "grooved" and "groove_count" not in converted:
            converted["groove_count"] = 8
        # Add groove_depth_mm if not specified
        if "groove_depth_mm" not in converted:
            converted["groove_depth_mm"] = 0.15

    elif scaffold_type == ScaffoldType.LATTICE:
        # Handle bounding_box as dict or tuple
        bbox = converted.get("bounding_box", {})
        if isinstance(bbox, dict):
            converted["bounding_box_mm"] = (
                bbox.get("x", 10.0),
                bbox.get("y", 10.0),
                bbox.get("z", 10.0),
            )
            if "bounding_box" in converted:
                del converted["bounding_box"]

    # Handle bounding_box conversion for all scaffold types that use it
    elif scaffold_type in (
        ScaffoldType.TRABECULAR_BONE,
        ScaffoldType.HAVERSIAN_BONE,
        ScaffoldType.LUNG_ALVEOLI,
        ScaffoldType.GYROID,
        ScaffoldType.SCHWARZ_P,
        ScaffoldType.OCTET_TRUSS,
        ScaffoldType.VORONOI,
        ScaffoldType.HONEYCOMB,
        ScaffoldType.PERFUSABLE_NETWORK,
    ):
        bbox = converted.get("bounding_box", {})
        if isinstance(bbox, dict):
            converted["bounding_box_mm"] = (
                bbox.get("x", 10.0),
                bbox.get("y", 10.0),
                bbox.get("z", 10.0),
            )
            if "bounding_box" in converted:
                del converted["bounding_box"]

    # Handle patch_size/chamber_size/dimensions for microfluidic types
    elif scaffold_type == ScaffoldType.CARDIAC_PATCH:
        patch_size = converted.get("patch_size", {})
        if isinstance(patch_size, dict):
            converted["patch_size_mm"] = (
                patch_size.get("x", 10.0),
                patch_size.get("y", 10.0),
                patch_size.get("z", 1.0),
            )
            if "patch_size" in converted:
                del converted["patch_size"]

    elif scaffold_type == ScaffoldType.ORGAN_ON_CHIP:
        chamber_size = converted.get("chamber_size", {})
        if isinstance(chamber_size, dict):
            converted["chamber_size_mm"] = (
                chamber_size.get("x", 3.0),
                chamber_size.get("y", 2.0),
                chamber_size.get("z", 0.15),
            )
            if "chamber_size" in converted:
                del converted["chamber_size"]
        chip_size = converted.get("chip_size", {})
        if isinstance(chip_size, dict):
            converted["chip_size_mm"] = (
                chip_size.get("x", 15.0),
                chip_size.get("y", 10.0),
                chip_size.get("z", 2.0),
            )
            if "chip_size" in converted:
                del converted["chip_size"]

    elif scaffold_type == ScaffoldType.GRADIENT_SCAFFOLD:
        dimensions = converted.get("dimensions", {})
        if isinstance(dimensions, dict):
            converted["dimensions_mm"] = (
                dimensions.get("x", 10.0),
                dimensions.get("y", 10.0),
                dimensions.get("z", 10.0),
            )
            if "dimensions" in converted:
                del converted["dimensions"]

    return converted


def _generate_scaffold(scaffold_type: ScaffoldType, params: Dict[str, Any], preview_only: bool = False):
    """
    Generate scaffold based on type and parameters.

    Returns:
        Tuple of (manifold, stats_dict)
    """
    converted_params = _convert_params_for_generator(scaffold_type, params)

    # -------------------------------------------------------------------------
    # Legacy (3 types: vascular_network, porous_disc, primitive)
    # -------------------------------------------------------------------------
    if scaffold_type == ScaffoldType.VASCULAR_NETWORK:
        # Vascular returns (body, channels, result) - we want result
        _, _, manifold = generate_vascular_network_from_dict(converted_params)
        mesh = manifold.to_mesh()
        volume = manifold.volume() if hasattr(manifold, "volume") else 0
        stats = {
            "triangle_count": len(mesh.tri_verts) if hasattr(mesh, "tri_verts") else 0,
            "volume_mm3": volume,
            "scaffold_type": "vascular_network",
        }
        return manifold, stats

    elif scaffold_type == ScaffoldType.POROUS_DISC:
        return generate_porous_disc_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.PRIMITIVE:
        return generate_primitive_from_dict(converted_params)

    # -------------------------------------------------------------------------
    # Tubular (7 types: tubular_conduit, vascular_perfusion_dish, blood_vessel, nerve_conduit, spinal_cord, bladder, trachea)
    # -------------------------------------------------------------------------
    elif scaffold_type == ScaffoldType.TUBULAR_CONDUIT:
        return generate_tubular_conduit_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.VASCULAR_PERFUSION_DISH:
        # Vascular perfusion dish returns (body, channels, result) - we want result
        _, _, manifold = generate_vascular_perfusion_dish_from_dict(converted_params)
        mesh = manifold.to_mesh()
        volume = manifold.volume() if hasattr(manifold, "volume") else 0
        stats = {
            "triangle_count": len(mesh.tri_verts) if hasattr(mesh, "tri_verts") else 0,
            "volume_mm3": volume,
            "scaffold_type": "vascular_perfusion_dish",
        }
        return manifold, stats

    elif scaffold_type == ScaffoldType.BLOOD_VESSEL:
        return generate_blood_vessel_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.NERVE_CONDUIT:
        return generate_nerve_conduit_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.SPINAL_CORD:
        return generate_spinal_cord_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.BLADDER:
        return generate_bladder_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.TRACHEA:
        return generate_trachea_from_dict(converted_params)

    # -------------------------------------------------------------------------
    # Organ (6 types)
    # -------------------------------------------------------------------------
    elif scaffold_type == ScaffoldType.HEPATIC_LOBULE:
        return generate_hepatic_lobule_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.CARDIAC_PATCH:
        return generate_cardiac_patch_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.KIDNEY_TUBULE:
        return generate_kidney_tubule_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.LUNG_ALVEOLI:
        return generate_lung_alveoli_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.PANCREATIC_ISLET:
        return generate_pancreatic_islet_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.LIVER_SINUSOID:
        return generate_liver_sinusoid_from_dict(converted_params)

    # -------------------------------------------------------------------------
    # Skeletal (7 types)
    # -------------------------------------------------------------------------
    elif scaffold_type == ScaffoldType.TRABECULAR_BONE:
        return generate_trabecular_bone_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.OSTEOCHONDRAL:
        return generate_osteochondral_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.ARTICULAR_CARTILAGE:
        return generate_articular_cartilage_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.MENISCUS:
        return generate_meniscus_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.TENDON_LIGAMENT:
        return generate_tendon_ligament_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.INTERVERTEBRAL_DISC:
        return generate_intervertebral_disc_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.HAVERSIAN_BONE:
        return generate_haversian_bone_from_dict(converted_params)

    # -------------------------------------------------------------------------
    # Soft Tissue (4 types)
    # -------------------------------------------------------------------------
    elif scaffold_type == ScaffoldType.MULTILAYER_SKIN:
        return generate_multilayer_skin_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.SKELETAL_MUSCLE:
        return generate_skeletal_muscle_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.CORNEA:
        return generate_cornea_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.ADIPOSE:
        return generate_adipose_tissue_from_dict(converted_params)

    # -------------------------------------------------------------------------
    # Dental (3 types)
    # -------------------------------------------------------------------------
    elif scaffold_type == ScaffoldType.DENTIN_PULP:
        return generate_dentin_pulp_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.EAR_AURICLE:
        return generate_ear_auricle_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.NASAL_SEPTUM:
        return generate_nasal_septum_from_dict(converted_params)

    # -------------------------------------------------------------------------
    # Lattice (6 types: basic, gyroid, schwarz_p, octet_truss, voronoi, honeycomb)
    # -------------------------------------------------------------------------
    elif scaffold_type == ScaffoldType.LATTICE:
        return generate_lattice_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.GYROID:
        return generate_gyroid_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.SCHWARZ_P:
        return generate_schwarz_p_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.OCTET_TRUSS:
        return generate_octet_truss_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.VORONOI:
        return generate_voronoi_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.HONEYCOMB:
        return generate_honeycomb_from_dict(converted_params)

    # -------------------------------------------------------------------------
    # Microfluidic (3 types)
    # -------------------------------------------------------------------------
    elif scaffold_type == ScaffoldType.ORGAN_ON_CHIP:
        return generate_organ_on_chip_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.GRADIENT_SCAFFOLD:
        return generate_gradient_scaffold_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.PERFUSABLE_NETWORK:
        return generate_perfusable_network_from_dict(converted_params)

    # -------------------------------------------------------------------------
    # Advanced Vascular (2 types: space_colonization, bifurcating_tree)
    # -------------------------------------------------------------------------
    elif scaffold_type == ScaffoldType.SPACE_COLONIZATION:
        return generate_space_colonization_from_dict(converted_params)

    elif scaffold_type == ScaffoldType.BIFURCATING_TREE:
        return generate_bifurcating_tree_from_dict(converted_params)

    else:
        raise ValueError(f"Unknown scaffold type: {scaffold_type}")


def _validate_params(scaffold_type: ScaffoldType, params: Dict[str, Any]) -> tuple[bool, List[str], List[str]]:
    """
    Validate scaffold parameters.

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []

    try:
        # -------------------------------------------------------------------------
        # Legacy (3 types: vascular_network, porous_disc, primitive)
        # -------------------------------------------------------------------------
        if scaffold_type == ScaffoldType.VASCULAR_NETWORK:
            VascularNetworkParams(**params)
            if params.get("levels", 3) > 5:
                warnings.append("Branching levels > 5 may take a long time to generate")
            if params.get("inlets", 4) > 16:
                warnings.append("More than 16 inlets may produce very dense geometry")

        elif scaffold_type == ScaffoldType.POROUS_DISC:
            PorousDiscParams(**params)
            pore_diameter = params.get("pore_diameter_um", 200)
            if pore_diameter < 100:
                warnings.append("Pore diameter < 100um may be difficult to print")

        elif scaffold_type == ScaffoldType.PRIMITIVE:
            PrimitiveParams(**params)

        # -------------------------------------------------------------------------
        # Tubular (7 types: tubular_conduit, vascular_perfusion_dish, blood_vessel, nerve_conduit, spinal_cord, bladder, trachea)
        # -------------------------------------------------------------------------
        elif scaffold_type == ScaffoldType.TUBULAR_CONDUIT:
            TubularConduitParams(**params)
            wall = params.get("wall_thickness_mm", 1.0)
            if wall < 0.4:
                warnings.append("Wall thickness < 0.4mm is at minimum printable size")

        elif scaffold_type == ScaffoldType.VASCULAR_PERFUSION_DISH:
            VascularPerfusionDishParams(**params)
            if params.get("levels", 2) > 5:
                warnings.append("Branching levels > 5 may take a long time to generate")
            if params.get("inlets", 4) > 16:
                warnings.append("More than 16 inlets may produce very dense geometry")
            if params.get("collision_buffer", 0.08) < 0.03:
                warnings.append("Collision buffer < 0.03 may cause overlapping branches")

        elif scaffold_type == ScaffoldType.BLOOD_VESSEL:
            BloodVesselParams(**params)
            wall = params.get("wall_thickness_mm", 1.0)
            if wall < 0.5:
                warnings.append("Wall thickness < 0.5mm may be fragile for blood vessel")

        elif scaffold_type == ScaffoldType.NERVE_CONDUIT:
            NerveConduitParams(**params)

        elif scaffold_type == ScaffoldType.SPINAL_CORD:
            SpinalCordParams(**params)

        elif scaffold_type == ScaffoldType.BLADDER:
            BladderParams(**params)

        elif scaffold_type == ScaffoldType.TRACHEA:
            TracheaParams(**params)
            ring = params.get("ring_thickness_mm", 3.0)
            if ring < 2.0:
                warnings.append("Ring thickness < 2mm may not provide adequate support")

        # -------------------------------------------------------------------------
        # Organ (6 types)
        # -------------------------------------------------------------------------
        elif scaffold_type == ScaffoldType.HEPATIC_LOBULE:
            HepaticLobuleParams(**params)
            wall = params.get("wall_thickness", 0.1)
            if wall < 0.08:
                warnings.append("Wall thickness < 0.08mm may be difficult to print")

        elif scaffold_type == ScaffoldType.CARDIAC_PATCH:
            CardiacPatchParams(**params)
            fiber = params.get("fiber_diameter", 100)
            if fiber < 80:
                warnings.append("Fiber diameter < 80um may be difficult to print")

        elif scaffold_type == ScaffoldType.KIDNEY_TUBULE:
            KidneyTubuleParams(**params)

        elif scaffold_type == ScaffoldType.LUNG_ALVEOLI:
            LungAlveoliParams(**params)
            alveoli = params.get("alveoli_diameter", 200)
            if alveoli < 150:
                warnings.append("Alveoli diameter < 150um may be difficult to print")

        elif scaffold_type == ScaffoldType.PANCREATIC_ISLET:
            PancreaticIsletParams(**params)

        elif scaffold_type == ScaffoldType.LIVER_SINUSOID:
            LiverSinusoidParams(**params)
            sinusoid = params.get("sinusoid_diameter", 20)
            if sinusoid < 15:
                warnings.append("Sinusoid diameter < 15um may be difficult to print")

        # -------------------------------------------------------------------------
        # Skeletal (7 types)
        # -------------------------------------------------------------------------
        elif scaffold_type == ScaffoldType.TRABECULAR_BONE:
            TrabecularBoneParams(**params)
            strut = params.get("strut_thickness_um", 200)
            if strut < 100:
                warnings.append("Strut thickness < 100um may be difficult to print")

        elif scaffold_type == ScaffoldType.OSTEOCHONDRAL:
            OsteochondralParams(**params)

        elif scaffold_type == ScaffoldType.ARTICULAR_CARTILAGE:
            ArticularCartilageParams(**params)

        elif scaffold_type == ScaffoldType.MENISCUS:
            MeniscusParams(**params)
            fiber = params.get("fiber_diameter", 0.2)
            if fiber < 0.15:
                warnings.append("Fiber diameter < 0.15mm may be difficult to print")

        elif scaffold_type == ScaffoldType.TENDON_LIGAMENT:
            TendonLigamentParams(**params)
            fiber = params.get("fiber_diameter", 0.15)
            if fiber < 0.1:
                warnings.append("Fiber diameter < 0.1mm may be difficult to print")

        elif scaffold_type == ScaffoldType.INTERVERTEBRAL_DISC:
            IntervertebralDiscParams(**params)

        elif scaffold_type == ScaffoldType.HAVERSIAN_BONE:
            HaversianBoneParams(**params)
            canal = params.get("canal_diameter_um", 50)
            if canal < 30:
                warnings.append("Canal diameter < 30um may be difficult to print")

        # -------------------------------------------------------------------------
        # Soft Tissue (4 types)
        # -------------------------------------------------------------------------
        elif scaffold_type == ScaffoldType.MULTILAYER_SKIN:
            MultilayerSkinParams(**params)
            epidermis = params.get("epidermis_thickness_mm", 0.15)
            if epidermis < 0.1:
                warnings.append("Epidermis thickness < 0.1mm may be difficult to print")

        elif scaffold_type == ScaffoldType.SKELETAL_MUSCLE:
            SkeletalMuscleParams(**params)

        elif scaffold_type == ScaffoldType.CORNEA:
            CorneaParams(**params)
            thickness = params.get("thickness_mm", 0.55)
            if thickness < 0.4:
                warnings.append("Cornea thickness < 0.4mm may be fragile")

        elif scaffold_type == ScaffoldType.ADIPOSE:
            AdiposeParams(**params)

        # -------------------------------------------------------------------------
        # Dental (3 types)
        # -------------------------------------------------------------------------
        elif scaffold_type == ScaffoldType.DENTIN_PULP:
            DentinPulpParams(**params)

        elif scaffold_type == ScaffoldType.EAR_AURICLE:
            EarAuricleParams(**params)
            thickness = params.get("thickness", 2.0)
            if thickness < 1.5:
                warnings.append("Cartilage thickness < 1.5mm may be too thin")

        elif scaffold_type == ScaffoldType.NASAL_SEPTUM:
            NasalSeptumParams(**params)

        # -------------------------------------------------------------------------
        # Lattice (6 types: basic, gyroid, schwarz_p, octet_truss, voronoi, honeycomb)
        # -------------------------------------------------------------------------
        elif scaffold_type == ScaffoldType.LATTICE:
            LatticeParams(**params)
            strut = params.get("strut_diameter_mm", 0.5)
            if strut < 0.3:
                warnings.append("Strut diameter < 0.3mm may be difficult to print")

        elif scaffold_type == ScaffoldType.GYROID:
            GyroidParams(**params)
            wall = params.get("wall_thickness_mm", 0.3)
            if wall < 0.2:
                warnings.append("Wall thickness < 0.2mm may be difficult to print")

        elif scaffold_type == ScaffoldType.SCHWARZ_P:
            SchwarzPParams(**params)
            wall = params.get("wall_thickness_mm", 0.3)
            if wall < 0.2:
                warnings.append("Wall thickness < 0.2mm may be difficult to print")

        elif scaffold_type == ScaffoldType.OCTET_TRUSS:
            OctetTrussParams(**params)
            strut = params.get("strut_diameter_mm", 0.3)
            if strut < 0.2:
                warnings.append("Strut diameter < 0.2mm may be difficult to print")

        elif scaffold_type == ScaffoldType.VORONOI:
            VoronoiParams(**params)
            strut = params.get("strut_diameter_mm", 0.3)
            if strut < 0.2:
                warnings.append("Strut diameter < 0.2mm may be difficult to print")

        elif scaffold_type == ScaffoldType.HONEYCOMB:
            HoneycombParams(**params)
            wall = params.get("wall_thickness_mm", 0.3)
            if wall < 0.2:
                warnings.append("Wall thickness < 0.2mm may be difficult to print")

        # -------------------------------------------------------------------------
        # Microfluidic (3 types)
        # -------------------------------------------------------------------------
        elif scaffold_type == ScaffoldType.ORGAN_ON_CHIP:
            OrganOnChipParams(**params)
            channel = params.get("channel_width_mm", 0.3)
            if channel < 0.2:
                warnings.append("Channel width < 0.2mm may be difficult to print")

        elif scaffold_type == ScaffoldType.GRADIENT_SCAFFOLD:
            GradientScaffoldParams(**params)

        elif scaffold_type == ScaffoldType.PERFUSABLE_NETWORK:
            PerfusableNetworkParams(**params)

        # -------------------------------------------------------------------------
        # Advanced Vascular (2 types: space_colonization, bifurcating_tree)
        # -------------------------------------------------------------------------
        elif scaffold_type == ScaffoldType.SPACE_COLONIZATION:
            SpaceColonizationParams(**params)
            if params.get("num_attractors", 50000) > 200000:
                warnings.append("More than 200k attractors may be very slow (>60s generation time)")
            if params.get("max_iterations", 300) > 1000:
                warnings.append("More than 1000 iterations may be very slow")

        elif scaffold_type == ScaffoldType.BIFURCATING_TREE:
            BifurcatingTreeParams(**params)
            if params.get("branching_levels", 5) > 8:
                warnings.append("More than 8 levels may create very dense geometry")

        else:
            errors.append(f"Unknown scaffold type: {scaffold_type}")

    except Exception as e:
        errors.append(str(e))

    return len(errors) == 0, errors, warnings


def _cache_scaffold(scaffold_id: str, manifold, stl_bytes: bytes, metadata: Dict[str, Any]) -> None:
    """Cache a generated scaffold."""
    global _scaffold_cache

    # Simple LRU-like eviction
    if len(_scaffold_cache) >= _CACHE_MAX_SIZE:
        # Remove oldest entry
        oldest_key = next(iter(_scaffold_cache))
        del _scaffold_cache[oldest_key]

    _scaffold_cache[scaffold_id] = (manifold, stl_bytes, metadata)


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/generate", response_model=GenerateResponse)
async def generate_scaffold(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a scaffold from parameters.

    Full generation including boolean operations and STL export.
    Timeout is configurable via GENERATION_TIMEOUT_SECONDS env var (default: 60s, must be multiple of 30).
    """
    settings = get_settings()
    timeout_seconds = settings.generation_timeout_seconds

    if not check_manifold_available():
        logger.error("manifold3d library not available")
        raise HTTPException(
            status_code=503,
            detail="manifold3d library not available. Install with: pip install manifold3d",
        )

    logger.info(f"Generating scaffold: type={request.type}, preview_only={request.preview_only}, invert={request.invert}, timeout={timeout_seconds}s")
    start_time = time.time()

    try:
        # Generate the scaffold with timeout
        # Run synchronous generation in thread pool with asyncio timeout
        manifold, gen_stats = await asyncio.wait_for(
            asyncio.to_thread(
                _generate_scaffold,
                request.type,
                request.params,
                request.preview_only,
            ),
            timeout=timeout_seconds,
        )

        # Apply inversion if requested (swap solid/void spaces)
        if request.invert:
            logger.info("Applying geometry inversion...")
            manifold = _invert_manifold(manifold, padding_mm=1.0)

        generation_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Scaffold generated successfully in {generation_time_ms:.2f}ms")

        # Convert to mesh data
        mesh_dict = manifold_to_mesh_dict(manifold)

        # Get bounding box
        bbox_min, bbox_max = get_bounding_box(manifold)

        # Generate STL and encode
        stl_bytes = manifold_to_stl_binary(manifold)
        stl_base64 = stl_to_base64(stl_bytes)

        # Generate scaffold ID and cache
        scaffold_id = str(uuid.uuid4())
        _cache_scaffold(
            scaffold_id,
            manifold,
            stl_bytes,
            {
                "type": request.type,
                "params": request.params,
                "stats": gen_stats,
                "inverted": request.invert,
            },
        )

        return GenerateResponse(
            success=True,
            scaffold_id=scaffold_id,
            mesh=MeshResponse(
                vertices=mesh_dict["vertices"],
                indices=mesh_dict["indices"],
                normals=mesh_dict["normals"],
            ),
            stl_base64=stl_base64,
            stats=StatsResponse(
                triangle_count=mesh_dict["triangle_count"],
                volume_mm3=gen_stats.get("volume_mm3", 0.0),
                generation_time_ms=generation_time_ms,
            ),
            bounding_box={
                "min": list(bbox_min),
                "max": list(bbox_max),
            },
            inverted=request.invert,
        )

    except asyncio.TimeoutError:
        elapsed = time.time() - start_time
        logger.error(f"Scaffold generation timed out after {elapsed:.1f}s (limit: {timeout_seconds}s)")
        raise HTTPException(
            status_code=408,
            detail=f"Generation timed out after {timeout_seconds} seconds. Try reducing resolution or complexity.",
        )
    except ValueError as e:
        logger.error(f"Invalid parameters for scaffold generation: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Scaffold generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/preview", response_model=GenerateResponse)
async def preview_scaffold(request: GenerateRequest) -> GenerateResponse:
    """
    Fast scaffold preview.

    Same as /generate but optimized for speed (may skip some boolean operations).
    """
    # Force preview mode
    request.preview_only = True
    return await generate_scaffold(request)


@router.post("/validate", response_model=ValidateResponse)
async def validate_params(request: ValidateRequest) -> ValidateResponse:
    """
    Validate scaffold parameters without generating.

    Returns validation errors and warnings.
    """
    is_valid, errors, warnings = _validate_params(request.type, request.params)

    return ValidateResponse(
        valid=is_valid,
        errors=errors,
        warnings=warnings,
    )


@router.get("/export/{scaffold_id}")
async def export_scaffold(
    scaffold_id: str,
    format: str = Query(default="binary", description="STL format: binary or ascii"),
) -> Response:
    """
    Download a previously generated scaffold as STL.

    Query params:
        format: 'binary' (default) or 'ascii'
    """
    if scaffold_id not in _scaffold_cache:
        logger.warning(f"Scaffold not found in cache: {scaffold_id}")
        raise HTTPException(status_code=404, detail="Scaffold not found. Generate it first.")

    logger.info(f"Exporting scaffold {scaffold_id} as {format}")
    manifold, stl_bytes, metadata = _scaffold_cache[scaffold_id]

    if format == "ascii":
        stl_content = manifold_to_stl_ascii(manifold)
        return Response(
            content=stl_content,
            media_type="text/plain",
            headers={
                "Content-Disposition": f'attachment; filename="scaffold_{scaffold_id[:8]}.stl"',
            },
        )
    else:
        return Response(
            content=stl_bytes,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="scaffold_{scaffold_id[:8]}.stl"',
            },
        )


@router.get("/presets", response_model=PresetsResponse)
async def list_presets(
    category: Optional[str] = Query(default=None, description="Filter by category")
) -> PresetsResponse:
    """
    List available scaffold presets.

    Presets are predefined parameter combinations for common use cases.
    Optionally filter by category.
    """
    if category:
        filtered = [p for p in PRESETS if p.category == category]
        return PresetsResponse(presets=filtered)
    return PresetsResponse(presets=PRESETS)


@router.get("/presets/{preset_id}")
async def get_preset(preset_id: str) -> PresetInfo:
    """
    Get a specific preset by ID.
    """
    for preset in PRESETS:
        if preset.id == preset_id:
            return preset

    raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")


@router.get("/categories")
async def list_categories() -> Dict[str, List[str]]:
    """
    List all available scaffold categories and their types.
    """
    categories = {}
    for preset in PRESETS:
        if preset.category not in categories:
            categories[preset.category] = []
        if preset.type.value not in categories[preset.category]:
            categories[preset.category].append(preset.type.value)
    return categories


@router.get("/types")
async def list_scaffold_types() -> List[Dict[str, str]]:
    """
    List all supported scaffold types with descriptions.
    """
    type_descriptions = {
        # Legacy (3)
        "vascular_network": "Branching vascular network following Murray's law",
        "porous_disc": "Cylindrical disc with patterned pores",
        "primitive": "Basic geometric shapes with modifications",
        # Tubular (7)
        "tubular_conduit": "Hollow tube with optional surface texturing",
        "vascular_perfusion_dish": "Vascular perfusion dish with collision-aware branching",
        "blood_vessel": "Multi-layer vascular graft",
        "nerve_conduit": "Multi-channel nerve guidance conduit",
        "spinal_cord": "Spinal cord guidance scaffold",
        "bladder": "Dome-shaped bladder scaffold",
        "trachea": "C-shaped cartilage ring structure",
        # Organ (6)
        "hepatic_lobule": "Hexagonal liver lobule units",
        "cardiac_patch": "Aligned microfiber cardiac scaffold",
        "kidney_tubule": "Convoluted tubule structure",
        "lung_alveoli": "Branching airways with alveolar sacs",
        "pancreatic_islet": "Spherical islet cluster",
        "liver_sinusoid": "Fenestrated tubular channels",
        # Skeletal (7)
        "trabecular_bone": "Porous trabecular bone architecture",
        "osteochondral": "Bone-cartilage gradient scaffold",
        "articular_cartilage": "Zonal cartilage with gradient porosity",
        "meniscus": "Wedge-shaped fibrocartilage scaffold",
        "tendon_ligament": "Aligned crimped fiber structure",
        "intervertebral_disc": "Annulus fibrosus with nucleus pulposus",
        "haversian_bone": "Cortical bone with Haversian canals",
        # Soft Tissue (4)
        "multilayer_skin": "3-layer skin with vascular channels",
        "skeletal_muscle": "Aligned myofiber channels",
        "cornea": "Curved lamellar cornea structure",
        "adipose": "Honeycomb adipocyte housing",
        # Dental (3)
        "dentin_pulp": "Tooth scaffold with pulp chamber",
        "ear_auricle": "Ear cartilage framework",
        "nasal_septum": "Curved nasal cartilage sheet",
        # Lattice (6)
        "lattice": "Periodic lattice structure (cubic or BCC)",
        "gyroid": "Gyroid TPMS with excellent permeability",
        "schwarz_p": "Schwarz P TPMS with cubic symmetry",
        "octet_truss": "High strength-to-weight lattice",
        "voronoi": "Organic randomized cellular structure",
        "honeycomb": "Hexagonal honeycomb for load-bearing",
        # Microfluidic (3)
        "organ_on_chip": "Microfluidic chip with tissue chambers",
        "gradient_scaffold": "Continuous porosity gradient",
        "perfusable_network": "Murray's law vascular network",
    }

    return [
        {"type": t.value, "description": type_descriptions.get(t.value, "")}
        for t in ScaffoldType
    ]


# ============================================================================
# Primitives API Endpoints
# ============================================================================


class PrimitiveSchemaInfo(BaseModel):
    """Schema information for a primitive."""
    name: str = Field(description="Primitive name")
    schema: Dict[str, Any] = Field(description="Parameter schema with type/min/max/default")
    defaults: Dict[str, Any] = Field(description="Default parameter values")
    category: str = Field(description="Primitive category (geometric, architectural, organic)")


class PrimitivesListResponse(BaseModel):
    """Response listing available primitives."""
    primitives: List[str] = Field(description="List of primitive names")
    count: int = Field(description="Total number of primitives")


class PrimitivesSchemaResponse(BaseModel):
    """Response with all primitive schemas."""
    schemas: Dict[str, PrimitiveSchemaInfo] = Field(description="Schemas indexed by primitive name")


@router.get("/primitives/list", response_model=PrimitivesListResponse)
async def list_primitives_endpoint() -> PrimitivesListResponse:
    """
    List all available primitives.

    Returns the names of all registered primitives that can be used
    for scaffold generation.
    """
    from app.geometry.primitives import list_primitives
    primitives = list_primitives()
    return PrimitivesListResponse(primitives=primitives, count=len(primitives))


@router.get("/primitives/schema", response_model=PrimitivesSchemaResponse)
async def get_primitives_schema() -> PrimitivesSchemaResponse:
    """
    Get schemas for all primitives.

    Returns parameter schemas including type, min, max, and default values
    for each primitive. This can be used to build dynamic UIs.
    """
    from app.geometry.primitives import get_all_schemas, get_defaults, get_primitive

    schemas = get_all_schemas()

    # Categorize primitives
    geometric = {'cylinder', 'sphere', 'box', 'cone', 'torus', 'capsule', 'pyramid',
                 'wedge', 'prism', 'tube', 'ellipsoid', 'hemisphere'}
    architectural = {'fillet', 'chamfer', 'slot', 'counterbore', 'countersink', 'boss', 'rib'}
    organic = {'branch', 'bifurcation', 'pore', 'channel', 'fiber', 'membrane',
               'lattice_cell', 'pore_array'}

    def get_category(name: str) -> str:
        if name in geometric:
            return 'geometric'
        elif name in architectural:
            return 'architectural'
        elif name in organic:
            return 'organic'
        return 'other'

    result = {}
    for name, schema in schemas.items():
        try:
            defaults = get_defaults(name)
        except:
            defaults = {}

        result[name] = PrimitiveSchemaInfo(
            name=name,
            schema=schema,
            defaults=defaults,
            category=get_category(name)
        )

    return PrimitivesSchemaResponse(schemas=result)


@router.get("/primitives/schema/{primitive_name}")
async def get_primitive_schema(primitive_name: str) -> PrimitiveSchemaInfo:
    """
    Get schema for a specific primitive.

    Args:
        primitive_name: Name of the primitive (e.g., 'torus', 'capsule')

    Returns:
        Schema with parameter definitions and defaults
    """
    from app.geometry.primitives import get_schema, get_defaults

    try:
        schema = get_schema(primitive_name)
        defaults = get_defaults(primitive_name)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Primitive '{primitive_name}' not found")

    # Categorize
    geometric = {'cylinder', 'sphere', 'box', 'cone', 'torus', 'capsule', 'pyramid',
                 'wedge', 'prism', 'tube', 'ellipsoid', 'hemisphere'}
    architectural = {'fillet', 'chamfer', 'slot', 'counterbore', 'countersink', 'boss', 'rib'}
    organic = {'branch', 'bifurcation', 'pore', 'channel', 'fiber', 'membrane',
               'lattice_cell', 'pore_array'}

    if primitive_name in geometric:
        category = 'geometric'
    elif primitive_name in architectural:
        category = 'architectural'
    elif primitive_name in organic:
        category = 'organic'
    else:
        category = 'other'

    return PrimitiveSchemaInfo(
        name=primitive_name,
        schema=schema,
        defaults=defaults,
        category=category
    )
