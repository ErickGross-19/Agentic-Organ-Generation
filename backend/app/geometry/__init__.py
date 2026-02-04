"""
Geometry module for scaffold generation.

Provides generators for various scaffold geometries:

Original Types:
- Vascular networks (branching tree structures)
- Primitive shapes (cylinder, sphere, box, cone)
- Tubular conduits (hollow tubes)
- Porous discs (cylindrical scaffolds with pore patterns)
- Lattice structures (cubic and BCC)

Skeletal Tissue (7 types):
- Trabecular bone (cancellous/spongy bone)
- Osteochondral (bone-cartilage gradient)
- Articular cartilage (zonal architecture)
- Meniscus (fibrocartilage with wedge shape)
- Tendon/ligament (aligned crimped fibers)
- Intervertebral disc (annulus fibrosus + nucleus pulposus)
- Haversian bone (cortical bone with osteons)

Organ-Specific (6 types):
- Hepatic lobule (hexagonal liver lobule units)
- Cardiac patch (aligned microfibers)
- Kidney tubule (convoluted tubules)
- Lung alveoli (branching airways)
- Pancreatic islet (spherical clusters)
- Liver sinusoid (fenestrated channels)

Soft Tissue (4 types):
- Multilayer skin (epidermis/dermis/hypodermis)
- Skeletal muscle (aligned myofibers)
- Cornea (curved dome with lamellae)
- Adipose (honeycomb structure)

Tubular Organs (6 types):
- Blood vessel (multi-layer vascular graft)
- Nerve conduit (multi-channel guidance)
- Spinal cord (gray/white matter zones)
- Bladder (dome-shaped)
- Trachea (C-shaped cartilage rings)
- Vascular perfusion dish (collision-aware branching network)

Dental/Craniofacial (3 types):
- Dentin-pulp (tooth scaffold)
- Ear auricle (auricular framework)
- Nasal septum (curved cartilage)

Advanced Lattice (5 types):
- Gyroid (TPMS)
- Schwarz P (TPMS)
- Octet truss (high strength-to-weight)
- Voronoi (organic cellular)
- Honeycomb (hexagonal cells)

Microfluidic (3 types):
- Organ-on-chip (microfluidic chambers)
- Gradient scaffold (porosity gradient)
- Perfusable network (Murray's law vascular)
"""

# Core utilities
from .core import (
    batch_union,
    tree_union,
    tree_union_parallel,
    union_pair,
    check_manifold_available,
    get_manifold_module,
)

# Original generators
from .vascular import (
    VascularParams,
    make_cyl,
    generate_vascular_network,
    generate_vascular_network_from_dict,
)
from .vascular_perfusion_dish import (
    VascularPerfusionDishParams,
    generate_vascular_perfusion_dish,
    generate_vascular_perfusion_dish_from_dict,
)
from .primitives import (
    PrimitiveParams,
    create_cylinder,
    create_sphere,
    create_box,
    create_cone,
    apply_hole,
    apply_shell,
    generate_primitive,
    generate_primitive_from_dict,
    SHAPE_CREATORS,
    MODIFICATIONS,
)

# New primitives module exports (registry-based)
from .primitives import (
    get_primitive,
    list_primitives,
    get_schema,
    get_all_schemas,
    evaluate_csg_tree,
    apply_transforms,
    PRIMITIVES,
)
from .tubular import (
    TubularConduitParams,
    generate_tubular_conduit,
    generate_tubular_conduit_from_dict,
)
from .porous_disc import (
    PorousDiscParams,
    generate_porous_disc,
    generate_porous_disc_from_dict,
    generate_hexagonal_positions,
    generate_grid_positions,
)
from .lattice import (
    LatticeParams,
    make_strut,
    get_cubic_struts,
    get_bcc_struts,
    generate_lattice,
    generate_lattice_from_dict,
)

# Tubular organ generators
from .tubular.blood_vessel import (
    BloodVesselParams,
    generate_blood_vessel,
    generate_blood_vessel_from_dict,
)
from .tubular.nerve_conduit import (
    NerveConduitParams,
    generate_nerve_conduit,
    generate_nerve_conduit_from_dict,
)
from .tubular.spinal_cord import (
    SpinalCordParams,
    generate_spinal_cord,
    generate_spinal_cord_from_dict,
)
from .tubular.bladder import (
    BladderParams,
    generate_bladder,
    generate_bladder_from_dict,
)
from .tubular.trachea import (
    TracheaParams,
    generate_trachea,
    generate_trachea_from_dict,
)

# Advanced lattice generators (TPMS and strut-based)
from .lattice import (
    GyroidParams,
    generate_gyroid,
    generate_gyroid_from_dict,
    SchwarzPParams,
    generate_schwarz_p,
    generate_schwarz_p_from_dict,
    OctetTrussParams,
    generate_octet_truss,
    generate_octet_truss_from_dict,
    VoronoiParams,
    generate_voronoi,
    generate_voronoi_from_dict,
    HoneycombParams,
    generate_honeycomb,
    generate_honeycomb_from_dict,
)

# Skeletal tissue generators
from .skeletal import (
    TrabecularBoneParams,
    generate_trabecular_bone,
    generate_trabecular_bone_from_dict,
    OsteochondralParams,
    generate_osteochondral,
    generate_osteochondral_from_dict,
    ArticularCartilageParams,
    generate_articular_cartilage,
    generate_articular_cartilage_from_dict,
    MeniscusParams,
    generate_meniscus,
    generate_meniscus_from_dict,
    TendonLigamentParams,
    generate_tendon_ligament,
    generate_tendon_ligament_from_dict,
    IntervertebralDiscParams,
    generate_intervertebral_disc,
    generate_intervertebral_disc_from_dict,
    HaversianBoneParams,
    generate_haversian_bone,
    generate_haversian_bone_from_dict,
)

# Organ-specific generators
from .organ import (
    HepaticLobuleParams,
    generate_hepatic_lobule,
    generate_hepatic_lobule_from_dict,
    CardiacPatchParams,
    generate_cardiac_patch,
    generate_cardiac_patch_from_dict,
    KidneyTubuleParams,
    generate_kidney_tubule,
    generate_kidney_tubule_from_dict,
    LungAlveoliParams,
    generate_lung_alveoli,
    generate_lung_alveoli_from_dict,
    PancreaticIsletParams,
    generate_pancreatic_islet,
    generate_pancreatic_islet_from_dict,
    LiverSinusoidParams,
    generate_liver_sinusoid,
    generate_liver_sinusoid_from_dict,
)

# Soft tissue generators
from .soft_tissue import (
    MultilayerSkinParams,
    generate_multilayer_skin,
    generate_multilayer_skin_from_dict,
    SkeletalMuscleParams,
    generate_skeletal_muscle,
    generate_skeletal_muscle_from_dict,
    CorneaParams,
    generate_cornea,
    generate_cornea_from_dict,
    AdiposeTissueParams,
    generate_adipose_tissue,
    generate_adipose_tissue_from_dict,
)

# Dental/craniofacial generators
from .dental import (
    DentinPulpParams,
    generate_dentin_pulp,
    generate_dentin_pulp_from_dict,
    EarAuricleParams,
    generate_ear_auricle,
    generate_ear_auricle_from_dict,
    NasalSeptumParams,
    generate_nasal_septum,
    generate_nasal_septum_from_dict,
)

# Microfluidic generators
from .microfluidic import (
    OrganOnChipParams,
    generate_organ_on_chip,
    generate_organ_on_chip_from_dict,
    GradientScaffoldParams,
    generate_gradient_scaffold,
    generate_gradient_scaffold_from_dict,
    PerfusableNetworkParams,
    generate_perfusable_network,
    generate_perfusable_network_from_dict,
)

__all__ = [
    # Core utilities
    "batch_union",
    "tree_union",
    "tree_union_parallel",
    "union_pair",
    "check_manifold_available",
    "get_manifold_module",
    # Vascular network
    "VascularParams",
    "make_cyl",
    "generate_vascular_network",
    "generate_vascular_network_from_dict",
    # Vascular perfusion dish
    "VascularPerfusionDishParams",
    "generate_vascular_perfusion_dish",
    "generate_vascular_perfusion_dish_from_dict",
    # Primitives
    "PrimitiveParams",
    "create_cylinder",
    "create_sphere",
    "create_box",
    "create_cone",
    "apply_hole",
    "apply_shell",
    "generate_primitive",
    "generate_primitive_from_dict",
    "SHAPE_CREATORS",
    "MODIFICATIONS",
    # New primitives module (registry-based)
    "get_primitive",
    "list_primitives",
    "get_schema",
    "get_all_schemas",
    "evaluate_csg_tree",
    "apply_transforms",
    "PRIMITIVES",
    # Tubular conduit
    "TubularConduitParams",
    "generate_tubular_conduit",
    "generate_tubular_conduit_from_dict",
    # Porous disc
    "PorousDiscParams",
    "generate_porous_disc",
    "generate_porous_disc_from_dict",
    "generate_hexagonal_positions",
    "generate_grid_positions",
    # Lattice structures
    "LatticeParams",
    "make_strut",
    "get_cubic_struts",
    "get_bcc_struts",
    "generate_lattice",
    "generate_lattice_from_dict",
    # Tubular organ generators
    "BloodVesselParams",
    "generate_blood_vessel",
    "generate_blood_vessel_from_dict",
    "NerveConduitParams",
    "generate_nerve_conduit",
    "generate_nerve_conduit_from_dict",
    "SpinalCordParams",
    "generate_spinal_cord",
    "generate_spinal_cord_from_dict",
    "BladderParams",
    "generate_bladder",
    "generate_bladder_from_dict",
    "TracheaParams",
    "generate_trachea",
    "generate_trachea_from_dict",
    # Advanced lattice (TPMS)
    "GyroidParams",
    "generate_gyroid",
    "generate_gyroid_from_dict",
    "SchwarzPParams",
    "generate_schwarz_p",
    "generate_schwarz_p_from_dict",
    "OctetTrussParams",
    "generate_octet_truss",
    "generate_octet_truss_from_dict",
    "VoronoiParams",
    "generate_voronoi",
    "generate_voronoi_from_dict",
    "HoneycombParams",
    "generate_honeycomb",
    "generate_honeycomb_from_dict",
    # Skeletal tissue
    "TrabecularBoneParams",
    "generate_trabecular_bone",
    "generate_trabecular_bone_from_dict",
    "OsteochondralParams",
    "generate_osteochondral",
    "generate_osteochondral_from_dict",
    "ArticularCartilageParams",
    "generate_articular_cartilage",
    "generate_articular_cartilage_from_dict",
    "MeniscusParams",
    "generate_meniscus",
    "generate_meniscus_from_dict",
    "TendonLigamentParams",
    "generate_tendon_ligament",
    "generate_tendon_ligament_from_dict",
    "IntervertebralDiscParams",
    "generate_intervertebral_disc",
    "generate_intervertebral_disc_from_dict",
    "HaversianBoneParams",
    "generate_haversian_bone",
    "generate_haversian_bone_from_dict",
    # Organ-specific
    "HepaticLobuleParams",
    "generate_hepatic_lobule",
    "generate_hepatic_lobule_from_dict",
    "CardiacPatchParams",
    "generate_cardiac_patch",
    "generate_cardiac_patch_from_dict",
    "KidneyTubuleParams",
    "generate_kidney_tubule",
    "generate_kidney_tubule_from_dict",
    "LungAlveoliParams",
    "generate_lung_alveoli",
    "generate_lung_alveoli_from_dict",
    "PancreaticIsletParams",
    "generate_pancreatic_islet",
    "generate_pancreatic_islet_from_dict",
    "LiverSinusoidParams",
    "generate_liver_sinusoid",
    "generate_liver_sinusoid_from_dict",
    # Soft tissue
    "MultilayerSkinParams",
    "generate_multilayer_skin",
    "generate_multilayer_skin_from_dict",
    "SkeletalMuscleParams",
    "generate_skeletal_muscle",
    "generate_skeletal_muscle_from_dict",
    "CorneaParams",
    "generate_cornea",
    "generate_cornea_from_dict",
    "AdiposeTissueParams",
    "generate_adipose_tissue",
    "generate_adipose_tissue_from_dict",
    # Dental/craniofacial
    "DentinPulpParams",
    "generate_dentin_pulp",
    "generate_dentin_pulp_from_dict",
    "EarAuricleParams",
    "generate_ear_auricle",
    "generate_ear_auricle_from_dict",
    "NasalSeptumParams",
    "generate_nasal_septum",
    "generate_nasal_septum_from_dict",
    # Microfluidic
    "OrganOnChipParams",
    "generate_organ_on_chip",
    "generate_organ_on_chip_from_dict",
    "GradientScaffoldParams",
    "generate_gradient_scaffold",
    "generate_gradient_scaffold_from_dict",
    "PerfusableNetworkParams",
    "generate_perfusable_network",
    "generate_perfusable_network_from_dict",
]
