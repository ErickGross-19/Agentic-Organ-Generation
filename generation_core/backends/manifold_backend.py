"""
ManifoldBackend - Bridge MorphoStruct geometry generators into the DesignSpec pipeline.

Dispatches to 41 geometry generators from the MorphoStruct backend, converting
their output (manifold3d.Manifold or marching-cubes mesh wrappers) to
trimesh.Trimesh objects compatible with the DesignSpec pipeline.

UNIT CONVENTION
---------------
MorphoStruct generators work in millimeters internally.
The DesignSpec pipeline works in meters internally.
This backend handles the mm-to-m conversion automatically when
convert_units=True (the default).

GENERATOR CATEGORIES (41 total)
-------------------------------
Original (6):
    vascular_network, vascular_perfusion_dish, primitive,
    tubular_conduit, porous_disc, lattice

Advanced Lattice / TPMS (5):
    gyroid, schwarz_p, octet_truss, voronoi, honeycomb

Skeletal Tissue (7):
    trabecular_bone, osteochondral, articular_cartilage, meniscus,
    tendon_ligament, intervertebral_disc, haversian_bone

Organ-Specific (6):
    hepatic_lobule, cardiac_patch, kidney_tubule, lung_alveoli,
    pancreatic_islet, liver_sinusoid

Soft Tissue (4):
    multilayer_skin, skeletal_muscle, cornea, adipose_tissue

Tubular Organs (5):
    blood_vessel, nerve_conduit, spinal_cord, bladder, trachea

Dental / Craniofacial (3):
    dentin_pulp, ear_auricle, nasal_septum

Microfluidic (3):
    organ_on_chip, gradient_scaffold, perfusable_network

Vascular Backends (2):
    space_colonization, top_down_scaffold
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MM_TO_M = 0.001

GENERATOR_REGISTRY: Dict[str, Tuple[str, str]] = {
    "vascular_network": (
        "app.geometry.vascular",
        "generate_vascular_network_from_dict",
    ),
    "vascular_perfusion_dish": (
        "app.geometry.vascular_perfusion_dish",
        "generate_vascular_perfusion_dish_from_dict",
    ),
    "primitive": (
        "app.geometry.primitives",
        "generate_primitive_from_dict",
    ),
    "tubular_conduit": (
        "app.geometry.tubular",
        "generate_tubular_conduit_from_dict",
    ),
    "porous_disc": (
        "app.geometry.porous_disc",
        "generate_porous_disc_from_dict",
    ),
    "lattice": (
        "app.geometry.lattice.basic",
        "generate_lattice_from_dict",
    ),
    "gyroid": (
        "app.geometry.lattice.gyroid",
        "generate_gyroid_from_dict",
    ),
    "schwarz_p": (
        "app.geometry.lattice.schwarz_p",
        "generate_schwarz_p_from_dict",
    ),
    "octet_truss": (
        "app.geometry.lattice.octet_truss",
        "generate_octet_truss_from_dict",
    ),
    "voronoi": (
        "app.geometry.lattice.voronoi",
        "generate_voronoi_from_dict",
    ),
    "honeycomb": (
        "app.geometry.lattice.honeycomb",
        "generate_honeycomb_from_dict",
    ),
    "trabecular_bone": (
        "app.geometry.skeletal.trabecular_bone",
        "generate_trabecular_bone_from_dict",
    ),
    "osteochondral": (
        "app.geometry.skeletal.osteochondral",
        "generate_osteochondral_from_dict",
    ),
    "articular_cartilage": (
        "app.geometry.skeletal.articular_cartilage",
        "generate_articular_cartilage_from_dict",
    ),
    "meniscus": (
        "app.geometry.skeletal.meniscus",
        "generate_meniscus_from_dict",
    ),
    "tendon_ligament": (
        "app.geometry.skeletal.tendon_ligament",
        "generate_tendon_ligament_from_dict",
    ),
    "intervertebral_disc": (
        "app.geometry.skeletal.intervertebral_disc",
        "generate_intervertebral_disc_from_dict",
    ),
    "haversian_bone": (
        "app.geometry.skeletal.haversian_bone",
        "generate_haversian_bone_from_dict",
    ),
    "hepatic_lobule": (
        "app.geometry.organ.hepatic_lobule",
        "generate_hepatic_lobule_from_dict",
    ),
    "cardiac_patch": (
        "app.geometry.organ.cardiac_patch",
        "generate_cardiac_patch_from_dict",
    ),
    "kidney_tubule": (
        "app.geometry.organ.kidney_tubule",
        "generate_kidney_tubule_from_dict",
    ),
    "lung_alveoli": (
        "app.geometry.organ.lung_alveoli",
        "generate_lung_alveoli_from_dict",
    ),
    "pancreatic_islet": (
        "app.geometry.organ.pancreatic_islet",
        "generate_pancreatic_islet_from_dict",
    ),
    "liver_sinusoid": (
        "app.geometry.organ.liver_sinusoid",
        "generate_liver_sinusoid_from_dict",
    ),
    "multilayer_skin": (
        "app.geometry.soft_tissue.multilayer_skin",
        "generate_multilayer_skin_from_dict",
    ),
    "skeletal_muscle": (
        "app.geometry.soft_tissue.skeletal_muscle",
        "generate_skeletal_muscle_from_dict",
    ),
    "cornea": (
        "app.geometry.soft_tissue.cornea",
        "generate_cornea_from_dict",
    ),
    "adipose_tissue": (
        "app.geometry.soft_tissue.adipose",
        "generate_adipose_tissue_from_dict",
    ),
    "blood_vessel": (
        "app.geometry.tubular.blood_vessel",
        "generate_blood_vessel_from_dict",
    ),
    "nerve_conduit": (
        "app.geometry.tubular.nerve_conduit",
        "generate_nerve_conduit_from_dict",
    ),
    "spinal_cord": (
        "app.geometry.tubular.spinal_cord",
        "generate_spinal_cord_from_dict",
    ),
    "bladder": (
        "app.geometry.tubular.bladder",
        "generate_bladder_from_dict",
    ),
    "trachea": (
        "app.geometry.tubular.trachea",
        "generate_trachea_from_dict",
    ),
    "dentin_pulp": (
        "app.geometry.dental.dentin_pulp",
        "generate_dentin_pulp_from_dict",
    ),
    "ear_auricle": (
        "app.geometry.dental.ear_auricle",
        "generate_ear_auricle_from_dict",
    ),
    "nasal_septum": (
        "app.geometry.dental.nasal_septum",
        "generate_nasal_septum_from_dict",
    ),
    "organ_on_chip": (
        "app.geometry.microfluidic.organ_on_chip",
        "generate_organ_on_chip_from_dict",
    ),
    "gradient_scaffold": (
        "app.geometry.microfluidic.gradient_scaffold",
        "generate_gradient_scaffold_from_dict",
    ),
    "perfusable_network": (
        "app.geometry.microfluidic.perfusable_network",
        "generate_perfusable_network_from_dict",
    ),
}

GENERATOR_CATEGORIES: Dict[str, List[str]] = {
    "original": [
        "vascular_network",
        "vascular_perfusion_dish",
        "primitive",
        "tubular_conduit",
        "porous_disc",
        "lattice",
    ],
    "lattice": [
        "gyroid",
        "schwarz_p",
        "octet_truss",
        "voronoi",
        "honeycomb",
    ],
    "skeletal": [
        "trabecular_bone",
        "osteochondral",
        "articular_cartilage",
        "meniscus",
        "tendon_ligament",
        "intervertebral_disc",
        "haversian_bone",
    ],
    "organ": [
        "hepatic_lobule",
        "cardiac_patch",
        "kidney_tubule",
        "lung_alveoli",
        "pancreatic_islet",
        "liver_sinusoid",
    ],
    "soft_tissue": [
        "multilayer_skin",
        "skeletal_muscle",
        "cornea",
        "adipose_tissue",
    ],
    "tubular": [
        "blood_vessel",
        "nerve_conduit",
        "spinal_cord",
        "bladder",
        "trachea",
    ],
    "dental": [
        "dentin_pulp",
        "ear_auricle",
        "nasal_septum",
    ],
    "microfluidic": [
        "organ_on_chip",
        "gradient_scaffold",
        "perfusable_network",
    ],
}


def import_generator_function(module_path: str, function_name: str):
    """
    Import a generator function by module path and function name.

    Tries the ``app.geometry`` path first (backend server context) and falls
    back to ``backend.app.geometry`` (monorepo root context).

    Parameters
    ----------
    module_path : str
        Dotted module path starting with ``app.geometry``.
    function_name : str
        Name of the ``generate_*_from_dict`` function to import.

    Returns
    -------
    callable
        The imported generator function.

    Raises
    ------
    ImportError
        If the function cannot be imported from either path.
    """
    for prefix in ("", "backend."):
        try:
            mod = importlib.import_module(f"{prefix}{module_path}")
            return getattr(mod, function_name)
        except (ImportError, AttributeError):
            continue

    raise ImportError(
        f"Could not import '{function_name}' from '{module_path}'. "
        f"Ensure manifold3d is installed and the geometry package is importable."
    )


def manifold_to_trimesh(manifold_result, convert_units: bool = True):
    """
    Convert a manifold3d.Manifold or marching-cubes mesh wrapper to trimesh.

    Both manifold3d.Manifold and the ``_MarchingCubesMeshWrapper`` used by
    TPMS generators expose a ``.to_mesh()`` method returning an object with
    ``vert_properties`` (Nx3+ float array) and ``tri_verts`` (Mx3 int array).

    Parameters
    ----------
    manifold_result : manifold3d.Manifold or mesh wrapper
        Raw output from a MorphoStruct generator.
    convert_units : bool
        If True, multiply vertex coordinates by 0.001 (mm -> m).

    Returns
    -------
    trimesh.Trimesh
        Converted mesh ready for the DesignSpec pipeline.
    """
    import trimesh

    mesh_data = manifold_result.to_mesh()
    vertices = np.array(mesh_data.vert_properties, dtype=np.float64)[:, :3].copy()
    faces = np.array(mesh_data.tri_verts, dtype=np.int64).copy()

    if convert_units:
        vertices *= MM_TO_M

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


@dataclass
class ManifoldBackend:
    """
    Bridge MorphoStruct geometry generators into the DesignSpec pipeline.

    Maintains a registry of 41 generator types organised by category.
    Each generator is imported lazily when first needed, so the heavy
    ``manifold3d`` dependency is only required at generation time.
    """

    def generate(
        self,
        generator_type: str,
        params: Dict[str, Any],
        convert_units: bool = True,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Generate a scaffold mesh using a MorphoStruct generator.

        Parameters
        ----------
        generator_type : str
            Type of generator (e.g., ``"gyroid"``, ``"trabecular_bone"``).
        params : dict
            Generator parameters forwarded to the ``generate_*_from_dict``
            function.
        convert_units : bool
            If True, convert output vertices from mm to meters.

        Returns
        -------
        Tuple[trimesh.Trimesh, dict]
            The generated mesh and a statistics dictionary.

        Raises
        ------
        ValueError
            If *generator_type* is not in the registry.
        ImportError
            If the generator module cannot be loaded.
        """
        if generator_type not in GENERATOR_REGISTRY:
            available = ", ".join(sorted(GENERATOR_REGISTRY.keys()))
            raise ValueError(
                f"Unknown generator type: '{generator_type}'. "
                f"Available types: {available}"
            )

        module_path, function_name = GENERATOR_REGISTRY[generator_type]

        logger.info(
            f"  ManifoldBackend: generating '{generator_type}' "
            f"via {function_name}"
        )

        gen_func = import_generator_function(module_path, function_name)
        manifold_result, stats = gen_func(params)

        mesh = manifold_to_trimesh(manifold_result, convert_units=convert_units)

        stats["generator_type"] = generator_type
        stats["units_converted"] = convert_units
        stats["vertex_count"] = len(mesh.vertices)
        stats["face_count"] = len(mesh.faces)

        logger.info(
            f"  ManifoldBackend: generated mesh with {len(mesh.vertices)} "
            f"vertices, {len(mesh.faces)} faces"
        )

        return mesh, stats

    @staticmethod
    def get_available_generators() -> List[str]:
        """Return sorted list of all registered generator type names."""
        return sorted(GENERATOR_REGISTRY.keys())

    @staticmethod
    def get_generator_categories() -> Dict[str, List[str]]:
        """Return a copy of the category-to-types mapping."""
        return dict(GENERATOR_CATEGORIES)
