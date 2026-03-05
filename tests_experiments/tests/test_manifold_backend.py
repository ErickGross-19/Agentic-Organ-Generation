"""
Tests for ManifoldBackend, ManifoldGeneratorPolicy, and runner integration.

These tests validate the Phase 3 integration of MorphoStruct geometry
generators into the DesignSpec pipeline without requiring manifold3d
to be installed.  Generator functions are mocked so that the registry,
dispatch, conversion, and runner plumbing can be verified in isolation.
"""

import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aog_policies.manifold import ManifoldGeneratorPolicy  # noqa: E402
from generation.backends.manifold_backend import (  # noqa: E402
    GENERATOR_CATEGORIES,
    GENERATOR_REGISTRY,
    MM_TO_M,
    ManifoldBackend,
    import_generator_function,
    manifold_to_trimesh,
)


class _FakeMeshData:
    """Mimics the object returned by manifold3d.Manifold.to_mesh()."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self.vert_properties = vertices
        self.tri_verts = faces


class _FakeManifold:
    """Mimics a manifold3d.Manifold with a .to_mesh() method."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        self._mesh_data = _FakeMeshData(vertices, faces)

    def to_mesh(self):
        return self._mesh_data


def _make_cube_manifold(size_mm: float = 10.0):
    """Return a fake manifold representing a cube in millimetres."""
    s = size_mm / 2.0
    vertices = np.array(
        [
            [-s, -s, -s],
            [s, -s, -s],
            [s, s, -s],
            [-s, s, -s],
            [-s, -s, s],
            [s, -s, s],
            [s, s, s],
            [-s, s, s],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [4, 6, 5],
            [4, 7, 6],
            [0, 4, 5],
            [0, 5, 1],
            [2, 6, 7],
            [2, 7, 3],
            [0, 3, 7],
            [0, 7, 4],
            [1, 5, 6],
            [1, 6, 2],
        ],
        dtype=np.int64,
    )
    return _FakeManifold(vertices, faces)


class TestManifoldGeneratorPolicy:
    def test_default_values(self):
        policy = ManifoldGeneratorPolicy()
        assert policy.generator_type == ""
        assert policy.generator_params == {}
        assert policy.convert_units is True

    def test_custom_values(self):
        policy = ManifoldGeneratorPolicy(
            generator_type="gyroid",
            generator_params={"cell_size": 2.0, "wall_thickness": 0.3},
            convert_units=False,
        )
        assert policy.generator_type == "gyroid"
        assert policy.generator_params["cell_size"] == 2.0
        assert policy.convert_units is False

    def test_to_dict(self):
        policy = ManifoldGeneratorPolicy(
            generator_type="trabecular_bone",
            generator_params={"porosity": 0.7},
        )
        d = policy.to_dict()
        assert d["generator_type"] == "trabecular_bone"
        assert d["generator_params"]["porosity"] == 0.7
        assert d["convert_units"] is True

    def test_from_dict(self):
        d = {
            "generator_type": "hepatic_lobule",
            "generator_params": {"lobule_radius": 1.5},
            "convert_units": False,
        }
        policy = ManifoldGeneratorPolicy.from_dict(d)
        assert policy.generator_type == "hepatic_lobule"
        assert policy.generator_params["lobule_radius"] == 1.5
        assert policy.convert_units is False

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "generator_type": "gyroid",
            "unknown_key": 42,
        }
        policy = ManifoldGeneratorPolicy.from_dict(d)
        assert policy.generator_type == "gyroid"
        assert not hasattr(policy, "unknown_key")

    def test_round_trip(self):
        original = ManifoldGeneratorPolicy(
            generator_type="voronoi",
            generator_params={"seed_count": 100},
            convert_units=True,
        )
        restored = ManifoldGeneratorPolicy.from_dict(original.to_dict())
        assert asdict(original) == asdict(restored)


class TestGeneratorRegistry:
    def test_registry_is_non_empty(self):
        assert len(GENERATOR_REGISTRY) > 0

    def test_all_registry_entries_have_correct_format(self):
        for name, entry in GENERATOR_REGISTRY.items():
            assert isinstance(entry, tuple), f"{name}: expected tuple"
            assert len(entry) == 2, f"{name}: expected (module_path, function_name)"
            module_path, function_name = entry
            assert isinstance(module_path, str)
            assert isinstance(function_name, str)
            assert module_path.startswith("app.geometry.")
            assert function_name.startswith("generate_")
            assert function_name.endswith("_from_dict")

    def test_expected_generator_types_present(self):
        expected = [
            "gyroid",
            "schwarz_p",
            "octet_truss",
            "voronoi",
            "honeycomb",
            "trabecular_bone",
            "osteochondral",
            "articular_cartilage",
            "meniscus",
            "tendon_ligament",
            "intervertebral_disc",
            "haversian_bone",
            "hepatic_lobule",
            "cardiac_patch",
            "kidney_tubule",
            "lung_alveoli",
            "pancreatic_islet",
            "liver_sinusoid",
            "multilayer_skin",
            "skeletal_muscle",
            "cornea",
            "adipose_tissue",
            "blood_vessel",
            "nerve_conduit",
            "spinal_cord",
            "bladder",
            "trachea",
            "dentin_pulp",
            "ear_auricle",
            "nasal_septum",
            "organ_on_chip",
            "gradient_scaffold",
            "perfusable_network",
            "vascular_network",
            "vascular_perfusion_dish",
            "primitive",
            "tubular_conduit",
            "porous_disc",
            "lattice",
        ]
        for name in expected:
            assert name in GENERATOR_REGISTRY, f"Missing generator: {name}"

    def test_registry_has_39_generators(self):
        assert len(GENERATOR_REGISTRY) == 39

    def test_categories_cover_all_generators(self):
        all_from_categories = set()
        for types_list in GENERATOR_CATEGORIES.values():
            all_from_categories.update(types_list)
        all_registered = set(GENERATOR_REGISTRY.keys())
        assert all_from_categories == all_registered

    def test_no_duplicate_entries_in_categories(self):
        seen = set()
        for category, types_list in GENERATOR_CATEGORIES.items():
            for t in types_list:
                assert t not in seen, f"Duplicate '{t}' in category '{category}'"
                seen.add(t)


class TestManifoldToTrimesh:
    def test_basic_conversion(self):
        fake = _make_cube_manifold(size_mm=10.0)
        mesh = manifold_to_trimesh(fake, convert_units=True)
        assert mesh.vertices.shape[1] == 3
        assert mesh.faces.shape[1] == 3
        assert len(mesh.vertices) == 8
        assert len(mesh.faces) == 12

    def test_unit_conversion_mm_to_m(self):
        fake = _make_cube_manifold(size_mm=10.0)
        mesh = manifold_to_trimesh(fake, convert_units=True)
        expected_half = 10.0 / 2.0 * MM_TO_M
        assert np.allclose(mesh.vertices.max(), expected_half, atol=1e-9)
        assert np.allclose(mesh.vertices.min(), -expected_half, atol=1e-9)

    def test_no_unit_conversion(self):
        fake = _make_cube_manifold(size_mm=10.0)
        mesh = manifold_to_trimesh(fake, convert_units=False)
        expected_half = 10.0 / 2.0
        assert np.allclose(mesh.vertices.max(), expected_half, atol=1e-9)

    def test_preserves_face_indices(self):
        fake = _make_cube_manifold()
        mesh = manifold_to_trimesh(fake, convert_units=False)
        assert np.array_equal(mesh.faces, np.array(fake.to_mesh().tri_verts))


class TestManifoldBackendGenerate:
    def _mock_gen_func(self, params):
        """Fake generator returning a cube manifold and stats."""
        manifold = _make_cube_manifold(size_mm=20.0)
        stats = {"type": "fake", "param_count": len(params)}
        return manifold, stats

    def test_generate_dispatches_correctly(self):
        backend = ManifoldBackend()
        with patch.dict(
            GENERATOR_REGISTRY,
            {"test_gen": ("app.geometry.test", "generate_test_from_dict")},
        ):
            with patch(
                "generation.backends.manifold_backend.import_generator_function",
                return_value=self._mock_gen_func,
            ):
                mesh, stats = backend.generate(
                    generator_type="test_gen",
                    params={"a": 1},
                    convert_units=True,
                )
                assert mesh.vertices.shape == (8, 3)
                assert mesh.faces.shape == (12, 3)
                assert stats["generator_type"] == "test_gen"
                assert stats["units_converted"] is True
                assert stats["vertex_count"] == 8
                assert stats["face_count"] == 12

    def test_generate_unknown_type_raises(self):
        backend = ManifoldBackend()
        with pytest.raises(ValueError, match="Unknown generator type"):
            backend.generate(generator_type="nonexistent", params={})

    def test_generate_without_unit_conversion(self):
        backend = ManifoldBackend()
        with patch.dict(
            GENERATOR_REGISTRY,
            {"test_gen": ("app.geometry.test", "generate_test_from_dict")},
        ):
            with patch(
                "generation.backends.manifold_backend.import_generator_function",
                return_value=self._mock_gen_func,
            ):
                mesh, stats = backend.generate(
                    generator_type="test_gen",
                    params={},
                    convert_units=False,
                )
                assert stats["units_converted"] is False
                expected_half = 20.0 / 2.0
                assert np.allclose(mesh.vertices.max(), expected_half, atol=1e-9)

    def test_get_available_generators(self):
        generators = ManifoldBackend.get_available_generators()
        assert isinstance(generators, list)
        assert generators == sorted(generators)
        assert len(generators) == len(GENERATOR_REGISTRY)

    def test_get_generator_categories(self):
        categories = ManifoldBackend.get_generator_categories()
        assert "lattice" in categories
        assert "skeletal" in categories
        assert "organ" in categories
        assert "soft_tissue" in categories
        assert "tubular" in categories
        assert "dental" in categories
        assert "microfluidic" in categories
        assert "original" in categories


class TestImportGeneratorFunction:
    def test_import_from_real_module(self):
        func = import_generator_function("os.path", "join")
        assert callable(func)

    def test_import_nonexistent_raises(self):
        with pytest.raises(ImportError, match="Could not import"):
            import_generator_function("no.such.module", "no_func")


class TestRunnerManifoldGeneratorBuildType:
    """Verify the runner correctly dispatches manifold_generator build types."""

    def _make_minimal_spec_dict(self, generator_type="gyroid"):
        return {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42},
            "domains": {
                "main_domain": {
                    "shape": "cylinder",
                    "radius": 0.005,
                    "height": 0.01,
                }
            },
            "components": [
                {
                    "id": "scaffold",
                    "domain_ref": "main_domain",
                    "build": {
                        "type": "manifold_generator",
                        "generator_type": generator_type,
                        "generator_params": {"cell_size": 2.0},
                    },
                    "ports": {},
                }
            ],
            "policies": {},
        }

    def test_runner_recognises_manifold_generator_build_type(self):
        from designspec.runner import DesignSpecRunner, StageReport
        from designspec.spec import DesignSpec

        spec_dict = self._make_minimal_spec_dict()
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)

        runner._compiled_domains["main_domain"] = MagicMock()

        fake_mesh = MagicMock()
        fake_mesh.vertices = np.zeros((10, 3))
        fake_mesh.faces = np.zeros((5, 3), dtype=int)

        fake_report = StageReport(
            stage="component_build:scaffold",
            success=True,
            metadata={"generator_type": "gyroid"},
        )

        with patch.object(
            runner,
            "_build_manifold_generator",
            return_value=(fake_mesh, fake_report),
        ):
            report = runner._stage_component_build("scaffold")
            assert report.success is True

    def test_build_manifold_generator_calls_backend(self):
        from designspec.runner import DesignSpecRunner
        from designspec.spec import DesignSpec

        spec_dict = self._make_minimal_spec_dict()
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)

        fake_trimesh = MagicMock()
        fake_trimesh.vertices = np.zeros((8, 3))
        fake_trimesh.faces = np.zeros((12, 3), dtype=int)

        mock_backend_instance = MagicMock()
        mock_backend_instance.generate.return_value = (
            fake_trimesh,
            {"generator_type": "gyroid", "vertex_count": 8, "face_count": 12},
        )

        with patch(
            "generation.backends.manifold_backend.ManifoldBackend",
            return_value=mock_backend_instance,
        ):
            build = {
                "type": "manifold_generator",
                "generator_type": "gyroid",
                "generator_params": {"cell_size": 2.0},
            }
            mesh, report = runner._build_manifold_generator(build, "scaffold")

            mock_backend_instance.generate.assert_called_once_with(
                generator_type="gyroid",
                params={"cell_size": 2.0},
                convert_units=True,
            )
            assert report.success is True

    def test_build_manifold_generator_uses_policy_fallback(self):
        from designspec.runner import DesignSpecRunner
        from designspec.spec import DesignSpec

        spec_dict = self._make_minimal_spec_dict()
        spec_dict["policies"]["manifold_generator"] = {
            "generator_type": "trabecular_bone",
            "generator_params": {"porosity": 0.8},
            "convert_units": False,
        }
        spec_dict["components"][0]["build"] = {
            "type": "manifold_generator",
        }
        spec = DesignSpec.from_dict(spec_dict)
        runner = DesignSpecRunner(spec)

        runner._stage_compile_policies()

        fake_trimesh = MagicMock()
        fake_trimesh.vertices = np.zeros((8, 3))
        fake_trimesh.faces = np.zeros((12, 3), dtype=int)

        mock_backend_instance = MagicMock()
        mock_backend_instance.generate.return_value = (
            fake_trimesh,
            {"generator_type": "trabecular_bone", "vertex_count": 8, "face_count": 12},
        )

        with patch(
            "generation.backends.manifold_backend.ManifoldBackend",
            return_value=mock_backend_instance,
        ):
            build = {"type": "manifold_generator"}
            mesh, report = runner._build_manifold_generator(build, "scaffold")

            mock_backend_instance.generate.assert_called_once_with(
                generator_type="trabecular_bone",
                params={"porosity": 0.8},
                convert_units=False,
            )


class TestManifoldGeneratorPolicyInAogPolicies:
    def test_importable_from_aog_policies(self):
        from aog_policies import ManifoldGeneratorPolicy as MgpTop

        assert MgpTop is ManifoldGeneratorPolicy

    def test_listed_in_all(self):
        import aog_policies

        assert "ManifoldGeneratorPolicy" in aog_policies.__all__


class TestManifoldBackendInBackendsInit:
    def test_importable_from_generation_backends(self):
        from generation.backends import ManifoldBackend as MbInit

        assert MbInit is ManifoldBackend

    def test_listed_in_all(self):
        from generation import backends

        assert "ManifoldBackend" in backends.__all__
