"""Tests for golden fixture roundtrip."""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec
from designspec.schema import SCHEMA_NAME, SCHEMA_VERSION


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "designspec"


class TestGoldenFixtureLoad:
    """Tests for loading the golden fixture."""
    
    def test_golden_fixture_exists(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        assert golden_path.exists(), f"Golden fixture not found at {golden_path}"
    
    def test_golden_fixture_is_valid_json(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        with open(golden_path) as f:
            data = json.load(f)
        
        assert isinstance(data, dict)
    
    def test_golden_fixture_has_required_keys(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        with open(golden_path) as f:
            data = json.load(f)
        
        assert "schema" in data
        assert "meta" in data
        assert "policies" in data
        assert "domains" in data
        assert "components" in data
    
    def test_golden_fixture_schema_correct(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        with open(golden_path) as f:
            data = json.load(f)
        
        assert data["schema"]["name"] == SCHEMA_NAME
        assert data["schema"]["version"] == SCHEMA_VERSION


class TestGoldenFixtureNormalize:
    """Tests for normalizing the golden fixture."""
    
    def test_golden_fixture_loads_as_designspec(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert spec is not None
        assert spec.spec_hash is not None
    
    def test_golden_fixture_meta_preserved(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert spec.meta["name"] == "golden_example_v1"
        assert spec.seed == 1234
        assert spec.input_units == "mm"
    
    def test_golden_fixture_domains_normalized(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        main_domain = spec.domains["main_domain"]
        assert main_domain["x_min"] == pytest.approx(-0.015)
        assert main_domain["x_max"] == pytest.approx(0.015)
        assert main_domain["z_min"] == pytest.approx(-0.010)
        assert main_domain["z_max"] == pytest.approx(0.010)
    
    def test_golden_fixture_policies_normalized(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        resolution = spec.policies.get("resolution", {})
        assert resolution.get("min_channel_diameter") == pytest.approx(0.00002)
    
    def test_golden_fixture_components_normalized(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert len(spec.components) == 2
        
        net_1 = spec.components[0]
        assert net_1["id"] == "net_1"
        
        inlet = net_1["ports"]["inlets"][0]
        assert inlet["radius"] == pytest.approx(0.0003)


class TestGoldenFixtureValidate:
    """Tests for validating the golden fixture."""
    
    def test_golden_fixture_no_validation_errors(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert len(spec.warnings) == 0 or all("alias" not in w.lower() for w in spec.warnings)
    
    def test_golden_fixture_has_multi_component(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert len(spec.components) >= 2
    
    def test_golden_fixture_has_backend_network_component(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        backend_network_components = [
            c for c in spec.components
            if c.get("build", {}).get("type") == "backend_network"
        ]
        assert len(backend_network_components) >= 1
    
    def test_golden_fixture_has_primitive_channels_component(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        primitive_channels_components = [
            c for c in spec.components
            if c.get("build", {}).get("type") == "primitive_channels"
        ]
        assert len(primitive_channels_components) >= 1


class TestGoldenFixtureHashStability:
    """Tests for golden fixture hash stability."""
    
    def test_golden_fixture_hash_stable(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec1 = DesignSpec.from_json(golden_path)
        spec2 = DesignSpec.from_json(golden_path)
        
        assert spec1.spec_hash == spec2.spec_hash
    
    def test_golden_fixture_hash_from_dict_matches(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        with open(golden_path) as f:
            data = json.load(f)
        
        spec_from_file = DesignSpec.from_json(golden_path)
        spec_from_dict = DesignSpec.from_dict(data)
        
        assert spec_from_file.spec_hash == spec_from_dict.spec_hash
    
    def test_golden_fixture_hash_is_hex(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert all(c in "0123456789abcdef" for c in spec.spec_hash)
    
    def test_golden_fixture_hash_reasonable_length(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        assert len(spec.spec_hash) >= 16


class TestGoldenFixtureRoundtrip:
    """Tests for golden fixture roundtrip serialization."""
    
    def test_golden_fixture_raw_preserved(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        with open(golden_path) as f:
            original = json.load(f)
        
        spec = DesignSpec.from_json(golden_path)
        
        assert spec.raw["schema"]["name"] == original["schema"]["name"]
        assert spec.raw["meta"]["seed"] == original["meta"]["seed"]
    
    def test_golden_fixture_normalized_different_from_raw(self):
        golden_path = FIXTURES_DIR / "golden_example_v1.json"
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec = DesignSpec.from_json(golden_path)
        
        raw_x_min = spec.raw["domains"]["main_domain"]["x_min"]
        normalized_x_min = spec.normalized["domains"]["main_domain"]["x_min"]
        
        assert raw_x_min != normalized_x_min
        assert raw_x_min == -15
        assert normalized_x_min == pytest.approx(-0.015)
