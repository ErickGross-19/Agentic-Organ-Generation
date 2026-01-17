"""Tests for spec hash stability."""

import pytest
import json
from pathlib import Path

from designspec.spec import DesignSpec


class TestSpecHashStability:
    """Tests for spec hash stability across different key orderings."""
    
    def test_same_spec_same_hash(self):
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        
        spec1 = DesignSpec.from_dict(spec_dict)
        spec2 = DesignSpec.from_dict(spec_dict)
        
        assert spec1.spec_hash == spec2.spec_hash
    
    def test_different_key_order_same_hash(self):
        spec_dict1 = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        
        spec_dict2 = {
            "components": [],
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "policies": {},
            "meta": {"input_units": "m", "seed": 42},
            "schema": {"version": "1.0.0", "name": "aog_designspec"},
        }
        
        spec1 = DesignSpec.from_dict(spec_dict1)
        spec2 = DesignSpec.from_dict(spec_dict2)
        
        assert spec1.spec_hash == spec2.spec_hash
    
    def test_different_seed_different_hash(self):
        spec_dict1 = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        
        spec_dict2 = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 123, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        
        spec1 = DesignSpec.from_dict(spec_dict1)
        spec2 = DesignSpec.from_dict(spec_dict2)
        
        assert spec1.spec_hash != spec2.spec_hash
    
    def test_different_domain_different_hash(self):
        spec_dict1 = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        
        spec_dict2 = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -2, "x_max": 2, "y_min": -2, "y_max": 2, "z_min": -2, "z_max": 2}},
            "components": [],
        }
        
        spec1 = DesignSpec.from_dict(spec_dict1)
        spec2 = DesignSpec.from_dict(spec_dict2)
        
        assert spec1.spec_hash != spec2.spec_hash
    
    def test_hash_is_hex_string(self):
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        assert all(c in "0123456789abcdef" for c in spec.spec_hash)
    
    def test_hash_has_reasonable_length(self):
        spec_dict = {
            "schema": {"name": "aog_designspec", "version": "1.0.0"},
            "meta": {"seed": 42, "input_units": "m"},
            "policies": {},
            "domains": {"main": {"type": "box", "x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1, "z_min": -1, "z_max": 1}},
            "components": [],
        }
        
        spec = DesignSpec.from_dict(spec_dict)
        
        assert len(spec.spec_hash) >= 16


class TestGoldenExampleHashStability:
    """Tests for golden example hash stability."""
    
    def test_golden_example_hash_stable(self):
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "designspec"
        golden_path = fixtures_dir / "golden_example_v1.json"
        
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        spec1 = DesignSpec.from_json(golden_path)
        spec2 = DesignSpec.from_json(golden_path)
        
        assert spec1.spec_hash == spec2.spec_hash
    
    def test_golden_example_hash_from_dict_matches_file(self):
        fixtures_dir = Path(__file__).parent.parent / "fixtures" / "designspec"
        golden_path = fixtures_dir / "golden_example_v1.json"
        
        if not golden_path.exists():
            pytest.skip("Golden example fixture not found")
        
        with open(golden_path) as f:
            spec_dict = json.load(f)
        
        spec_from_file = DesignSpec.from_json(golden_path)
        spec_from_dict = DesignSpec.from_dict(spec_dict)
        
        assert spec_from_file.spec_hash == spec_from_dict.spec_hash
