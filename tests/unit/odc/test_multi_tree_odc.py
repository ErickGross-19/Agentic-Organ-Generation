"""Unit tests for multi-tree ODC coordination."""

import pytest
import numpy as np
from generation.ops.multi_tree_odc import TreeConfig, MultiTreeResult


class TestTreeConfig:
    def test_default_construction(self):
        tc = TreeConfig(tree_id="arterial")
        assert tc.tree_id == "arterial"
        assert tc.vessel_type == "arterial"
        assert tc.params == {}

    def test_custom_construction(self):
        tc = TreeConfig(
            tree_id="venous",
            vessel_type="venous",
            inlet_position=(0.0, 0.0, -0.005),
            inlet_radius=0.0008,
            params={"step_size": 0.003},
        )
        assert tc.tree_id == "venous"
        assert tc.vessel_type == "venous"
        assert tc.inlet_radius == 0.0008
        assert tc.params["step_size"] == 0.003

    def test_to_dict(self):
        tc = TreeConfig(tree_id="arterial", inlet_radius=0.001)
        d = tc.to_dict()
        assert d["tree_id"] == "arterial"
        assert d["inlet_radius"] == 0.001

    def test_from_dict(self):
        d = {
            "tree_id": "portal",
            "vessel_type": "arterial",
            "inlet_position": [0.0, 0.0, 0.005],
            "inlet_radius": 0.0012,
        }
        tc = TreeConfig.from_dict(d)
        assert tc.tree_id == "portal"
        assert tc.inlet_radius == 0.0012

    def test_roundtrip(self):
        tc = TreeConfig(
            tree_id="venous",
            vessel_type="venous",
            inlet_position=(0.0, 0.0, -0.005),
            inlet_radius=0.0008,
            params={"step_size": 0.003},
        )
        d = tc.to_dict()
        tc2 = TreeConfig.from_dict(d)
        assert tc2.tree_id == tc.tree_id
        assert tc2.vessel_type == tc.vessel_type
        assert tc2.inlet_radius == tc.inlet_radius


class TestMultiTreeResult:
    def test_construction(self):
        result = MultiTreeResult(
            networks={},
            collision_count=0,
            iterations_per_tree={},
        )
        assert result.collision_count == 0
        assert len(result.networks) == 0

    def test_with_data(self):
        from unittest.mock import MagicMock
        mock_net = MagicMock()
        result = MultiTreeResult(
            networks={"arterial": mock_net},
            collision_count=5,
            iterations_per_tree={"arterial": 100},
        )
        assert "arterial" in result.networks
        assert result.collision_count == 5
        assert result.iterations_per_tree["arterial"] == 100
