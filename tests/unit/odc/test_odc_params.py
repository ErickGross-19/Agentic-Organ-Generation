"""Unit tests for ODCParams dataclass."""

import pytest
from generation.ops.odc_params import ODCParams


class TestODCParamsDefaults:
    def test_default_construction(self):
        p = ODCParams()
        assert p.influence_radius == 0.015
        assert p.kill_radius == 0.003
        assert p.step_size == 0.005
        assert p.enable_bifurcation is True
        assert p.progressive_tissue_reveal is True
        assert p.max_initial_branches == 3
        assert p.force_bifurcation_depth == 3
        assert p.vessel_type == "arterial"

    def test_custom_construction(self):
        p = ODCParams(
            influence_radius=0.02,
            kill_radius=0.005,
            step_size=0.003,
            max_initial_branches=5,
            force_bifurcation_depth=4,
        )
        assert p.influence_radius == 0.02
        assert p.kill_radius == 0.005
        assert p.step_size == 0.003
        assert p.max_initial_branches == 5
        assert p.force_bifurcation_depth == 4


class TestODCParamsSerialization:
    def test_to_dict(self):
        p = ODCParams(influence_radius=0.02)
        d = p.to_dict()
        assert d["influence_radius"] == 0.02
        assert "kill_radius" in d
        assert "step_size" in d
        assert "progressive_tissue_reveal" in d

    def test_from_dict(self):
        d = {"influence_radius": 0.02, "kill_radius": 0.005}
        p = ODCParams.from_dict(d)
        assert p.influence_radius == 0.02
        assert p.kill_radius == 0.005
        assert p.step_size == 0.005

    def test_from_dict_ignores_unknown_keys(self):
        d = {"influence_radius": 0.02, "unknown_key": 999}
        p = ODCParams.from_dict(d)
        assert p.influence_radius == 0.02

    def test_roundtrip(self):
        p = ODCParams(
            influence_radius=0.02,
            min_generations_before_tissue=4,
            max_initial_branches=5,
        )
        d = p.to_dict()
        p2 = ODCParams.from_dict(d)
        assert p2.influence_radius == p.influence_radius
        assert p2.min_generations_before_tissue == p.min_generations_before_tissue
        assert p2.max_initial_branches == p.max_initial_branches


class TestODCParamsToSCParams:
    def test_to_sc_params_dict(self):
        p = ODCParams(
            influence_radius=0.02,
            kill_radius=0.005,
            step_size=0.003,
        )
        sc = p.to_sc_params_dict()
        assert sc["influence_radius"] == 0.02
        assert sc["kill_radius"] == 0.005
        assert sc["step_size"] == 0.003
        assert "vessel_type" in sc
        assert "max_steps" in sc


class TestODCParamsAntiStarburst:
    def test_anti_starburst_defaults(self):
        p = ODCParams()
        assert p.min_generations_before_tissue == 2
        assert p.progressive_tissue_reveal is True
        assert p.reveal_depth_per_generation == 0.3
        assert p.max_initial_branches == 3
        assert p.branching_quota_per_length == 2.0
        assert p.force_bifurcation_depth == 3

    def test_anti_starburst_custom(self):
        p = ODCParams(
            min_generations_before_tissue=4,
            progressive_tissue_reveal=False,
            max_initial_branches=2,
            force_bifurcation_depth=5,
        )
        assert p.min_generations_before_tissue == 4
        assert p.progressive_tissue_reveal is False
        assert p.max_initial_branches == 2
        assert p.force_bifurcation_depth == 5
