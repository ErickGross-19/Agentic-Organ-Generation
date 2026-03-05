"""Unit tests for ODC policy with new parameters and presets."""

import pytest
from aog_policies.odc import (
    ODCPolicy,
    HierarchicalTissuePolicy,
    AntiStarburstPolicy,
    MultiTreePolicy,
    get_odc_preset,
)


class TestODCPolicyDefaults:
    def test_default_weights(self):
        p = ODCPolicy()
        assert p.coverage_weight == 0.30
        assert p.ordering_weight == 0.25
        assert p.murray_weight == 0.15
        assert p.flow_weight == 0.10
        assert p.anti_starburst_weight == 0.10
        assert p.branching_regularity_weight == 0.10

    def test_has_anti_starburst_policy(self):
        p = ODCPolicy()
        assert isinstance(p.anti_starburst, AntiStarburstPolicy)
        assert p.anti_starburst.enabled is True

    def test_has_multi_tree_policy(self):
        p = ODCPolicy()
        assert isinstance(p.multi_tree, MultiTreePolicy)
        assert p.multi_tree.enabled is False

    def test_has_tissue_policy(self):
        p = ODCPolicy()
        assert isinstance(p.tissue, HierarchicalTissuePolicy)
        assert p.tissue.distribution_type == "uniform"


class TestAntiStarburstPolicy:
    def test_defaults(self):
        asp = AntiStarburstPolicy()
        assert asp.min_generations_before_tissue == 2
        assert asp.progressive_tissue_reveal is True
        assert asp.reveal_depth_per_generation == 0.3
        assert asp.max_initial_branches == 3
        assert asp.branching_quota_per_length == 2.0
        assert asp.force_bifurcation_depth == 3

    def test_to_dict(self):
        asp = AntiStarburstPolicy(max_initial_branches=5)
        d = asp.to_dict()
        assert d["max_initial_branches"] == 5

    def test_from_dict(self):
        d = {"max_initial_branches": 5, "force_bifurcation_depth": 4}
        asp = AntiStarburstPolicy.from_dict(d)
        assert asp.max_initial_branches == 5
        assert asp.force_bifurcation_depth == 4

    def test_roundtrip(self):
        asp = AntiStarburstPolicy(
            min_generations_before_tissue=4,
            max_initial_branches=2,
        )
        d = asp.to_dict()
        asp2 = AntiStarburstPolicy.from_dict(d)
        assert asp2.min_generations_before_tissue == asp.min_generations_before_tissue
        assert asp2.max_initial_branches == asp.max_initial_branches


class TestMultiTreePolicy:
    def test_defaults(self):
        mtp = MultiTreePolicy()
        assert mtp.enabled is False
        assert mtp.collision_radius == 0.001
        assert mtp.interleave_strategy == "round_robin"
        assert mtp.tree_configs == []

    def test_to_dict(self):
        mtp = MultiTreePolicy(enabled=True, collision_radius=0.002)
        d = mtp.to_dict()
        assert d["enabled"] is True
        assert d["collision_radius"] == 0.002

    def test_from_dict(self):
        d = {"enabled": True, "collision_radius": 0.002}
        mtp = MultiTreePolicy.from_dict(d)
        assert mtp.enabled is True
        assert mtp.collision_radius == 0.002


class TestODCPolicySerialization:
    def test_to_dict_includes_new_fields(self):
        p = ODCPolicy()
        d = p.to_dict()
        assert "anti_starburst" in d
        assert "multi_tree" in d
        assert "anti_starburst_weight" in d
        assert "branching_regularity_weight" in d
        assert "preset" in d

    def test_from_dict(self):
        d = {
            "anti_starburst_weight": 0.15,
            "anti_starburst": {"max_initial_branches": 4},
            "multi_tree": {"enabled": True},
        }
        p = ODCPolicy.from_dict(d)
        assert p.anti_starburst_weight == 0.15
        assert p.anti_starburst.max_initial_branches == 4
        assert p.multi_tree.enabled is True

    def test_roundtrip(self):
        p = ODCPolicy(
            anti_starburst_weight=0.15,
            anti_starburst=AntiStarburstPolicy(max_initial_branches=5),
            multi_tree=MultiTreePolicy(enabled=True),
            preset="custom",
        )
        d = p.to_dict()
        p2 = ODCPolicy.from_dict(d)
        assert p2.anti_starburst_weight == p.anti_starburst_weight
        assert p2.anti_starburst.max_initial_branches == p.anti_starburst.max_initial_branches
        assert p2.multi_tree.enabled == p.multi_tree.enabled
        assert p2.preset == p.preset


class TestODCPresets:
    def test_conservative_preset(self):
        p = get_odc_preset("conservative")
        assert p.preset == "conservative"
        assert p.anti_starburst.min_generations_before_tissue == 3
        assert p.anti_starburst.max_initial_branches == 2

    def test_aggressive_preset(self):
        p = get_odc_preset("aggressive")
        assert p.preset == "aggressive"
        assert p.anti_starburst.min_generations_before_tissue == 1
        assert p.anti_starburst.max_initial_branches == 4

    def test_balanced_preset(self):
        p = get_odc_preset("balanced")
        assert p.preset == "balanced"

    def test_liver_preset(self):
        p = get_odc_preset("liver")
        assert p.preset == "liver"
        assert p.tissue.distribution_type == "liver_lobule"

    def test_lung_preset(self):
        p = get_odc_preset("lung")
        assert p.preset == "lung"
        assert p.tissue.distribution_type == "lung_bronchiole"

    def test_kidney_preset(self):
        p = get_odc_preset("kidney")
        assert p.preset == "kidney"
        assert p.tissue.distribution_type == "kidney_nephron"

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown ODC preset"):
            get_odc_preset("nonexistent")

    def test_all_presets_are_odc_policy(self):
        for name in ["conservative", "aggressive", "balanced", "liver", "lung", "kidney"]:
            p = get_odc_preset(name)
            assert isinstance(p, ODCPolicy)
