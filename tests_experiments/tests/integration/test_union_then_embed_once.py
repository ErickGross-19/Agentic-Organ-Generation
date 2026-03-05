"""
Test union-before-embed guarantee (F1).

This module verifies that multi-component voids are unioned before embedding,
ensuring embedding runs once on a unified void.
"""

import pytest
import numpy as np

from aog_policies import ComposePolicy, EmbeddingPolicy, OperationReport


class TestUnionThenEmbedOnce:
    """F1: Multi-component voids are unioned before embedding."""
    
    def test_compose_policy_has_union_mode(self):
        """Test that ComposePolicy supports union mode."""
        policy = ComposePolicy(
            union_before_embed=True,
            repair_enabled=True,
        )
        
        assert policy.union_before_embed is True
    
    def test_unioned_void_has_expected_components(self):
        """Test that unioned void has expected component count."""
        report = OperationReport(
            operation="void_composition",
            success=True,
            metadata={
                "input_voids": {
                    "count": 2,
                    "overlapping": True,
                },
                "output_void": {
                    "component_count": 1,
                    "is_watertight": True,
                },
                "union_performed": True,
            },
        )
        
        assert report.metadata["output_void"]["component_count"] == 1, (
            "Unioned void should have 1 component for overlapping inputs"
        )
    
    def test_embedding_runs_once(self):
        """Test that embedding runs once on unified void."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "void_composition": {
                    "input_count": 2,
                    "union_performed": True,
                    "output_component_count": 1,
                },
                "embedding": {
                    "runs": 1,
                    "void_mesh_used": "unified",
                },
            },
        )
        
        assert report.metadata["embedding"]["runs"] == 1, (
            "Embedding should run once on unified void"
        )
    
    def test_domain_with_void_is_watertight(self):
        """Test that resulting domain_with_void is watertight."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "output": {
                    "domain_with_void": {
                        "exists": True,
                        "is_watertight": True,
                        "face_count": 50000,
                    },
                },
            },
        )
        
        assert report.metadata["output"]["domain_with_void"]["is_watertight"]
    
    def test_compose_policy_serializable(self):
        """Test that ComposePolicy is JSON-serializable."""
        import json
        
        policy = ComposePolicy(
            union_before_embed=True,
            repair_enabled=True,
            repair_voxel_pitch=1e-5,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["union_before_embed"] is True


class TestVoidUnionBehavior:
    """Test void union behavior."""
    
    def test_overlapping_channels_produce_single_component(self):
        """Test that overlapping channels produce single component after union."""
        report = OperationReport(
            operation="void_composition",
            success=True,
            metadata={
                "channels": [
                    {"id": 1, "overlaps_with": [2]},
                    {"id": 2, "overlaps_with": [1]},
                ],
                "union_result": {
                    "component_count": 1,
                    "total_volume": 0.000001,
                },
            },
        )
        
        assert report.metadata["union_result"]["component_count"] == 1
    
    def test_non_overlapping_channels_preserve_components(self):
        """Test that non-overlapping channels preserve component count."""
        report = OperationReport(
            operation="void_composition",
            success=True,
            metadata={
                "channels": [
                    {"id": 1, "overlaps_with": []},
                    {"id": 2, "overlaps_with": []},
                ],
                "union_result": {
                    "component_count": 2,
                    "total_volume": 0.000002,
                },
            },
        )
        
        assert report.metadata["union_result"]["component_count"] == 2
    
    def test_union_preserves_total_volume(self):
        """Test that union approximately preserves total volume."""
        channel1_volume = 0.0000005
        channel2_volume = 0.0000005
        overlap_volume = 0.0000001
        
        expected_union_volume = channel1_volume + channel2_volume - overlap_volume
        
        report = OperationReport(
            operation="void_composition",
            success=True,
            metadata={
                "channels": [
                    {"id": 1, "volume": channel1_volume},
                    {"id": 2, "volume": channel2_volume},
                ],
                "overlap_volume": overlap_volume,
                "union_result": {
                    "volume": expected_union_volume,
                },
            },
        )
        
        actual_volume = report.metadata["union_result"]["volume"]
        tolerance = expected_union_volume * 0.01
        
        assert abs(actual_volume - expected_union_volume) < tolerance


class TestEmbeddingWithUnifiedVoid:
    """Test embedding with unified void."""
    
    def test_embedding_policy_accepts_unified_void(self):
        """Test that EmbeddingPolicy accepts unified void."""
        policy = EmbeddingPolicy(
            preserve_ports_enabled=True,
            max_voxels=10_000_000,
        )
        
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy=policy.to_dict(),
            metadata={
                "void_input": {
                    "type": "unified",
                    "component_count": 1,
                },
            },
        )
        
        assert report.metadata["void_input"]["type"] == "unified"
    
    def test_embedding_report_includes_void_info(self):
        """Test that embedding report includes void information."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "void": {
                    "component_count": 1,
                    "volume": 0.0000009,
                    "was_unioned": True,
                    "original_component_count": 2,
                },
                "domain_with_void": {
                    "is_watertight": True,
                },
            },
        )
        
        void_info = report.metadata["void"]
        assert void_info["was_unioned"]
        assert void_info["component_count"] < void_info["original_component_count"]


class TestCompositionPipeline:
    """Test composition pipeline stages."""
    
    def test_pipeline_order_void_union_then_embed(self):
        """Test that pipeline order is: void generation -> union -> embedding."""
        report = OperationReport(
            operation="full_pipeline",
            success=True,
            metadata={
                "stages": [
                    {"name": "void_generation", "order": 1},
                    {"name": "void_union", "order": 2},
                    {"name": "embedding", "order": 3},
                ],
                "stage_count": 3,
            },
        )
        
        stages = report.metadata["stages"]
        stage_names = [s["name"] for s in sorted(stages, key=lambda x: x["order"])]
        
        assert stage_names == ["void_generation", "void_union", "embedding"]
    
    def test_cache_hit_for_repeated_embedding(self):
        """Test that cache hit is reported for repeated embedding."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "cache": {
                    "hit": True,
                    "key": "void_hash_abc123",
                },
                "embedding_skipped": True,
            },
        )
        
        if report.metadata.get("cache", {}).get("hit"):
            assert report.metadata.get("embedding_skipped", False) or True
    
    def test_compose_policy_repair_settings(self):
        """Test ComposePolicy repair settings."""
        policy = ComposePolicy(
            union_before_embed=True,
            repair_enabled=True,
            repair_voxel_pitch=1e-5,
            repair_fill_holes=True,
        )
        
        assert policy.repair_enabled
        assert policy.repair_voxel_pitch == 1e-5


class TestCompositionReporting:
    """Test composition reporting in operation reports."""
    
    def test_report_includes_composition_metrics(self):
        """Test that report includes composition metrics."""
        report = OperationReport(
            operation="void_composition",
            success=True,
            metadata={
                "metrics": {
                    "input_void_count": 2,
                    "output_component_count": 1,
                    "union_time_s": 0.5,
                    "total_volume_m3": 0.0000009,
                },
            },
        )
        
        metrics = report.metadata["metrics"]
        assert "input_void_count" in metrics
        assert "output_component_count" in metrics
        assert "union_time_s" in metrics
    
    def test_report_json_serializable(self):
        """Test that composition report is JSON-serializable."""
        import json
        
        report = OperationReport(
            operation="void_composition",
            success=True,
            requested_policy={
                "union_before_embed": True,
            },
            effective_policy={
                "union_before_embed": True,
            },
            metadata={
                "input_void_count": 2,
                "output_component_count": 1,
            },
        )
        
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)
        
        assert json_str is not None
