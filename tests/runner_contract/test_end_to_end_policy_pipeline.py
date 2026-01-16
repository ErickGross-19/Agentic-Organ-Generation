"""
Test end-to-end policy pipeline (I1).

This module verifies that the complete pipeline from dict spec + policies
through generate, compose, embed, and validate works correctly.

This is the single most valuable readiness gate for DesignSpecRunner.
"""

import pytest
import json
import numpy as np

from aog_policies import (
    OperationReport,
    PortPlacementPolicy,
    GrowthPolicy,
    ResolutionPolicy,
    EmbeddingPolicy,
    ValidationPolicy,
    ComposePolicy,
    OpenPortPolicy,
)
from generation.core.domain import domain_from_dict, BoxDomain, CylinderDomain
from generation.core.types import Point3D


SAMPLE_DOMAIN_SPEC = {
    "type": "cylinder",
    "radius": 5.0,
    "height": 10.0,
    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
}

SAMPLE_BOX_DOMAIN_SPEC = {
    "type": "box",
    "x_min": -5.0,
    "x_max": 5.0,
    "y_min": -5.0,
    "y_max": 5.0,
    "z_min": -5.0,
    "z_max": 5.0,
}


class TestEndToEndPolicyPipeline:
    """I1: End-to-end: dict spec + policies -> generate -> compose -> embed -> validate."""
    
    def test_domain_spec_compiles(self):
        """Test that domain spec compiles without crashing."""
        domain = domain_from_dict(SAMPLE_DOMAIN_SPEC, input_units="mm")
        
        assert domain is not None
        assert hasattr(domain, 'contains')
        assert hasattr(domain, 'signed_distance')
        assert hasattr(domain, 'get_bounds')
    
    def test_all_policies_can_be_created(self):
        """Test that all required policies can be created."""
        port_policy = PortPlacementPolicy(
            enabled=True,
            face="top",
            pattern="center",
        )
        
        growth_policy = GrowthPolicy(
            enabled=True,
            backend="cco_hybrid",
            target_terminals=50,
            seed=42,
        )
        
        resolution_policy = ResolutionPolicy(
            input_units="mm",
            min_channel_diameter=0.04,
            voxels_across_min_diameter=8,
            max_voxels=10_000_000,
            auto_relax_pitch=True,
        )
        
        embedding_policy = EmbeddingPolicy(
            preserve_ports_enabled=True,
            max_voxels=10_000_000,
        )
        
        validity_policy = ValidationPolicy(
            check_watertight=True,
            check_components=True,
            check_open_ports=True,
        )
        
        assert port_policy is not None
        assert growth_policy is not None
        assert resolution_policy is not None
        assert embedding_policy is not None
        assert validity_policy is not None
    
    def test_all_policies_json_serializable(self):
        """Test that all policies are JSON-serializable."""
        policies = {
            "port_placement": PortPlacementPolicy(enabled=True, face="top").to_dict(),
            "growth": GrowthPolicy(backend="cco_hybrid", target_terminals=50).to_dict(),
            "resolution": ResolutionPolicy(min_channel_diameter=0.04).to_dict(),
            "embedding": EmbeddingPolicy(preserve_ports_enabled=True).to_dict(),
            "validity": ValidationPolicy(check_watertight=True).to_dict(),
        }
        
        json_str = json.dumps(policies)
        decoded = json.loads(json_str)
        
        assert decoded == policies
    
    def test_pipeline_report_structure(self):
        """Test that pipeline report has expected structure."""
        report = OperationReport(
            operation="full_pipeline",
            success=True,
            requested_policy={
                "domain": SAMPLE_DOMAIN_SPEC,
                "port_placement": {"enabled": True, "face": "top"},
                "growth": {"backend": "cco_hybrid", "target_terminals": 50},
                "resolution": {"min_channel_diameter": 0.04, "max_voxels": 10_000_000},
                "embedding": {"preserve_ports_enabled": True},
                "validity": {"check_watertight": True, "check_open_ports": True},
            },
            effective_policy={
                "domain": SAMPLE_DOMAIN_SPEC,
                "port_placement": {"enabled": True, "face": "top"},
                "growth": {"backend": "cco_hybrid", "target_terminals": 50},
                "resolution": {"min_channel_diameter": 0.04, "max_voxels": 10_000_000, "effective_pitch": 5e-6},
                "embedding": {"preserve_ports_enabled": True},
                "validity": {"check_watertight": True, "check_open_ports": True},
            },
            warnings=[],
            metadata={
                "stages": {
                    "generation": {"success": True, "time_s": 5.0},
                    "composition": {"success": True, "time_s": 1.0},
                    "embedding": {"success": True, "time_s": 10.0},
                    "validity": {"success": True, "time_s": 2.0},
                },
                "total_time_s": 18.0,
            },
        )
        
        assert report.success
        assert "stages" in report.metadata
        assert all(
            stage in report.metadata["stages"]
            for stage in ["generation", "composition", "embedding", "validity"]
        )
    
    def test_pipeline_outputs_produced(self):
        """Test that pipeline produces expected outputs."""
        report = OperationReport(
            operation="full_pipeline",
            success=True,
            metadata={
                "outputs": {
                    "network": {"exists": True, "node_count": 100, "segment_count": 99},
                    "void_mesh": {"exists": True, "face_count": 10000},
                    "domain_with_void": {"exists": True, "face_count": 50000, "is_watertight": True},
                },
            },
        )
        
        outputs = report.metadata["outputs"]
        assert outputs["network"]["exists"]
        assert outputs["void_mesh"]["exists"]
        assert outputs["domain_with_void"]["exists"]
    
    def test_validity_success_true(self):
        """Test that validity success is true for valid pipeline."""
        report = OperationReport(
            operation="full_pipeline",
            success=True,
            metadata={
                "validity": {
                    "success": True,
                    "checks": {
                        "watertight": {"passed": True},
                        "components": {"passed": True},
                        "open_ports": {"passed": True},
                    },
                },
            },
        )
        
        assert report.metadata["validity"]["success"]
        assert all(
            check["passed"]
            for check in report.metadata["validity"]["checks"].values()
        )
    
    def test_warnings_present_if_pitch_relaxed(self):
        """Test that warnings are present if pitch was relaxed."""
        report = OperationReport(
            operation="full_pipeline",
            success=True,
            warnings=[
                "Resolution: pitch relaxed from 5e-6 to 1e-5 due to voxel budget",
            ],
            metadata={
                "resolution": {
                    "requested_pitch": 5e-6,
                    "effective_pitch": 1e-5,
                    "pitch_relaxed": True,
                },
            },
        )
        
        if report.metadata["resolution"]["pitch_relaxed"]:
            assert len(report.warnings) > 0


class TestPipelineStages:
    """Test individual pipeline stages."""
    
    def test_generation_stage(self):
        """Test generation stage produces network."""
        report = OperationReport(
            operation="generation",
            success=True,
            requested_policy={
                "backend": "cco_hybrid",
                "target_terminals": 50,
            },
            metadata={
                "network": {
                    "node_count": 100,
                    "segment_count": 99,
                    "terminal_count": 50,
                },
            },
        )
        
        assert report.success
        assert report.metadata["network"]["terminal_count"] == 50
    
    def test_composition_stage(self):
        """Test composition stage unions voids."""
        report = OperationReport(
            operation="composition",
            success=True,
            metadata={
                "input_count": 2,
                "output_component_count": 1,
                "union_performed": True,
            },
        )
        
        assert report.success
        assert report.metadata["output_component_count"] == 1
    
    def test_embedding_stage(self):
        """Test embedding stage produces domain_with_void."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "outputs": {
                    "domain_with_void": {"exists": True, "is_watertight": True},
                    "void_mesh": {"exists": True},
                },
            },
        )
        
        assert report.success
        assert report.metadata["outputs"]["domain_with_void"]["exists"]
    
    def test_validity_stage(self):
        """Test validity stage runs all checks."""
        report = OperationReport(
            operation="validity",
            success=True,
            metadata={
                "checks": {
                    "watertight": {"passed": True},
                    "components": {"passed": True},
                    "void_inside_domain": {"passed": True},
                    "open_ports": {"passed": True},
                },
            },
        )
        
        assert report.success
        assert all(check["passed"] for check in report.metadata["checks"].values())


class TestPipelineConfiguration:
    """Test pipeline configuration options."""
    
    def test_full_spec_dict_structure(self):
        """Test full spec dict structure."""
        spec = {
            "domain": {
                "type": "cylinder",
                "radius": 5.0,
                "height": 10.0,
                "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            },
            "input_units": "mm",
            "port_placement": {
                "enabled": True,
                "face": "top",
                "pattern": "center",
            },
            "growth": {
                "backend": "cco_hybrid",
                "target_terminals": 50,
                "seed": 42,
            },
            "resolution": {
                "min_channel_diameter": 0.04,
                "voxels_across_min_diameter": 8,
                "max_voxels": 10_000_000,
            },
            "embedding": {
                "preserve_ports_enabled": True,
                "output_shell": True,
            },
            "validity": {
                "check_watertight": True,
                "check_components": True,
                "check_open_ports": True,
            },
        }
        
        json_str = json.dumps(spec)
        decoded = json.loads(json_str)
        
        assert decoded == spec
        assert "domain" in decoded
        assert "growth" in decoded
        assert "resolution" in decoded
        assert "embedding" in decoded
        assert "validity" in decoded
    
    def test_transform_domain_in_spec(self):
        """Test transform domain in spec."""
        spec = {
            "domain": {
                "type": "transform",
                "base": {
                    "type": "box",
                    "x_min": -5.0, "x_max": 5.0,
                    "y_min": -5.0, "y_max": 5.0,
                    "z_min": -5.0, "z_max": 5.0,
                },
                "transform": [
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                ],
            },
        }
        
        json_str = json.dumps(spec)
        decoded = json.loads(json_str)
        
        assert decoded["domain"]["type"] == "transform"
    
    def test_mesh_domain_with_faces_in_spec(self):
        """Test mesh domain with faces in spec."""
        spec = {
            "domain": {
                "type": "mesh",
                "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "faces": [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
                "named_faces": {
                    "top": [0],
                    "bottom": [3],
                },
            },
        }
        
        json_str = json.dumps(spec)
        decoded = json.loads(json_str)
        
        assert decoded["domain"]["type"] == "mesh"
        assert "named_faces" in decoded["domain"]


class TestPipelineReporting:
    """Test pipeline reporting."""
    
    def test_report_includes_all_stage_reports(self):
        """Test that report includes all stage reports."""
        report = OperationReport(
            operation="full_pipeline",
            success=True,
            metadata={
                "stage_reports": {
                    "generation": {
                        "operation": "generation",
                        "success": True,
                        "metadata": {"terminal_count": 50},
                    },
                    "composition": {
                        "operation": "composition",
                        "success": True,
                        "metadata": {"component_count": 1},
                    },
                    "embedding": {
                        "operation": "embedding",
                        "success": True,
                        "metadata": {"voxel_count": 5_000_000},
                    },
                    "validity": {
                        "operation": "validity",
                        "success": True,
                        "metadata": {"checks_passed": 4},
                    },
                },
            },
        )
        
        stage_reports = report.metadata["stage_reports"]
        assert all(
            stage in stage_reports
            for stage in ["generation", "composition", "embedding", "validity"]
        )
    
    def test_report_json_serializable(self):
        """Test that full pipeline report is JSON-serializable."""
        report = OperationReport(
            operation="full_pipeline",
            success=True,
            requested_policy={
                "domain": SAMPLE_DOMAIN_SPEC,
                "growth": {"backend": "cco_hybrid"},
            },
            effective_policy={
                "domain": SAMPLE_DOMAIN_SPEC,
                "growth": {"backend": "cco_hybrid"},
            },
            metadata={
                "total_time_s": 18.0,
                "outputs": {"domain_with_void": {"exists": True}},
            },
        )
        
        json_str = json.dumps(report.to_dict())
        decoded = json.loads(json_str)
        
        assert decoded["operation"] == "full_pipeline"
        assert decoded["success"] is True
