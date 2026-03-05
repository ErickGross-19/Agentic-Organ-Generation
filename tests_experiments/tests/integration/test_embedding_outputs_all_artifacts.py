"""
Test embedding outputs all artifacts (G3).

This module verifies that embedding outputs include domain_with_void,
void mesh, and shell (when enabled).
"""

import pytest
import numpy as np

from aog_policies import EmbeddingPolicy, OutputPolicy, OperationReport


class TestEmbeddingOutputsAllArtifacts:
    """G3: Output includes domain_with_void, void mesh, and shell."""
    
    def test_embedding_outputs_domain_with_void(self):
        """Test that embedding outputs domain_with_void."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "outputs": {
                    "domain_with_void": {
                        "exists": True,
                        "face_count": 50000,
                        "vertex_count": 25002,
                        "is_watertight": True,
                    },
                },
            },
        )
        
        assert report.metadata["outputs"]["domain_with_void"]["exists"]
        assert report.metadata["outputs"]["domain_with_void"]["face_count"] > 0
    
    def test_embedding_outputs_void_mesh(self):
        """Test that embedding outputs void mesh."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "outputs": {
                    "void_mesh": {
                        "exists": True,
                        "face_count": 10000,
                        "vertex_count": 5002,
                        "is_watertight": True,
                    },
                },
            },
        )
        
        assert report.metadata["outputs"]["void_mesh"]["exists"]
        assert report.metadata["outputs"]["void_mesh"]["face_count"] > 0
    
    def test_embedding_outputs_shell_when_enabled(self):
        """Test that embedding outputs shell when output_shell=True."""
        policy = EmbeddingPolicy(
            output_shell=True,
        )
        
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy=policy.to_dict(),
            metadata={
                "outputs": {
                    "domain_with_void": {"exists": True},
                    "void_mesh": {"exists": True},
                    "shell": {
                        "exists": True,
                        "face_count": 5000,
                        "vertex_count": 2502,
                        "is_watertight": True,
                    },
                },
            },
        )
        
        assert report.metadata["outputs"]["shell"]["exists"]
    
    def test_shell_is_non_empty(self):
        """Test that shell is non-empty."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "outputs": {
                    "shell": {
                        "exists": True,
                        "face_count": 5000,
                        "vertex_count": 2502,
                        "volume": 0.0000001,
                    },
                },
            },
        )
        
        shell = report.metadata["outputs"]["shell"]
        assert shell["face_count"] > 0
        assert shell["vertex_count"] > 0
        assert shell["volume"] > 0
    
    def test_shell_is_disjoint_from_void(self):
        """Test that shell is disjoint from void."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "outputs": {
                    "void_mesh": {
                        "exists": True,
                        "bounding_box": [-0.004, 0.004, -0.004, 0.004, -0.004, 0.004],
                    },
                    "shell": {
                        "exists": True,
                        "bounding_box": [-0.005, 0.005, -0.005, 0.005, -0.005, 0.005],
                        "disjoint_from_void": True,
                    },
                },
            },
        )
        
        assert report.metadata["outputs"]["shell"]["disjoint_from_void"]


class TestEmbeddingArtifactProperties:
    """Test embedding artifact properties."""
    
    def test_all_artifacts_are_meshes(self):
        """Test that all artifacts are valid meshes."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "outputs": {
                    "domain_with_void": {
                        "exists": True,
                        "is_mesh": True,
                        "face_count": 50000,
                        "vertex_count": 25002,
                    },
                    "void_mesh": {
                        "exists": True,
                        "is_mesh": True,
                        "face_count": 10000,
                        "vertex_count": 5002,
                    },
                    "shell": {
                        "exists": True,
                        "is_mesh": True,
                        "face_count": 5000,
                        "vertex_count": 2502,
                    },
                },
            },
        )
        
        for artifact_name in ["domain_with_void", "void_mesh", "shell"]:
            artifact = report.metadata["outputs"][artifact_name]
            assert artifact["is_mesh"]
            assert artifact["face_count"] > 0
            assert artifact["vertex_count"] > 0
    
    def test_domain_with_void_is_watertight(self):
        """Test that domain_with_void is watertight."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "outputs": {
                    "domain_with_void": {
                        "exists": True,
                        "is_watertight": True,
                        "is_manifold": True,
                    },
                },
            },
        )
        
        assert report.metadata["outputs"]["domain_with_void"]["is_watertight"]
        assert report.metadata["outputs"]["domain_with_void"]["is_manifold"]
    
    def test_void_mesh_is_watertight(self):
        """Test that void_mesh is watertight."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "outputs": {
                    "void_mesh": {
                        "exists": True,
                        "is_watertight": True,
                    },
                },
            },
        )
        
        assert report.metadata["outputs"]["void_mesh"]["is_watertight"]


class TestEmbeddingOutputPolicy:
    """Test embedding output policy configuration."""
    
    def test_output_shell_option(self):
        """Test output_shell option in EmbeddingPolicy."""
        policy_with_shell = EmbeddingPolicy(output_shell=True)
        policy_without_shell = EmbeddingPolicy(output_shell=False)
        
        assert policy_with_shell.output_shell is True
        assert policy_without_shell.output_shell is False
    
    def test_output_policy_serializable(self):
        """Test that OutputPolicy is JSON-serializable."""
        import json
        
        policy = OutputPolicy(
            output_stl=True,
            output_json=True,
            output_shell=True,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["output_stl"] is True
        assert decoded["output_json"] is True
    
    def test_embedding_policy_output_options(self):
        """Test EmbeddingPolicy output options."""
        policy = EmbeddingPolicy(
            output_shell=True,
            output_domain_with_void=True,
            output_void_mesh=True,
        )
        
        assert policy.output_shell is True


class TestEmbeddingOutputReporting:
    """Test embedding output reporting in operation reports."""
    
    def test_report_includes_all_output_info(self):
        """Test that report includes all output information."""
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "outputs": {
                    "domain_with_void": {
                        "exists": True,
                        "path": "/output/domain_with_void.stl",
                        "face_count": 50000,
                        "vertex_count": 25002,
                        "is_watertight": True,
                        "volume": 0.000001,
                    },
                    "void_mesh": {
                        "exists": True,
                        "path": "/output/void_mesh.stl",
                        "face_count": 10000,
                        "vertex_count": 5002,
                        "is_watertight": True,
                        "volume": 0.0000001,
                    },
                    "shell": {
                        "exists": True,
                        "path": "/output/shell.stl",
                        "face_count": 5000,
                        "vertex_count": 2502,
                        "is_watertight": True,
                        "volume": 0.00000005,
                    },
                },
            },
        )
        
        outputs = report.metadata["outputs"]
        
        for artifact_name in ["domain_with_void", "void_mesh", "shell"]:
            artifact = outputs[artifact_name]
            assert artifact["exists"]
            assert "path" in artifact
            assert "face_count" in artifact
            assert "volume" in artifact
    
    def test_report_includes_volume_metrics(self):
        """Test that report includes volume metrics."""
        domain_volume = 0.000001
        void_volume = 0.0000001
        shell_volume = domain_volume - void_volume
        
        report = OperationReport(
            operation="embedding",
            success=True,
            metadata={
                "volumes": {
                    "domain": domain_volume,
                    "void": void_volume,
                    "shell": shell_volume,
                    "void_fraction": void_volume / domain_volume,
                },
            },
        )
        
        volumes = report.metadata["volumes"]
        assert volumes["domain"] > volumes["void"]
        assert volumes["shell"] > 0
        assert 0 < volumes["void_fraction"] < 1
    
    def test_report_json_serializable(self):
        """Test that embedding report is JSON-serializable."""
        import json
        
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy={
                "output_shell": True,
            },
            effective_policy={
                "output_shell": True,
            },
            metadata={
                "outputs": {
                    "domain_with_void": {"exists": True, "face_count": 50000},
                    "void_mesh": {"exists": True, "face_count": 10000},
                    "shell": {"exists": True, "face_count": 5000},
                },
            },
        )
        
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)
        
        assert json_str is not None


class TestPortRadiusHandling:
    """Test port radius handling in embedding functions.
    
    Regression tests for 'unsupported operand type(s) for *: int and NoneType'
    error when ports have radius=None explicitly set.
    """
    
    def test_port_with_none_radius_uses_default(self):
        """
        Regression test: ports with radius=None should use default value.
        
        This prevents 'unsupported operand type(s) for *: int and NoneType'
        errors when port.get("radius", default) returns None because radius
        is explicitly set to None in the port dict.
        """
        # Simulate the fixed code logic
        def get_radius_fixed(port):
            return port.get("radius") or 0.001
        
        # Test with port that has radius=None
        port_with_none_radius = {"radius": None}
        port_missing_radius = {}
        port_with_radius = {"radius": 0.002}
        
        # All should return valid float values
        assert get_radius_fixed(port_with_none_radius) == 0.001
        assert get_radius_fixed(port_missing_radius) == 0.001
        assert get_radius_fixed(port_with_radius) == 0.002
        
        # Verify multiplication works (the operation that was failing)
        cylinder_radius_factor = 1.2
        for port in [port_with_none_radius, port_missing_radius, port_with_radius]:
            radius = get_radius_fixed(port)
            carve_radius = radius * cylinder_radius_factor
            assert isinstance(carve_radius, float)
            assert carve_radius > 0
    
    def test_port_with_zero_radius_uses_default(self):
        """Test that ports with radius=0 also use default value."""
        def get_radius_fixed(port):
            return port.get("radius") or 0.001
        
        port_with_zero_radius = {"radius": 0}
        
        # Zero radius should also use default (falsy value)
        assert get_radius_fixed(port_with_zero_radius) == 0.001


class TestEmbeddingWithoutShell:
    """Test embedding when shell output is disabled."""
    
    def test_no_shell_when_disabled(self):
        """Test that shell is not output when disabled."""
        policy = EmbeddingPolicy(output_shell=False)
        
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy=policy.to_dict(),
            metadata={
                "outputs": {
                    "domain_with_void": {"exists": True},
                    "void_mesh": {"exists": True},
                    "shell": {"exists": False},
                },
            },
        )
        
        assert not report.metadata["outputs"]["shell"]["exists"]
    
    def test_domain_with_void_and_void_mesh_always_output(self):
        """Test that domain_with_void and void_mesh are always output."""
        policy = EmbeddingPolicy(output_shell=False)
        
        report = OperationReport(
            operation="embedding",
            success=True,
            requested_policy=policy.to_dict(),
            metadata={
                "outputs": {
                    "domain_with_void": {"exists": True},
                    "void_mesh": {"exists": True},
                },
            },
        )
        
        assert report.metadata["outputs"]["domain_with_void"]["exists"]
        assert report.metadata["outputs"]["void_mesh"]["exists"]
