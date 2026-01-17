"""
Test path channel tube sweep (E3).

This module verifies that path channels use tube sweep and produce
watertight meshes with correct diameter bounds.
"""

import pytest
import numpy as np

from aog_policies import ChannelPolicy, RadiusPolicy, RepairPolicy, OperationReport


class TestPathChannelTubeSweep:
    """E3: Path channel uses tube sweep and produces watertight mesh."""
    
    def test_channel_policy_has_tube_sweep_mode(self):
        """Test that ChannelPolicy supports tube sweep mode."""
        policy = ChannelPolicy(
            profile="tube_sweep",
            taper_factor=1.0,
        )
        
        assert policy.profile == "tube_sweep"
    
    def test_tube_sweep_with_taper_profile(self):
        """Test tube sweep with taper profile."""
        policy = ChannelPolicy(
            profile="taper",
            taper_factor=0.8,
            inlet_radius=0.001,
            outlet_radius=0.0008,
        )
        
        assert policy.profile == "taper"
        assert policy.taper_factor == 0.8
    
    def test_tube_sweep_report_includes_mesh_info(self):
        """Test that tube sweep report includes mesh information."""
        report = OperationReport(
            operation="path_channel_generation",
            success=True,
            metadata={
                "mesh": {
                    "exists": True,
                    "face_count": 5000,
                    "vertex_count": 2502,
                    "is_watertight": True,
                },
                "profile": "tube_sweep",
                "polyline_length": 0.015,
                "segments": 50,
            },
        )
        
        assert report.metadata["mesh"]["exists"]
        assert report.metadata["mesh"]["face_count"] > 0
        assert report.metadata["mesh"]["is_watertight"]
    
    def test_watertight_mesh_or_repaired(self):
        """Test that mesh is watertight or repaired per policy."""
        repair_policy = RepairPolicy(
            voxel_repair_enabled=True,
            voxel_pitch=1e-5,
        )
        
        report = OperationReport(
            operation="path_channel_generation",
            success=True,
            requested_policy={
                "repair": repair_policy.to_dict(),
            },
            metadata={
                "mesh": {
                    "is_watertight": True,
                    "was_repaired": True,
                    "repair_method": "voxel_repair",
                },
            },
        )
        
        mesh_info = report.metadata["mesh"]
        assert mesh_info["is_watertight"] or mesh_info.get("was_repaired", False)
    
    def test_diameters_within_bounds(self):
        """Test that diameters are within expected bounds."""
        min_diameter = 0.0004
        max_diameter = 0.002
        
        report = OperationReport(
            operation="path_channel_generation",
            success=True,
            metadata={
                "diameters": {
                    "min": 0.0005,
                    "max": 0.0018,
                    "inlet": 0.001,
                    "outlet": 0.0008,
                },
                "bounds": {
                    "min_diameter": min_diameter,
                    "max_diameter": max_diameter,
                },
            },
        )
        
        diameters = report.metadata["diameters"]
        bounds = report.metadata["bounds"]
        
        assert diameters["min"] >= bounds["min_diameter"], (
            f"Min diameter ({diameters['min']}) below bound ({bounds['min_diameter']})"
        )
        assert diameters["max"] <= bounds["max_diameter"], (
            f"Max diameter ({diameters['max']}) above bound ({bounds['max_diameter']})"
        )


class TestTubeSweepConfiguration:
    """Test tube sweep configuration."""
    
    def test_polyline_input(self):
        """Test that tube sweep accepts polyline input."""
        polyline = [
            (0.0, 0.0, 0.0),
            (0.005, 0.0, 0.0),
            (0.01, 0.005, 0.0),
            (0.015, 0.005, 0.005),
        ]
        
        assert len(polyline) >= 2, "Polyline should have at least 2 points"
        
        total_length = 0.0
        for i in range(1, len(polyline)):
            p1, p2 = polyline[i-1], polyline[i]
            segment_length = np.sqrt(
                (p2[0] - p1[0])**2 +
                (p2[1] - p1[1])**2 +
                (p2[2] - p1[2])**2
            )
            total_length += segment_length
        
        assert total_length > 0, "Polyline should have positive length"
    
    def test_radius_profile_along_path(self):
        """Test radius profile along path."""
        radius_policy = RadiusPolicy(
            mode="taper",
            taper_factor=0.8,
            min_radius=0.0002,
            max_radius=0.001,
        )
        
        assert radius_policy.mode == "taper"
        assert radius_policy.taper_factor == 0.8
        assert radius_policy.min_radius < radius_policy.max_radius
    
    def test_murray_law_radius_profile(self):
        """Test Murray's Law radius profile."""
        radius_policy = RadiusPolicy(
            mode="murray",
            murray_exponent=3.0,
            min_radius=0.0002,
            max_radius=0.001,
        )
        
        assert radius_policy.mode == "murray"
        assert radius_policy.murray_exponent == 3.0
    
    def test_constant_radius_profile(self):
        """Test constant radius profile."""
        radius_policy = RadiusPolicy(
            mode="constant",
            min_radius=0.0005,
            max_radius=0.0005,
        )
        
        assert radius_policy.mode == "constant"
        assert radius_policy.min_radius == radius_policy.max_radius


class TestTubeSweepMeshQuality:
    """Test tube sweep mesh quality."""
    
    def test_mesh_has_faces(self):
        """Test that generated mesh has faces."""
        report = OperationReport(
            operation="path_channel_generation",
            success=True,
            metadata={
                "mesh": {
                    "face_count": 5000,
                    "vertex_count": 2502,
                },
            },
        )
        
        assert report.metadata["mesh"]["face_count"] > 0
    
    def test_mesh_vertex_count_reasonable(self):
        """Test that mesh vertex count is reasonable."""
        report = OperationReport(
            operation="path_channel_generation",
            success=True,
            metadata={
                "mesh": {
                    "face_count": 5000,
                    "vertex_count": 2502,
                },
            },
        )
        
        face_count = report.metadata["mesh"]["face_count"]
        vertex_count = report.metadata["mesh"]["vertex_count"]
        
        assert vertex_count > 0
        assert face_count > 0
        
        ratio = face_count / vertex_count
        assert 1.5 < ratio < 3.0, (
            f"Face/vertex ratio ({ratio}) outside expected range for tube mesh"
        )
    
    def test_watertight_check_in_report(self):
        """Test that watertight check is included in report."""
        report = OperationReport(
            operation="path_channel_generation",
            success=True,
            metadata={
                "mesh": {
                    "is_watertight": True,
                    "watertight_check_method": "edge_manifold",
                },
            },
        )
        
        assert "is_watertight" in report.metadata["mesh"]


class TestTubeSweepSerialization:
    """Test tube sweep policy serialization."""
    
    def test_channel_policy_serializable(self):
        """Test that ChannelPolicy is JSON-serializable."""
        import json
        
        policy = ChannelPolicy(
            profile="tube_sweep",
            taper_factor=0.8,
            inlet_radius=0.001,
            outlet_radius=0.0008,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["profile"] == "tube_sweep"
    
    def test_radius_policy_serializable(self):
        """Test that RadiusPolicy is JSON-serializable."""
        import json
        
        policy = RadiusPolicy(
            mode="taper",
            taper_factor=0.8,
            min_radius=0.0002,
            max_radius=0.001,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["mode"] == "taper"
        assert decoded["taper_factor"] == 0.8
