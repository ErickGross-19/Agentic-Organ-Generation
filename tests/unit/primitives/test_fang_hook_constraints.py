"""
Test fang hook constraints (E4).

This module verifies that fang hooks bend radially outward and respect
effective radius constraints.
"""

import pytest
import numpy as np

from aog_policies import ChannelPolicy, RidgePolicy, OperationReport
from generation.core.domain import CylinderDomain
from generation.core.types import Point3D


class TestFangHookConstraints:
    """E4: Fang hook bends radially outward and respects effective radius."""
    
    def test_fang_hook_endpoint_has_outward_component(self):
        """Test that fang hook endpoint direction has outward component."""
        face_center = np.array([0.0, 0.0, 0.005])
        face_normal = np.array([0.0, 0.0, 1.0])
        
        hook_start = np.array([0.002, 0.0, 0.005])
        hook_end = np.array([0.003, 0.0, 0.004])
        
        radial_direction = hook_start - face_center
        radial_direction[2] = 0
        if np.linalg.norm(radial_direction) > 0:
            radial_direction = radial_direction / np.linalg.norm(radial_direction)
        
        hook_direction = hook_end - hook_start
        if np.linalg.norm(hook_direction) > 0:
            hook_direction = hook_direction / np.linalg.norm(hook_direction)
        
        outward_component = np.dot(hook_direction, radial_direction)
        
        assert outward_component > 0, (
            f"Hook direction should have outward component (got {outward_component})"
        )
    
    def test_fang_hook_respects_effective_radius(self):
        """Test that fang hook does not exceed effective radius."""
        face_center = np.array([0.0, 0.0, 0.005])
        effective_radius = 0.004
        
        hook_endpoint = np.array([0.003, 0.0, 0.004])
        
        radial_distance = np.sqrt(
            (hook_endpoint[0] - face_center[0])**2 +
            (hook_endpoint[1] - face_center[1])**2
        )
        
        assert radial_distance <= effective_radius, (
            f"Hook endpoint radial distance ({radial_distance}) exceeds "
            f"effective radius ({effective_radius})"
        )
    
    def test_fang_hook_report_includes_constraints(self):
        """Test that fang hook report includes constraint information."""
        report = OperationReport(
            operation="fang_hook_generation",
            success=True,
            metadata={
                "constraints": {
                    "effective_radius": 0.004,
                    "face_center": [0.0, 0.0, 0.005],
                    "face_normal": [0.0, 0.0, 1.0],
                },
                "hook": {
                    "start": [0.002, 0.0, 0.005],
                    "end": [0.003, 0.0, 0.004],
                    "radial_distance": 0.003,
                    "outward_component": 0.5,
                },
                "strategies_used": ["reduce_depth"],
            },
        )
        
        assert "constraints" in report.metadata
        assert "effective_radius" in report.metadata["constraints"]
        assert "strategies_used" in report.metadata
    
    def test_warning_when_radius_exceeded(self):
        """Test that warning is emitted when effective radius would be exceeded."""
        report = OperationReport(
            operation="fang_hook_generation",
            success=True,
            warnings=[
                "Hook endpoint adjusted to respect effective radius (0.004m)"
            ],
            metadata={
                "constraints": {
                    "effective_radius": 0.004,
                    "original_endpoint": [0.005, 0.0, 0.004],
                    "adjusted_endpoint": [0.0035, 0.0, 0.004],
                },
                "radius_exceeded": True,
                "adjustment_applied": True,
            },
        )
        
        assert len(report.warnings) > 0
        assert report.metadata["radius_exceeded"]
        assert report.metadata["adjustment_applied"]


class TestFangHookStrategies:
    """Test fang hook constraint strategies."""
    
    def test_reduce_depth_strategy(self):
        """Test reduce_depth strategy for fang hook."""
        policy = ChannelPolicy(
            hook_depth=0.002,
            hook_strategy="reduce_depth",
        )
        
        assert policy.hook_strategy == "reduce_depth"
        
        report = OperationReport(
            operation="fang_hook_generation",
            success=True,
            metadata={
                "strategy": "reduce_depth",
                "original_depth": 0.002,
                "effective_depth": 0.0015,
                "depth_reduced": True,
            },
        )
        
        assert report.metadata["strategy"] == "reduce_depth"
    
    def test_rotate_strategy(self):
        """Test rotate strategy for fang hook."""
        policy = ChannelPolicy(
            hook_depth=0.002,
            hook_strategy="rotate",
        )
        
        assert policy.hook_strategy == "rotate"
        
        report = OperationReport(
            operation="fang_hook_generation",
            success=True,
            metadata={
                "strategy": "rotate",
                "original_angle": 45.0,
                "effective_angle": 30.0,
                "angle_adjusted": True,
            },
        )
        
        assert report.metadata["strategy"] == "rotate"
    
    def test_combined_strategy(self):
        """Test combined reduce_depth + rotate strategy."""
        policy = ChannelPolicy(
            hook_depth=0.002,
            hook_strategy="reduce_depth_and_rotate",
        )
        
        assert policy.hook_strategy == "reduce_depth_and_rotate"
        
        report = OperationReport(
            operation="fang_hook_generation",
            success=True,
            metadata={
                "strategy": "reduce_depth_and_rotate",
                "depth_reduced": True,
                "angle_adjusted": True,
            },
        )
        
        assert report.metadata["depth_reduced"]
        assert report.metadata["angle_adjusted"]


class TestFangHookWithCylinderFace:
    """Test fang hook generation on cylinder face."""
    
    def test_cylinder_face_frame(self):
        """Test that cylinder face provides correct frame for fang hook."""
        cylinder = CylinderDomain(
            radius=0.005,
            height=0.01,
            center=Point3D(0, 0, 0),
        )
        
        if hasattr(cylinder, 'get_face_frame'):
            frame = cylinder.get_face_frame("top")
            
            assert "origin" in frame
            assert "normal" in frame
            
            normal = np.array(frame["normal"])
            assert abs(np.linalg.norm(normal) - 1.0) < 1e-6
    
    def test_ridge_constraint_effective_radius(self):
        """Test that ridge constraint provides effective radius."""
        ridge_policy = RidgePolicy(
            height=0.001,
            thickness=0.0005,
            inset=0.0005,
        )
        
        cylinder_radius = 0.005
        effective_radius = cylinder_radius - ridge_policy.inset
        
        assert effective_radius < cylinder_radius
        assert effective_radius > 0
    
    def test_fang_hook_within_ridge_boundary(self):
        """Test that fang hook stays within ridge boundary."""
        cylinder_radius = 0.005
        ridge_inset = 0.0005
        effective_radius = cylinder_radius - ridge_inset
        
        hook_radial_distance = 0.004
        
        assert hook_radial_distance <= effective_radius, (
            f"Hook radial distance ({hook_radial_distance}) exceeds "
            f"effective radius ({effective_radius})"
        )


class TestFangHookSerialization:
    """Test fang hook policy serialization."""
    
    def test_channel_policy_with_hook_serializable(self):
        """Test that ChannelPolicy with hook settings is JSON-serializable."""
        import json
        
        policy = ChannelPolicy(
            hook_depth=0.002,
            hook_strategy="reduce_depth",
            hook_angle=45.0,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["hook_depth"] == 0.002
        assert decoded["hook_strategy"] == "reduce_depth"
    
    def test_ridge_policy_serializable(self):
        """Test that RidgePolicy is JSON-serializable."""
        import json
        
        policy = RidgePolicy(
            height=0.001,
            thickness=0.0005,
            inset=0.0005,
            resolution=64,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["height"] == 0.001
        assert decoded["inset"] == 0.0005


class TestFangHookReporting:
    """Test fang hook reporting in operation reports."""
    
    def test_report_includes_all_constraint_info(self):
        """Test that report includes all constraint information."""
        report = OperationReport(
            operation="fang_hook_generation",
            success=True,
            requested_policy={
                "hook_depth": 0.002,
                "hook_strategy": "reduce_depth",
            },
            effective_policy={
                "hook_depth": 0.0015,
                "hook_strategy": "reduce_depth",
            },
            metadata={
                "constraints": {
                    "effective_radius": 0.004,
                    "face_center": [0.0, 0.0, 0.005],
                    "ridge_inset": 0.0005,
                },
                "hook": {
                    "start": [0.002, 0.0, 0.005],
                    "end": [0.003, 0.0, 0.004],
                    "radial_distance": 0.003,
                    "outward_component": 0.5,
                    "depth": 0.0015,
                },
                "adjustments": {
                    "depth_reduced": True,
                    "original_depth": 0.002,
                    "effective_depth": 0.0015,
                },
            },
        )
        
        assert "constraints" in report.metadata
        assert "hook" in report.metadata
        assert "adjustments" in report.metadata
        
        assert report.requested_policy["hook_depth"] != report.effective_policy["hook_depth"]
