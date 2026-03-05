"""
Test policy serialization and JSON compliance (A1, A2).

This module verifies that all policy classes:
- Round-trip through JSON serialization without data loss
- Contain no callables, lambdas, or closures
- Are fully JSON-serializable (no numpy types, no functions)
"""

import json
import pytest
import inspect
from dataclasses import fields, is_dataclass
from typing import Any, get_type_hints

import aog_policies
from aog_policies import (
    OperationReport,
    PortPlacementPolicy,
    ChannelPolicy,
    GrowthPolicy,
    TissueSamplingPolicy,
    CollisionPolicy,
    NetworkCleanupPolicy,
    MeshSynthesisPolicy,
    MeshMergePolicy,
    EmbeddingPolicy,
    OutputPolicy,
    ValidationPolicy,
    RepairPolicy,
    OpenPortPolicy,
    PitchLimits,
    ResolutionPolicy,
    PathfindingPolicy,
    WaypointPolicy,
    HierarchicalPathfindingPolicy,
    ComposePolicy,
    PrimitiveMeshingPolicy,
    MeshDomainPolicy,
    ImplicitMeshingPolicy,
    DomainMeshingPolicy,
    UnifiedCollisionPolicy,
    RadiusPolicy,
    RetryPolicy,
    RidgePolicy,
    PortPreservationPolicy,
)


ALL_POLICY_CLASSES = [
    PortPlacementPolicy,
    ChannelPolicy,
    GrowthPolicy,
    TissueSamplingPolicy,
    CollisionPolicy,
    NetworkCleanupPolicy,
    MeshSynthesisPolicy,
    MeshMergePolicy,
    EmbeddingPolicy,
    OutputPolicy,
    ValidationPolicy,
    RepairPolicy,
    OpenPortPolicy,
    PitchLimits,
    ResolutionPolicy,
    PathfindingPolicy,
    WaypointPolicy,
    HierarchicalPathfindingPolicy,
    ComposePolicy,
    PrimitiveMeshingPolicy,
    MeshDomainPolicy,
    ImplicitMeshingPolicy,
    DomainMeshingPolicy,
    UnifiedCollisionPolicy,
    RadiusPolicy,
    RetryPolicy,
    RidgePolicy,
    PortPreservationPolicy,
    OperationReport,
]


def create_policy_with_non_defaults(policy_class):
    """Create a policy instance with non-default values where possible."""
    if policy_class == PortPlacementPolicy:
        return PortPlacementPolicy(
            enabled=False,
            face="bottom",
            pattern="grid",
            ridge_width=0.0002,
            ridge_clearance=0.0002,
            port_margin=0.001,
        )
    elif policy_class == ChannelPolicy:
        return ChannelPolicy(
            enabled=False,
            profile="taper",
            length_mode="to_depth",
            length=0.01,
            taper_factor=0.7,
            hook_depth=0.002,
        )
    elif policy_class == GrowthPolicy:
        return GrowthPolicy(
            enabled=False,
            backend="cco_hybrid",
            target_terminals=100,
            max_iterations=1000,
            seed=42,
        )
    elif policy_class == TissueSamplingPolicy:
        return TissueSamplingPolicy(
            enabled=False,
            n_points=500,
            seed=123,
            strategy="depth_biased",
        )
    elif policy_class == CollisionPolicy:
        return CollisionPolicy(
            enabled=False,
            check_collisions=False,
            collision_clearance=0.0005,
        )
    elif policy_class == NetworkCleanupPolicy:
        return NetworkCleanupPolicy(
            enable_snap=False,
            snap_tol=0.0002,
            enable_prune=False,
        )
    elif policy_class == MeshSynthesisPolicy:
        return MeshSynthesisPolicy(
            add_node_spheres=False,
            cap_ends=False,
            voxel_repair_synthesis=True,
            voxel_repair_pitch=2e-4,
        )
    elif policy_class == MeshMergePolicy:
        return MeshMergePolicy(
            mode="boolean",
            voxel_pitch=1e-4,
            auto_adjust_pitch=False,
        )
    elif policy_class == EmbeddingPolicy:
        return EmbeddingPolicy(
            preserve_ports_enabled=False,
            max_voxels=1_000_000,
        )
    elif policy_class == OutputPolicy:
        return OutputPolicy(
            output_stl=False,
            output_json=False,
        )
    elif policy_class == ValidationPolicy:
        return ValidationPolicy(
            check_watertight=False,
            check_components=False,
            max_components=5,
        )
    elif policy_class == RepairPolicy:
        return RepairPolicy(
            voxel_repair_enabled=False,
            voxel_pitch=2e-4,
            fill_voxels=False,
        )
    elif policy_class == OpenPortPolicy:
        return OpenPortPolicy(
            enabled=False,
            probe_radius_factor=1.5,
            max_voxels_roi=500_000,
        )
    elif policy_class == PitchLimits:
        return PitchLimits(
            min_pitch=2e-6,
            max_pitch=2e-3,
        )
    elif policy_class == ResolutionPolicy:
        return ResolutionPolicy(
            input_units="mm",
            min_channel_diameter=4e-5,
            voxels_across_min_diameter=16,
            max_voxels=50_000_000,
        )
    elif policy_class == PathfindingPolicy:
        return PathfindingPolicy(
            voxel_pitch=0.001,
            clearance=0.0005,
            max_nodes=50000,
            timeout_s=60.0,
        )
    elif policy_class == WaypointPolicy:
        return WaypointPolicy(
            skip_unreachable=False,
            max_skip_count=5,
            emit_warnings=False,
        )
    elif policy_class == HierarchicalPathfindingPolicy:
        return HierarchicalPathfindingPolicy(
            pitch_coarse=0.0002,
            pitch_fine=0.00001,
            max_voxels_coarse=5_000_000,
            max_voxels_fine=25_000_000,
        )
    elif policy_class == ComposePolicy:
        return ComposePolicy(
            repair_enabled=False,
            repair_voxel_pitch=1e-4,
        )
    elif policy_class == PrimitiveMeshingPolicy:
        return PrimitiveMeshingPolicy(
            sections_radial=64,
            sections_axial=32,
        )
    elif policy_class == MeshDomainPolicy:
        return MeshDomainPolicy(
            validate_watertight=False,
            repair_if_needed=False,
            max_faces=1_000_000,
        )
    elif policy_class == ImplicitMeshingPolicy:
        return ImplicitMeshingPolicy(
            voxel_pitch=1e-4,
            max_voxels=25_000_000,
            smooth_iterations=5,
        )
    elif policy_class == DomainMeshingPolicy:
        return DomainMeshingPolicy(
            cache_meshes=False,
            emit_warnings=False,
        )
    elif policy_class == UnifiedCollisionPolicy:
        return UnifiedCollisionPolicy(
            enabled=False,
            min_clearance=0.0005,
            min_radius=0.0002,
        )
    elif policy_class == RadiusPolicy:
        return RadiusPolicy(
            mode="taper",
            murray_exponent=2.5,
            taper_factor=0.7,
        )
    elif policy_class == RetryPolicy:
        return RetryPolicy(
            max_retries=5,
            backoff_factor=2.0,
        )
    elif policy_class == RidgePolicy:
        return RidgePolicy(
            height=0.002,
            thickness=0.002,
            resolution=128,
        )
    elif policy_class == PortPreservationPolicy:
        return PortPreservationPolicy(
            enabled=False,
            cylinder_radius_factor=1.5,
            cylinder_depth=0.003,
        )
    elif policy_class == OperationReport:
        return OperationReport(
            operation="test_operation",
            success=False,
            requested_policy={"key": "value"},
            effective_policy={"key": "modified_value"},
            warnings=["test warning"],
            errors=["test error"],
            metadata={"metric": 123},
        )
    else:
        return policy_class()


def is_callable_value(value: Any) -> bool:
    """Check if a value is callable (function, lambda, method, class)."""
    if callable(value) and not isinstance(value, type):
        return True
    if inspect.isfunction(value) or inspect.ismethod(value):
        return True
    if inspect.isclass(value):
        return True
    return False


def walk_dict_for_callables(d: dict, path: str = "") -> list:
    """Recursively walk a dict and find any callable values."""
    callables_found = []
    
    for key, value in d.items():
        current_path = f"{path}.{key}" if path else key
        
        if is_callable_value(value):
            callables_found.append((current_path, type(value).__name__))
        elif isinstance(value, dict):
            callables_found.extend(walk_dict_for_callables(value, current_path))
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if is_callable_value(item):
                    callables_found.append((f"{current_path}[{i}]", type(item).__name__))
                elif isinstance(item, dict):
                    callables_found.extend(walk_dict_for_callables(item, f"{current_path}[{i}]"))
    
    return callables_found


def walk_object_for_callables(obj: Any, path: str = "", visited: set = None) -> list:
    """Recursively walk an object and find any callable field values."""
    if visited is None:
        visited = set()
    
    obj_id = id(obj)
    if obj_id in visited:
        return []
    visited.add(obj_id)
    
    callables_found = []
    
    if is_dataclass(obj):
        for f in fields(obj):
            value = getattr(obj, f.name)
            current_path = f"{path}.{f.name}" if path else f.name
            
            if is_callable_value(value):
                callables_found.append((current_path, type(value).__name__))
            elif is_dataclass(value):
                callables_found.extend(walk_object_for_callables(value, current_path, visited))
            elif isinstance(value, dict):
                callables_found.extend(walk_dict_for_callables(value, current_path))
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if is_callable_value(item):
                        callables_found.append((f"{current_path}[{i}]", type(item).__name__))
                    elif is_dataclass(item):
                        callables_found.extend(walk_object_for_callables(item, f"{current_path}[{i}]", visited))
    
    return callables_found


class TestPolicyJSONRoundTrip:
    """A1: All policies JSON round-trip without data loss."""
    
    @pytest.mark.parametrize("policy_class", ALL_POLICY_CLASSES)
    def test_policy_round_trip(self, policy_class):
        """Test that policy can be serialized to dict, JSON encoded/decoded, and reconstructed."""
        policy = create_policy_with_non_defaults(policy_class)
        
        policy_dict = policy.to_dict()
        
        json_str = json.dumps(policy_dict)
        assert json_str is not None, f"{policy_class.__name__} failed JSON encoding"
        
        decoded_dict = json.loads(json_str)
        assert decoded_dict == policy_dict, f"{policy_class.__name__} JSON decode mismatch"
        
        if hasattr(policy_class, 'from_dict'):
            reconstructed = policy_class.from_dict(decoded_dict)
            
            reconstructed_dict = reconstructed.to_dict()
            
            for key in policy_dict:
                if key in reconstructed_dict:
                    original_val = policy_dict[key]
                    reconstructed_val = reconstructed_dict[key]
                    
                    if isinstance(original_val, float):
                        assert abs(original_val - reconstructed_val) < 1e-10, (
                            f"{policy_class.__name__}.{key}: {original_val} != {reconstructed_val}"
                        )
                    else:
                        assert original_val == reconstructed_val, (
                            f"{policy_class.__name__}.{key}: {original_val} != {reconstructed_val}"
                        )
    
    @pytest.mark.parametrize("policy_class", ALL_POLICY_CLASSES)
    def test_policy_json_dumps_succeeds(self, policy_class):
        """Test that json.dumps succeeds on policy dict (no callables, no numpy types)."""
        policy = create_policy_with_non_defaults(policy_class)
        policy_dict = policy.to_dict()
        
        try:
            json_str = json.dumps(policy_dict)
            assert isinstance(json_str, str)
        except TypeError as e:
            pytest.fail(f"{policy_class.__name__} contains non-JSON-serializable type: {e}")


class TestNoCallablesInPolicies:
    """A2: No callables in policies guard."""
    
    @pytest.mark.parametrize("policy_class", ALL_POLICY_CLASSES)
    def test_no_callable_fields(self, policy_class):
        """Test that no field value is callable."""
        policy = create_policy_with_non_defaults(policy_class)
        
        callables = walk_object_for_callables(policy)
        
        assert len(callables) == 0, (
            f"{policy_class.__name__} contains callable fields: {callables}"
        )
    
    @pytest.mark.parametrize("policy_class", ALL_POLICY_CLASSES)
    def test_no_callable_in_dict(self, policy_class):
        """Test that serialized dict contains no callables."""
        policy = create_policy_with_non_defaults(policy_class)
        policy_dict = policy.to_dict()
        
        callables = walk_dict_for_callables(policy_dict)
        
        assert len(callables) == 0, (
            f"{policy_class.__name__}.to_dict() contains callable values: {callables}"
        )
    
    @pytest.mark.parametrize("policy_class", ALL_POLICY_CLASSES)
    def test_no_function_or_class_types(self, policy_class):
        """Test that no field is a function or class type."""
        policy = create_policy_with_non_defaults(policy_class)
        
        if is_dataclass(policy):
            for f in fields(policy):
                value = getattr(policy, f.name)
                
                assert not inspect.isfunction(value), (
                    f"{policy_class.__name__}.{f.name} is a function"
                )
                assert not inspect.isclass(value), (
                    f"{policy_class.__name__}.{f.name} is a class"
                )
                
                if callable(value) and not isinstance(value, (type, bool, int, float, str, list, dict, tuple)):
                    if not is_dataclass(value):
                        pytest.fail(f"{policy_class.__name__}.{f.name} is callable: {type(value)}")


class TestPolicyDefaultsAreValid:
    """Test that default policy values are valid and JSON-serializable."""
    
    @pytest.mark.parametrize("policy_class", ALL_POLICY_CLASSES)
    def test_default_policy_serializable(self, policy_class):
        """Test that default policy is JSON-serializable."""
        policy = policy_class()
        policy_dict = policy.to_dict()
        
        try:
            json_str = json.dumps(policy_dict)
            assert isinstance(json_str, str)
        except TypeError as e:
            pytest.fail(f"{policy_class.__name__} default contains non-JSON-serializable type: {e}")
    
    @pytest.mark.parametrize("policy_class", ALL_POLICY_CLASSES)
    def test_default_policy_round_trip(self, policy_class):
        """Test that default policy round-trips correctly."""
        policy = policy_class()
        policy_dict = policy.to_dict()
        
        json_str = json.dumps(policy_dict)
        decoded_dict = json.loads(json_str)
        
        if hasattr(policy_class, 'from_dict'):
            reconstructed = policy_class.from_dict(decoded_dict)
            assert reconstructed is not None
