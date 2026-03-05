"""
Test port connectivity fixes.

This module tests the port connectivity improvements:
1. Adaptive pitch selection based on port radius
2. Port typing validation and warnings
3. Port recarving with connectivity verification and iterative extension
"""

import pytest
import numpy as np

from aog_policies import OpenPortPolicy, ResolutionPolicy


class TestAdaptivePitchSelection:
    """Test adaptive pitch selection based on port radius."""
    
    def test_open_port_policy_has_adaptive_pitch_fields(self):
        """Test that OpenPortPolicy has adaptive pitch configuration fields."""
        policy = OpenPortPolicy(
            enabled=True,
            min_voxels_across_radius=8,
            adaptive_pitch=True,
        )
        
        assert hasattr(policy, 'min_voxels_across_radius')
        assert hasattr(policy, 'adaptive_pitch')
        assert policy.min_voxels_across_radius == 8
        assert policy.adaptive_pitch is True
    
    def test_compute_adaptive_pitch_basic(self):
        """Test basic adaptive pitch computation."""
        policy = OpenPortPolicy(
            enabled=True,
            min_voxels_across_radius=8,
            adaptive_pitch=True,
        )
        
        port_radius = 0.0005  # 0.5mm
        expected_pitch = port_radius / 8  # 0.0000625m = 62.5µm
        
        computed_pitch = policy.compute_adaptive_pitch(port_radius)
        assert abs(computed_pitch - expected_pitch) < 1e-10
    
    def test_compute_adaptive_pitch_respects_validation_pitch_override(self):
        """Test that explicit validation_pitch overrides adaptive computation."""
        policy = OpenPortPolicy(
            enabled=True,
            min_voxels_across_radius=8,
            adaptive_pitch=True,
            validation_pitch=0.0001,  # Explicit override
        )
        
        port_radius = 0.0005
        computed_pitch = policy.compute_adaptive_pitch(port_radius)
        
        assert computed_pitch == 0.0001
    
    def test_compute_adaptive_pitch_disabled(self):
        """Test pitch computation when adaptive_pitch is disabled."""
        policy = OpenPortPolicy(
            enabled=True,
            min_voxels_across_radius=8,
            adaptive_pitch=False,
        )
        
        port_radius = 0.0005
        computed_pitch = policy.compute_adaptive_pitch(port_radius)
        
        # Should fall back to port_radius / 4
        expected_pitch = port_radius / 4
        assert abs(computed_pitch - expected_pitch) < 1e-10
    
    def test_adaptive_pitch_ensures_sufficient_voxels(self):
        """Test that adaptive pitch ensures sufficient voxels across radius."""
        policy = OpenPortPolicy(
            enabled=True,
            min_voxels_across_radius=10,
            adaptive_pitch=True,
        )
        
        port_radius = 0.001  # 1mm
        computed_pitch = policy.compute_adaptive_pitch(port_radius)
        
        # Verify we get at least min_voxels_across_radius voxels
        voxels_across = port_radius / computed_pitch
        assert voxels_across >= policy.min_voxels_across_radius
    
    def test_default_min_voxels_across_radius(self):
        """Test default value for min_voxels_across_radius."""
        policy = OpenPortPolicy(enabled=True)
        
        # Default should be 8 (6-10 recommended range)
        assert policy.min_voxels_across_radius == 8


class TestPortTypingValidation:
    """Test port typing validation and warnings."""
    
    def test_open_port_policy_has_require_port_type(self):
        """Test that OpenPortPolicy has require_port_type field."""
        policy = OpenPortPolicy(
            enabled=True,
            require_port_type=True,
        )
        
        assert hasattr(policy, 'require_port_type')
        assert policy.require_port_type is True
    
    def test_require_port_type_default_false(self):
        """Test that require_port_type defaults to False."""
        policy = OpenPortPolicy(enabled=True)
        
        assert policy.require_port_type is False
    
    def test_warn_on_pitch_relaxation_field(self):
        """Test that OpenPortPolicy has warn_on_pitch_relaxation field."""
        policy = OpenPortPolicy(
            enabled=True,
            warn_on_pitch_relaxation=True,
        )
        
        assert hasattr(policy, 'warn_on_pitch_relaxation')
        assert policy.warn_on_pitch_relaxation is True


class TestOpenPortPolicyDefaults:
    """Test OpenPortPolicy default values for port connectivity fixes."""
    
    def test_increased_max_voxels_roi(self):
        """Test that max_voxels_roi is increased from 1M to 2M."""
        policy = OpenPortPolicy(enabled=True)
        
        # Should be 2M (increased from 1M to reduce pitch coarsening)
        assert policy.max_voxels_roi == 2_000_000
    
    def test_reduced_local_region_size(self):
        """Test that local_region_size is reduced from 5mm to 4mm."""
        policy = OpenPortPolicy(enabled=True)
        
        # Should be 0.004m (4mm, reduced from 5mm for finer pitch)
        assert policy.local_region_size == 0.004
    
    def test_adaptive_pitch_enabled_by_default(self):
        """Test that adaptive_pitch is enabled by default."""
        policy = OpenPortPolicy(enabled=True)
        
        assert policy.adaptive_pitch is True
    
    def test_warn_on_pitch_relaxation_enabled_by_default(self):
        """Test that warn_on_pitch_relaxation is enabled by default."""
        policy = OpenPortPolicy(enabled=True)
        
        assert policy.warn_on_pitch_relaxation is True


class TestOpenPortPolicySerialization:
    """Test OpenPortPolicy serialization with new fields."""
    
    def test_to_dict_includes_new_fields(self):
        """Test that to_dict includes all new port connectivity fields."""
        policy = OpenPortPolicy(
            enabled=True,
            min_voxels_across_radius=10,
            adaptive_pitch=True,
            warn_on_pitch_relaxation=True,
            require_port_type=True,
        )
        
        policy_dict = policy.to_dict()
        
        assert 'min_voxels_across_radius' in policy_dict
        assert 'adaptive_pitch' in policy_dict
        assert 'warn_on_pitch_relaxation' in policy_dict
        assert 'require_port_type' in policy_dict
        
        assert policy_dict['min_voxels_across_radius'] == 10
        assert policy_dict['adaptive_pitch'] is True
    
    def test_json_serializable(self):
        """Test that OpenPortPolicy with new fields is JSON-serializable."""
        import json
        
        policy = OpenPortPolicy(
            enabled=True,
            min_voxels_across_radius=8,
            adaptive_pitch=True,
            warn_on_pitch_relaxation=True,
            require_port_type=False,
            max_voxels_roi=2_000_000,
            local_region_size=0.004,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded['min_voxels_across_radius'] == 8
        assert decoded['max_voxels_roi'] == 2_000_000


class TestPortRecarveConnectivityVerification:
    """Test port recarve connectivity verification parameters."""
    
    @pytest.fixture
    def voxel_recarve_ports_func(self):
        """Import voxel_recarve_ports, skipping if dependencies unavailable."""
        pytest.importorskip("skimage")
        from generation.ops.embedding.enhanced_embedding import voxel_recarve_ports
        return voxel_recarve_ports
    
    def test_voxel_recarve_ports_signature(self, voxel_recarve_ports_func):
        """Test that voxel_recarve_ports has connectivity verification parameters."""
        import inspect
        
        sig = inspect.signature(voxel_recarve_ports_func)
        params = sig.parameters
        
        assert 'verify_void_connectivity' in params
        assert 'max_extension_iterations' in params
        assert 'extension_step_factor' in params
        assert 'min_overlap_voxels' in params
    
    def test_voxel_recarve_ports_default_values(self, voxel_recarve_ports_func):
        """Test default values for connectivity verification parameters."""
        import inspect
        
        sig = inspect.signature(voxel_recarve_ports_func)
        params = sig.parameters
        
        assert params['verify_void_connectivity'].default is True
        assert params['max_extension_iterations'].default == 5
        assert params['extension_step_factor'].default == 0.5
        assert params['min_overlap_voxels'].default == 5


class TestPortRecarveWithSimpleMesh:
    """Test port recarve with a simple mesh."""
    
    @pytest.fixture
    def simple_cube_mesh(self):
        """Create a simple cube mesh for testing."""
        import trimesh
        
        # Create a 10mm cube centered at origin
        mesh = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        return mesh
    
    @pytest.fixture
    def voxel_recarve_ports_func(self):
        """Import voxel_recarve_ports, skipping if dependencies unavailable."""
        pytest.importorskip("skimage")
        from generation.ops.embedding.enhanced_embedding import voxel_recarve_ports
        return voxel_recarve_ports
    
    def test_recarve_empty_ports_list(self, simple_cube_mesh, voxel_recarve_ports_func):
        """Test recarve with empty ports list returns original mesh."""
        result_mesh, report = voxel_recarve_ports_func(
            mesh=simple_cube_mesh,
            ports=[],
            voxel_pitch=0.0005,
        )
        
        assert report.success is True
        assert report.ports_carved == 0
    
    def test_recarve_single_port(self, simple_cube_mesh, voxel_recarve_ports_func):
        """Test recarve with a single port."""
        ports = [
            {
                "position": [0.0, 0.0, 0.005],  # Top face
                "direction": [0.0, 0.0, 1.0],   # Pointing outward
                "radius": 0.001,                 # 1mm radius
            }
        ]
        
        result_mesh, report = voxel_recarve_ports_func(
            mesh=simple_cube_mesh,
            ports=ports,
            voxel_pitch=0.0005,
            carve_depth=0.002,
            verify_void_connectivity=False,  # Disable for simple test
        )
        
        assert report.success is True
        assert report.ports_carved == 1
        assert len(report.port_results) == 1
        assert report.port_results[0].voxels_carved > 0
    
    def test_recarve_reports_carve_depth(self, simple_cube_mesh, voxel_recarve_ports_func):
        """Test that recarve reports the actual carve depth used."""
        ports = [
            {
                "position": [0.0, 0.0, 0.005],
                "direction": [0.0, 0.0, 1.0],
                "radius": 0.001,
            }
        ]
        
        result_mesh, report = voxel_recarve_ports_func(
            mesh=simple_cube_mesh,
            ports=ports,
            voxel_pitch=0.0005,
            carve_depth=0.002,
            verify_void_connectivity=False,
        )
        
        assert report.port_results[0].carve_depth == 0.002


class TestPortRecarveIterativeExtension:
    """Test port recarve iterative extension behavior."""
    
    @pytest.fixture
    def voxel_recarve_ports_func(self):
        """Import voxel_recarve_ports, skipping if dependencies unavailable."""
        pytest.importorskip("skimage")
        from generation.ops.embedding.enhanced_embedding import voxel_recarve_ports
        return voxel_recarve_ports
    
    def test_extension_step_factor_affects_depth(self, voxel_recarve_ports_func):
        """Test that extension_step_factor affects carve depth extension."""
        import trimesh
        
        mesh = trimesh.creation.box(extents=[0.01, 0.01, 0.01])
        
        ports = [
            {
                "position": [0.0, 0.0, 0.005],
                "direction": [0.0, 0.0, 1.0],
                "radius": 0.001,
            }
        ]
        
        # With verify_void_connectivity=False, no extension should happen
        result_mesh, report = voxel_recarve_ports_func(
            mesh=mesh,
            ports=ports,
            voxel_pitch=0.0005,
            carve_depth=0.002,
            verify_void_connectivity=False,
            max_extension_iterations=5,
            extension_step_factor=0.5,
        )
        
        # Carve depth should remain at original value
        assert report.port_results[0].carve_depth == 0.002
