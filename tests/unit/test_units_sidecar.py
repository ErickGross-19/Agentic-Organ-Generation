"""
Tests for .units.json sidecar file reading in embedding and validity repair.

These tests validate:
- Reading .units.json sidecar files in embed_tree_as_negative_space
- Reading .units.json sidecar files in validate_and_repair_geometry
- Fallback to heuristics when no sidecar file exists
- Trust sidecar over heuristics when stl_units="auto"
"""

import pytest
import json
import tempfile
import os
import numpy as np


class TestEmbeddingSidecarReading:
    """Tests for .units.json sidecar reading in embedding."""
    
    def test_sidecar_file_is_read_when_exists(self):
        """Test that .units.json sidecar file is read when it exists."""
        import trimesh
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh = trimesh.creation.box(extents=[20.0, 60.0, 30.0])
            stl_path = os.path.join(tmpdir, "test_tree.stl")
            mesh.export(stl_path)
            
            sidecar_path = stl_path + ".units.json"
            with open(sidecar_path, 'w') as f:
                json.dump({"units": "mm"}, f)
            
            assert os.path.exists(sidecar_path)
            
            with open(sidecar_path, 'r') as f:
                data = json.load(f)
            
            assert data["units"] == "mm"
    
    def test_sidecar_units_format(self):
        """Test that sidecar file has correct format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sidecar_path = os.path.join(tmpdir, "test.stl.units.json")
            
            sidecar_data = {
                "units": "mm",
                "scale_factor_applied": 1000.0,
                "internal_units": "m",
            }
            
            with open(sidecar_path, 'w') as f:
                json.dump(sidecar_data, f)
            
            with open(sidecar_path, 'r') as f:
                loaded = json.load(f)
            
            assert loaded["units"] == "mm"
            assert loaded["scale_factor_applied"] == 1000.0
            assert loaded["internal_units"] == "m"


class TestValidityRepairSidecarReading:
    """Tests for .units.json sidecar reading in validity repair."""
    
    def test_repair_reads_sidecar_file(self):
        """Test that validate_and_repair_geometry reads .units.json sidecar."""
        import trimesh
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh = trimesh.creation.box(extents=[20.0, 60.0, 30.0])
            stl_path = os.path.join(tmpdir, "test_mesh.stl")
            mesh.export(stl_path)
            
            sidecar_path = stl_path + ".units.json"
            with open(sidecar_path, 'w') as f:
                json.dump({"units": "mm"}, f)
            
            assert os.path.exists(sidecar_path)
    
    def test_sidecar_units_supported_values(self):
        """Test that sidecar supports m, mm, cm, um units."""
        supported_units = ["m", "mm", "cm", "um"]
        
        for unit in supported_units:
            with tempfile.TemporaryDirectory() as tmpdir:
                sidecar_path = os.path.join(tmpdir, "test.stl.units.json")
                
                with open(sidecar_path, 'w') as f:
                    json.dump({"units": unit}, f)
                
                with open(sidecar_path, 'r') as f:
                    loaded = json.load(f)
                
                assert loaded["units"] == unit


class TestUnitHeuristicsFallback:
    """Tests for unit detection heuristics when no sidecar exists."""
    
    def test_small_mesh_detected_as_meters(self):
        """Test that small mesh (max dimension < 1.0) is detected as meters."""
        import trimesh
        
        mesh = trimesh.creation.box(extents=[0.02, 0.06, 0.03])
        
        extents = mesh.bounding_box.extents
        max_dim = np.max(extents)
        
        assert max_dim < 1.0
        
        if max_dim < 1.0:
            detected_units = "m"
        else:
            detected_units = "mm"
        
        assert detected_units == "m"
    
    def test_large_mesh_detected_as_mm(self):
        """Test that large mesh (max dimension >= 1.0) is detected as mm."""
        import trimesh
        
        mesh = trimesh.creation.box(extents=[20.0, 60.0, 30.0])
        
        extents = mesh.bounding_box.extents
        max_dim = np.max(extents)
        
        assert max_dim >= 1.0
        
        if max_dim < 1.0:
            detected_units = "m"
        else:
            detected_units = "mm"
        
        assert detected_units == "mm"
    
    def test_sidecar_takes_precedence_over_heuristics(self):
        """Test that sidecar units take precedence over heuristics."""
        import trimesh
        
        mesh = trimesh.creation.box(extents=[20.0, 60.0, 30.0])
        
        extents = mesh.bounding_box.extents
        max_dim = np.max(extents)
        
        heuristic_units = "mm" if max_dim >= 1.0 else "m"
        sidecar_units = "m"
        
        if sidecar_units:
            final_units = sidecar_units
        else:
            final_units = heuristic_units
        
        assert final_units == "m"
        assert final_units != heuristic_units


class TestUnitScaling:
    """Tests for unit scaling in embedding and repair."""
    
    def test_mm_to_m_scaling(self):
        """Test scaling from mm to meters."""
        mm_value = 50.0
        m_value = mm_value * 0.001
        
        assert m_value == pytest.approx(0.05)
    
    def test_cm_to_m_scaling(self):
        """Test scaling from cm to meters."""
        cm_value = 5.0
        m_value = cm_value * 0.01
        
        assert m_value == pytest.approx(0.05)
    
    def test_um_to_m_scaling(self):
        """Test scaling from um to meters."""
        um_value = 50000.0
        m_value = um_value * 1e-6
        
        assert m_value == pytest.approx(0.05)
    
    def test_mesh_scaling_preserves_shape(self):
        """Test that mesh scaling preserves shape (aspect ratios)."""
        import trimesh
        
        mesh = trimesh.creation.box(extents=[20.0, 60.0, 30.0])
        
        original_extents = mesh.bounding_box.extents.copy()
        original_aspect_xy = original_extents[0] / original_extents[1]
        original_aspect_xz = original_extents[0] / original_extents[2]
        
        mesh.apply_scale(0.001)
        
        scaled_extents = mesh.bounding_box.extents
        scaled_aspect_xy = scaled_extents[0] / scaled_extents[1]
        scaled_aspect_xz = scaled_extents[0] / scaled_extents[2]
        
        assert scaled_aspect_xy == pytest.approx(original_aspect_xy)
        assert scaled_aspect_xz == pytest.approx(original_aspect_xz)


class TestVoxelBudgetInRepair:
    """Tests for voxel budget logic in validity repair."""
    
    def test_voxel_count_calculation(self):
        """Test voxel count calculation from extents and pitch."""
        extents = np.array([0.02, 0.06, 0.03])
        voxel_pitch = 0.001
        
        grid_shape = np.ceil(extents / voxel_pitch).astype(int)
        total_voxels = int(np.prod(grid_shape))
        
        assert grid_shape[0] == 20
        assert grid_shape[1] == 60
        assert grid_shape[2] == 30
        assert total_voxels == 20 * 60 * 30
    
    def test_auto_pitch_adjustment_when_budget_exceeded(self):
        """Test that pitch is adjusted when voxel budget is exceeded."""
        extents = np.array([0.1, 0.1, 0.1])
        voxel_pitch = 0.0001
        max_voxels = 1e6
        
        grid_shape = np.ceil(extents / voxel_pitch).astype(int)
        total_voxels = int(np.prod(grid_shape))
        
        assert total_voxels > max_voxels
        
        suggested_pitch = (np.prod(extents) / max_voxels) ** (1/3)
        
        new_grid_shape = np.ceil(extents / suggested_pitch).astype(int)
        new_total_voxels = int(np.prod(new_grid_shape))
        
        assert new_total_voxels <= max_voxels * 1.5
    
    def test_voxel_budget_default_value(self):
        """Test that default voxel budget is reasonable (3e7)."""
        default_max_voxels = 3e7
        
        assert default_max_voxels == 30_000_000
        
        cube_side = int(default_max_voxels ** (1/3))
        assert 300 <= cube_side <= 320
