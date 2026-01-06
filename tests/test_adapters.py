"""
Tests for the mesh adapters and exporters.

These tests validate:
- STL export with unit scaling
- Hollow tube mesh generation
- Unit metadata sidecar files
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, MagicMock, patch


class TestExportSTLUnitScaling:
    """Tests for STL export with unit scaling."""
    
    def test_export_stl_default_units(self):
        """Test that export_stl defaults to mm units."""
        from generation.adapters.mesh_adapter import export_stl
        import numpy as np
        
        mock_network = Mock()
        mock_network.segments = {}
        
        with patch('generation.adapters.mesh_adapter.to_trimesh') as mock_to_trimesh:
            mock_result = Mock()
            mock_result.is_success.return_value = True
            mock_mesh = Mock()
            mock_mesh.is_watertight = True
            vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            mock_mesh.vertices = vertices.copy()
            copy_mock = Mock()
            copy_mock.vertices = vertices.copy()
            mock_mesh.copy.return_value = copy_mock
            mock_result.metadata = {'mesh': mock_mesh}
            mock_to_trimesh.return_value = mock_result
            
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
                filepath = f.name
            
            try:
                result = export_stl(
                    mock_network,
                    filepath,
                    write_metadata=False,
                )
                
                assert result.metadata.get('output_units') == 'mm'
                assert result.metadata.get('scale_factor') == pytest.approx(1.0)
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)
    
    def test_export_stl_custom_units(self):
        """Test export_stl with custom output units."""
        from generation.adapters.mesh_adapter import export_stl
        import numpy as np
        
        mock_network = Mock()
        mock_network.segments = {}
        
        with patch('generation.adapters.mesh_adapter.to_trimesh') as mock_to_trimesh:
            mock_result = Mock()
            mock_result.is_success.return_value = True
            mock_mesh = Mock()
            mock_mesh.is_watertight = True
            vertices = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
            mock_mesh.vertices = vertices.copy()
            copy_mock = Mock()
            copy_mock.vertices = vertices.copy()
            mock_mesh.copy.return_value = copy_mock
            mock_result.metadata = {'mesh': mock_mesh}
            mock_to_trimesh.return_value = mock_result
            
            with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
                filepath = f.name
            
            try:
                result = export_stl(
                    mock_network,
                    filepath,
                    output_units="m",
                    write_metadata=False,
                )
                
                assert result.metadata.get('output_units') == 'm'
                assert result.metadata.get('scale_factor') == pytest.approx(0.001)
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)


class TestUnitContextScaleMesh:
    """Tests for UnitContext.scale_mesh method."""
    
    def test_scale_mesh_mm_no_change(self):
        """Test that mm output doesn't change mesh vertices."""
        from generation.utils.units import UnitContext
        import numpy as np
        
        ctx = UnitContext(output_units="mm")
        
        mock_mesh = Mock()
        mock_mesh.vertices = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mock_mesh.copy.return_value = Mock()
        mock_mesh.copy.return_value.vertices = mock_mesh.vertices.copy()
        
        scaled = ctx.scale_mesh(mock_mesh)
        
        np.testing.assert_array_almost_equal(
            scaled.vertices,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )
    
    def test_scale_mesh_to_meters(self):
        """Test scaling mesh to meters."""
        from generation.utils.units import UnitContext
        import numpy as np
        
        ctx = UnitContext(output_units="m")
        
        mock_mesh = Mock()
        original_vertices = np.array([[1000.0, 2000.0, 3000.0]])
        mock_mesh.vertices = original_vertices.copy()
        
        copy_mock = Mock()
        copy_mock.vertices = original_vertices.copy()
        mock_mesh.copy.return_value = copy_mock
        
        scaled = ctx.scale_mesh(mock_mesh)
        
        np.testing.assert_array_almost_equal(
            scaled.vertices,
            np.array([[1.0, 2.0, 3.0]])
        )
    
    def test_scale_mesh_to_cm(self):
        """Test scaling mesh to centimeters."""
        from generation.utils.units import UnitContext
        import numpy as np
        
        ctx = UnitContext(output_units="cm")
        
        mock_mesh = Mock()
        original_vertices = np.array([[100.0, 200.0, 300.0]])
        mock_mesh.vertices = original_vertices.copy()
        
        copy_mock = Mock()
        copy_mock.vertices = original_vertices.copy()
        mock_mesh.copy.return_value = copy_mock
        
        scaled = ctx.scale_mesh(mock_mesh)
        
        np.testing.assert_array_almost_equal(
            scaled.vertices,
            np.array([[10.0, 20.0, 30.0]])
        )
    
    def test_scale_mesh_to_um(self):
        """Test scaling mesh to micrometers."""
        from generation.utils.units import UnitContext
        import numpy as np
        
        ctx = UnitContext(output_units="um")
        
        mock_mesh = Mock()
        original_vertices = np.array([[1.0, 2.0, 3.0]])
        mock_mesh.vertices = original_vertices.copy()
        
        copy_mock = Mock()
        copy_mock.vertices = original_vertices.copy()
        mock_mesh.copy.return_value = copy_mock
        
        scaled = ctx.scale_mesh(mock_mesh)
        
        np.testing.assert_array_almost_equal(
            scaled.vertices,
            np.array([[1000.0, 2000.0, 3000.0]])
        )
    
    def test_scale_mesh_does_not_modify_original(self):
        """Test that scale_mesh doesn't modify the original mesh."""
        from generation.utils.units import UnitContext
        import numpy as np
        
        ctx = UnitContext(output_units="m")
        
        mock_mesh = Mock()
        original_vertices = np.array([[1000.0, 2000.0, 3000.0]])
        mock_mesh.vertices = original_vertices.copy()
        
        copy_mock = Mock()
        copy_mock.vertices = original_vertices.copy()
        mock_mesh.copy.return_value = copy_mock
        
        ctx.scale_mesh(mock_mesh)
        
        np.testing.assert_array_almost_equal(
            mock_mesh.vertices,
            original_vertices
        )


class TestMetadataSidecar:
    """Tests for unit metadata sidecar files."""
    
    def test_unit_context_get_metadata(self):
        """Test UnitContext.get_metadata returns correct structure."""
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="cm")
        metadata = ctx.get_metadata()
        
        assert "units" in metadata
        assert "scale_factor_applied" in metadata
        assert "internal_units" in metadata
        
        assert metadata["units"] == "cm"
        assert metadata["scale_factor_applied"] == pytest.approx(0.1)
    
    def test_default_unit_context(self):
        """Test DEFAULT_UNIT_CONTEXT is mm."""
        from generation.utils.units import DEFAULT_UNIT_CONTEXT
        
        assert DEFAULT_UNIT_CONTEXT.output_units == "mm"
        assert DEFAULT_UNIT_CONTEXT.scale_factor == pytest.approx(1.0)
