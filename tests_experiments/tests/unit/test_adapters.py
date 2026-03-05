"""
Tests for the mesh adapters and exporters.

These tests validate:
- STL export with unit scaling
- Hollow tube mesh generation
- Unit metadata sidecar files

Note: Internal units are meters (legacy convention from the codebase).
When output_units="mm", internal value 0.05 becomes 50mm in output (scale_factor=1000).
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, MagicMock, patch


class TestExportSTLUnitScaling:
    """Tests for STL export with unit scaling."""
    
    def test_export_stl_default_units(self):
        """Test that export_stl defaults to mm units.
        
        Internal units are meters, so scale_factor for mm output = 1000.
        """
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
                assert result.metadata.get('scale_factor') == pytest.approx(1000.0)
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)
    
    def test_export_stl_custom_units(self):
        """Test export_stl with custom output units.
        
        Internal units are meters, so scale_factor for m output = 1.0 (no change).
        """
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
                assert result.metadata.get('scale_factor') == pytest.approx(1.0)
            finally:
                if os.path.exists(filepath):
                    os.unlink(filepath)


class TestUnitContextScaleMesh:
    """Tests for UnitContext.scale_mesh method."""
    
    def test_scale_mesh_mm_no_change(self):
        """Test that mm output scales mesh vertices by 1000 (m -> mm).
        
        Internal units are meters, so mm output has scale_factor = 1000.
        Internal (0.001, 0.002, 0.003) meters becomes (1, 2, 3) mm.
        """
        from generation.utils.units import UnitContext
        import numpy as np
        
        ctx = UnitContext(output_units="mm")
        
        mock_mesh = Mock()
        mock_mesh.vertices = np.array([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]])
        mock_mesh.copy.return_value = Mock()
        mock_mesh.copy.return_value.vertices = mock_mesh.vertices.copy()
        
        scaled = ctx.scale_mesh(mock_mesh)
        
        np.testing.assert_array_almost_equal(
            scaled.vertices,
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        )
    
    def test_scale_mesh_to_meters(self):
        """Test scaling mesh to meters (no change since internal is meters).
        
        Internal units are meters, so m output has scale_factor = 1.0.
        Internal (1.0, 2.0, 3.0) meters stays (1.0, 2.0, 3.0) meters.
        """
        from generation.utils.units import UnitContext
        import numpy as np
        
        ctx = UnitContext(output_units="m")
        
        mock_mesh = Mock()
        original_vertices = np.array([[1.0, 2.0, 3.0]])
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
        """Test scaling mesh to centimeters.
        
        Internal units are meters, so cm output has scale_factor = 100.
        Internal (0.1, 0.2, 0.3) meters becomes (10, 20, 30) cm.
        """
        from generation.utils.units import UnitContext
        import numpy as np
        
        ctx = UnitContext(output_units="cm")
        
        mock_mesh = Mock()
        original_vertices = np.array([[0.1, 0.2, 0.3]])
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
        """Test scaling mesh to micrometers.
        
        Internal units are meters, so um output has scale_factor = 1e6.
        Internal (0.001, 0.002, 0.003) meters becomes (1000, 2000, 3000) um.
        """
        from generation.utils.units import UnitContext
        import numpy as np
        
        ctx = UnitContext(output_units="um")
        
        mock_mesh = Mock()
        original_vertices = np.array([[0.001, 0.002, 0.003]])
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
        """Test UnitContext.get_metadata returns correct structure.
        
        Internal units are meters, so cm output has scale_factor = 100.
        """
        from generation.utils.units import UnitContext
        
        ctx = UnitContext(output_units="cm")
        metadata = ctx.get_metadata()
        
        assert "units" in metadata
        assert "scale_factor_applied" in metadata
        assert "internal_units" in metadata
        
        assert metadata["units"] == "cm"
        assert metadata["scale_factor_applied"] == pytest.approx(100.0)
    
    def test_default_unit_context(self):
        """Test DEFAULT_UNIT_CONTEXT is mm.
        
        Internal units are meters, so mm output has scale_factor = 1000.
        """
        from generation.utils.units import DEFAULT_UNIT_CONTEXT
        
        assert DEFAULT_UNIT_CONTEXT.output_units == "mm"
        assert DEFAULT_UNIT_CONTEXT.scale_factor == pytest.approx(1000.0)
