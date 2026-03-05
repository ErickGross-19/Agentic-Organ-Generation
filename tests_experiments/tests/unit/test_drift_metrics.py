"""
Tests for drift-aware validation metrics.

These tests validate:
- compute_drift_metrics() function for min channel diameter, connectivity, and drift
- Actionable error messages and suggested fixes
- Integration with ValidationReport
"""

import pytest
import numpy as np
import networkx as nx


class TestComputeDriftMetrics:
    """Tests for compute_drift_metrics function."""
    
    def test_compute_drift_metrics_with_valid_graph(self):
        """Test compute_drift_metrics with a valid centerline graph."""
        from validity.pipeline import compute_drift_metrics
        
        G = nx.Graph()
        G.add_node(0, radius=0.001, position=(0.0, 0.0, 0.0))
        G.add_node(1, radius=0.0008, position=(0.01, 0.0, 0.0))
        G.add_node(2, radius=0.0005, position=(0.02, 0.0, 0.0))
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        
        connectivity_info = {
            "num_components": 1,
            "inlet_outlet_connected": True,
        }
        
        metrics = compute_drift_metrics(
            original_spec=None,
            centerline_graph=G,
            connectivity_info=connectivity_info,
            min_channel_diameter_threshold=0.0005,
        )
        
        assert "min_channel_diameter" in metrics
        assert "min_channel_diameter_mm" in metrics
        assert "min_channel_diameter_ok" in metrics
        assert "connectivity_preserved" in metrics
        assert "validation_passed" in metrics
        
        assert metrics["min_radius"] == pytest.approx(0.0005)
        assert metrics["max_radius"] == pytest.approx(0.001)
        assert metrics["min_channel_diameter"] == pytest.approx(0.001)
        assert metrics["min_channel_diameter_mm"] == pytest.approx(1.0)
    
    def test_compute_drift_metrics_with_empty_graph(self):
        """Test compute_drift_metrics with empty graph."""
        from validity.pipeline import compute_drift_metrics
        
        G = nx.Graph()
        
        connectivity_info = {
            "num_components": 0,
            "inlet_outlet_connected": False,
        }
        
        metrics = compute_drift_metrics(
            original_spec=None,
            centerline_graph=G,
            connectivity_info=connectivity_info,
        )
        
        assert metrics["min_radius"] == 0.0
        assert metrics["max_radius"] == 0.0
        assert metrics["min_channel_diameter"] == 0.0
        assert len(metrics["drift_warnings"]) > 0
    
    def test_compute_drift_metrics_below_threshold(self):
        """Test compute_drift_metrics when min diameter is below threshold."""
        from validity.pipeline import compute_drift_metrics
        
        G = nx.Graph()
        G.add_node(0, radius=0.0001, position=(0.0, 0.0, 0.0))
        G.add_node(1, radius=0.0001, position=(0.01, 0.0, 0.0))
        G.add_edge(0, 1)
        
        connectivity_info = {
            "num_components": 1,
            "inlet_outlet_connected": True,
        }
        
        metrics = compute_drift_metrics(
            original_spec=None,
            centerline_graph=G,
            connectivity_info=connectivity_info,
            min_channel_diameter_threshold=0.0005,
        )
        
        assert metrics["min_channel_diameter_ok"] is False
        assert len(metrics["drift_errors"]) > 0
        assert len(metrics["suggested_fixes"]) > 0
        assert metrics["validation_passed"] is False
    
    def test_compute_drift_metrics_multiple_components(self):
        """Test compute_drift_metrics with multiple disconnected components."""
        from validity.pipeline import compute_drift_metrics
        
        G = nx.Graph()
        G.add_node(0, radius=0.001, position=(0.0, 0.0, 0.0))
        G.add_node(1, radius=0.001, position=(0.01, 0.0, 0.0))
        G.add_node(2, radius=0.001, position=(0.1, 0.0, 0.0))
        G.add_node(3, radius=0.001, position=(0.11, 0.0, 0.0))
        G.add_edge(0, 1)
        G.add_edge(2, 3)
        
        connectivity_info = {
            "num_components": 2,
            "inlet_outlet_connected": False,
        }
        
        metrics = compute_drift_metrics(
            original_spec=None,
            centerline_graph=G,
            connectivity_info=connectivity_info,
        )
        
        assert metrics["connectivity_preserved"] is False
        assert metrics["num_components"] == 2
        assert len(metrics["drift_errors"]) > 0
        assert metrics["validation_passed"] is False
    
    def test_compute_drift_metrics_inlet_outlet_disconnected(self):
        """Test compute_drift_metrics when inlet and outlet are disconnected."""
        from validity.pipeline import compute_drift_metrics
        
        G = nx.Graph()
        G.add_node(0, radius=0.001, position=(0.0, 0.0, 0.0))
        G.add_node(1, radius=0.001, position=(0.01, 0.0, 0.0))
        G.add_edge(0, 1)
        
        connectivity_info = {
            "num_components": 1,
            "inlet_outlet_connected": False,
        }
        
        metrics = compute_drift_metrics(
            original_spec=None,
            centerline_graph=G,
            connectivity_info=connectivity_info,
        )
        
        assert metrics["inlet_outlet_connected"] is False
        assert len(metrics["drift_errors"]) > 0
        assert "inlet" in metrics["drift_errors"][0].lower() or "outlet" in metrics["drift_errors"][0].lower()
    
    def test_compute_drift_metrics_with_original_spec(self):
        """Test compute_drift_metrics with original spec for drift comparison."""
        from validity.pipeline import compute_drift_metrics
        
        G = nx.Graph()
        G.add_node(0, radius=0.0005, position=(0.0, 0.0, 0.0))
        G.add_node(1, radius=0.0005, position=(0.01, 0.0, 0.0))
        G.add_edge(0, 1)
        
        connectivity_info = {
            "num_components": 1,
            "inlet_outlet_connected": True,
        }
        
        original_spec = {
            "expected_min_radius": 0.001,
        }
        
        metrics = compute_drift_metrics(
            original_spec=original_spec,
            centerline_graph=G,
            connectivity_info=connectivity_info,
        )
        
        assert len(metrics["drift_warnings"]) > 0
        assert "drift" in metrics["drift_warnings"][0].lower()
    
    def test_compute_drift_metrics_validation_passed(self):
        """Test compute_drift_metrics returns validation_passed=True when all checks pass."""
        from validity.pipeline import compute_drift_metrics
        
        G = nx.Graph()
        G.add_node(0, radius=0.001, position=(0.0, 0.0, 0.0))
        G.add_node(1, radius=0.001, position=(0.01, 0.0, 0.0))
        G.add_edge(0, 1)
        
        connectivity_info = {
            "num_components": 1,
            "inlet_outlet_connected": True,
        }
        
        metrics = compute_drift_metrics(
            original_spec=None,
            centerline_graph=G,
            connectivity_info=connectivity_info,
            min_channel_diameter_threshold=0.0005,
        )
        
        assert metrics["validation_passed"] is True
        assert len(metrics["drift_errors"]) == 0
    
    def test_compute_drift_metrics_suggested_fixes_are_actionable(self):
        """Test that suggested fixes contain actionable parameter names."""
        from validity.pipeline import compute_drift_metrics
        
        G = nx.Graph()
        G.add_node(0, radius=0.0001, position=(0.0, 0.0, 0.0))
        G.add_edge(0, 0)
        
        connectivity_info = {
            "num_components": 2,
            "inlet_outlet_connected": False,
        }
        
        metrics = compute_drift_metrics(
            original_spec=None,
            centerline_graph=G,
            connectivity_info=connectivity_info,
            min_channel_diameter_threshold=0.0005,
        )
        
        all_fixes = " ".join(metrics["suggested_fixes"])
        
        assert any(param in all_fixes.lower() for param in [
            "voxel_pitch", "smoothing", "dilation", "pitch"
        ])


class TestDriftMetricsIntegration:
    """Integration tests for drift metrics in validation pipeline."""
    
    def test_validation_report_includes_drift_metrics(self):
        """Test that ValidationReport includes drift_metrics field."""
        from validity.models import ValidationReport
        
        report = ValidationReport(
            input_file="test.stl",
            intermediate_stl=None,
            cleaned_stl="test_cleaned.stl",
            scafold_stl="test_scaffold.stl",
            before=None,
            after_basic_clean=None,
            after_voxel=None,
            after_repair=None,
            flags=None,
            surface_before=None,
            surface_after=None,
            connectivity={},
            centerline_summary={},
            poiseuille_summary={},
            drift_metrics={"min_channel_diameter": 0.001},
        )
        
        assert report.drift_metrics is not None
        assert report.drift_metrics["min_channel_diameter"] == 0.001
    
    def test_validation_report_drift_metrics_optional(self):
        """Test that drift_metrics is optional in ValidationReport."""
        from validity.models import ValidationReport
        
        report = ValidationReport(
            input_file="test.stl",
            intermediate_stl=None,
            cleaned_stl="test_cleaned.stl",
            scafold_stl="test_scaffold.stl",
            before=None,
            after_basic_clean=None,
            after_voxel=None,
            after_repair=None,
            flags=None,
            surface_before=None,
            surface_after=None,
            connectivity={},
            centerline_summary={},
            poiseuille_summary={},
        )
        
        assert report.drift_metrics is None
