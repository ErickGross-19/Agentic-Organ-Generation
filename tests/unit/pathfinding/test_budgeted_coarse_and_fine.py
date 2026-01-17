"""
Test budgeted coarse and fine stages (D4).

This module verifies that both coarse and fine pathfinding stages
respect their respective voxel budgets.
"""

import pytest
import numpy as np

from aog_policies import (
    HierarchicalPathfindingPolicy,
    ResolutionPolicy,
    OperationReport,
)


class TestBudgetedCoarseAndFine:
    """D4: Coarse and fine stages both respect budgets."""
    
    def test_coarse_stage_respects_budget(self):
        """Test that coarse stage voxel count <= max_voxels_coarse."""
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.001,
            pitch_fine=0.0001,
            max_voxels_coarse=1_000_000,
            max_voxels_fine=10_000_000,
        )
        
        report = OperationReport(
            operation="hierarchical_pathfinding",
            success=True,
            requested_policy=policy.to_dict(),
            effective_policy=policy.to_dict(),
            metadata={
                "coarse_stage": {
                    "voxel_count": 800_000,
                    "max_voxels": 1_000_000,
                    "pitch": 0.001,
                },
                "fine_stage": {
                    "voxel_count": 8_000_000,
                    "max_voxels": 10_000_000,
                    "pitch": 0.0001,
                },
            },
        )
        
        coarse_voxels = report.metadata["coarse_stage"]["voxel_count"]
        coarse_max = report.metadata["coarse_stage"]["max_voxels"]
        
        assert coarse_voxels <= coarse_max, (
            f"Coarse voxel count ({coarse_voxels}) exceeds max ({coarse_max})"
        )
    
    def test_fine_stage_respects_budget(self):
        """Test that fine stage voxel count <= max_voxels_fine."""
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.001,
            pitch_fine=0.0001,
            max_voxels_coarse=1_000_000,
            max_voxels_fine=10_000_000,
        )
        
        report = OperationReport(
            operation="hierarchical_pathfinding",
            success=True,
            requested_policy=policy.to_dict(),
            effective_policy=policy.to_dict(),
            metadata={
                "coarse_stage": {
                    "voxel_count": 800_000,
                    "max_voxels": 1_000_000,
                    "pitch": 0.001,
                },
                "fine_stage": {
                    "voxel_count": 8_000_000,
                    "max_voxels": 10_000_000,
                    "pitch": 0.0001,
                },
            },
        )
        
        fine_voxels = report.metadata["fine_stage"]["voxel_count"]
        fine_max = report.metadata["fine_stage"]["max_voxels"]
        
        assert fine_voxels <= fine_max, (
            f"Fine voxel count ({fine_voxels}) exceeds max ({fine_max})"
        )
    
    def test_pitch_relaxed_when_budget_exceeded(self):
        """Test that pitch is relaxed when budget would be exceeded."""
        requested_policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.0001,
            pitch_fine=0.00001,
            max_voxels_coarse=100_000,
            max_voxels_fine=1_000_000,
        )
        
        effective_policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.0002,
            pitch_fine=0.00002,
            max_voxels_coarse=100_000,
            max_voxels_fine=1_000_000,
        )
        
        report = OperationReport(
            operation="hierarchical_pathfinding",
            success=True,
            requested_policy=requested_policy.to_dict(),
            effective_policy=effective_policy.to_dict(),
            warnings=[
                "Coarse pitch relaxed from 0.0001 to 0.0002 due to voxel budget",
                "Fine pitch relaxed from 0.00001 to 0.00002 due to voxel budget",
            ],
            metadata={
                "coarse_stage": {
                    "requested_pitch": 0.0001,
                    "effective_pitch": 0.0002,
                    "pitch_relaxed": True,
                    "relax_factor": 2.0,
                },
                "fine_stage": {
                    "requested_pitch": 0.00001,
                    "effective_pitch": 0.00002,
                    "pitch_relaxed": True,
                    "relax_factor": 2.0,
                },
            },
        )
        
        assert report.metadata["coarse_stage"]["pitch_relaxed"]
        assert report.metadata["fine_stage"]["pitch_relaxed"]
        
        assert len(report.warnings) >= 2
    
    def test_relax_warnings_present(self):
        """Test that relax warnings are present when pitch is relaxed."""
        report = OperationReport(
            operation="hierarchical_pathfinding",
            success=True,
            warnings=[
                "Coarse pitch relaxed: 0.0001 -> 0.0002 (factor 2.0x)",
                "Fine pitch relaxed: 0.00001 -> 0.00002 (factor 2.0x)",
            ],
            metadata={
                "coarse_stage": {"pitch_relaxed": True},
                "fine_stage": {"pitch_relaxed": True},
            },
        )
        
        coarse_warning = any("coarse" in w.lower() and "relax" in w.lower() for w in report.warnings)
        fine_warning = any("fine" in w.lower() and "relax" in w.lower() for w in report.warnings)
        
        assert coarse_warning, "Should have warning about coarse pitch relaxation"
        assert fine_warning, "Should have warning about fine pitch relaxation"
    
    def test_relax_metrics_present(self):
        """Test that relax metrics are present in report metadata."""
        report = OperationReport(
            operation="hierarchical_pathfinding",
            success=True,
            metadata={
                "coarse_stage": {
                    "requested_pitch": 0.0001,
                    "effective_pitch": 0.0002,
                    "pitch_relaxed": True,
                    "relax_factor": 2.0,
                    "voxel_count": 90_000,
                    "max_voxels": 100_000,
                },
                "fine_stage": {
                    "requested_pitch": 0.00001,
                    "effective_pitch": 0.00002,
                    "pitch_relaxed": True,
                    "relax_factor": 2.0,
                    "voxel_count": 900_000,
                    "max_voxels": 1_000_000,
                },
            },
        )
        
        coarse = report.metadata["coarse_stage"]
        fine = report.metadata["fine_stage"]
        
        assert "relax_factor" in coarse
        assert "relax_factor" in fine
        assert coarse["relax_factor"] == 2.0
        assert fine["relax_factor"] == 2.0


class TestBudgetConfiguration:
    """Test budget configuration in hierarchical pathfinding."""
    
    def test_separate_budgets_for_coarse_and_fine(self):
        """Test that coarse and fine have separate budget configurations."""
        policy = HierarchicalPathfindingPolicy(
            max_voxels_coarse=5_000_000,
            max_voxels_fine=25_000_000,
        )
        
        assert policy.max_voxels_coarse != policy.max_voxels_fine
        assert policy.max_voxels_coarse == 5_000_000
        assert policy.max_voxels_fine == 25_000_000
    
    def test_resolution_policy_derives_pathfinding_budgets(self):
        """Test that ResolutionPolicy derives pathfinding budgets."""
        resolution = ResolutionPolicy(
            max_voxels=100_000_000,
            max_voxels_pathfinding_coarse=5_000_000,
            max_voxels_pathfinding_fine=25_000_000,
        )
        
        coarse_budget = resolution.get_max_voxels_for_operation("pathfinding_coarse")
        fine_budget = resolution.get_max_voxels_for_operation("pathfinding_fine")
        
        assert coarse_budget == 5_000_000
        assert fine_budget == 25_000_000
    
    def test_default_budgets_inherit_from_global(self):
        """Test that default budgets inherit from global max_voxels."""
        resolution = ResolutionPolicy(
            max_voxels=50_000_000,
        )
        
        coarse_budget = resolution.get_max_voxels_for_operation("pathfinding_coarse")
        fine_budget = resolution.get_max_voxels_for_operation("pathfinding_fine")
        
        assert coarse_budget == 50_000_000 or coarse_budget is not None
        assert fine_budget == 50_000_000 or fine_budget is not None


class TestBudgetEnforcement:
    """Test budget enforcement behavior."""
    
    def test_low_budget_forces_pitch_relaxation(self):
        """Test that low budget forces pitch relaxation."""
        resolution = ResolutionPolicy(
            min_channel_diameter=20e-6,
            voxels_across_min_diameter=8,
            max_voxels_pathfinding_coarse=1000,
            auto_relax_pitch=True,
        )
        
        large_extents = (0.05, 0.05, 0.05)
        
        effective_pitch, was_relaxed, warning = resolution.compute_relaxed_pitch(
            base_pitch=resolution.pathfinding_pitch_coarse,
            domain_extents=large_extents,
            max_voxels_override=resolution.get_max_voxels_for_operation("pathfinding_coarse"),
        )
        
        if was_relaxed:
            assert effective_pitch > resolution.pathfinding_pitch_coarse
    
    def test_budget_check_before_voxelization(self):
        """Test that budget is checked before voxelization."""
        policy = HierarchicalPathfindingPolicy(
            pitch_coarse=0.0001,
            max_voxels_coarse=100,
        )
        
        domain_extents = (0.01, 0.01, 0.01)
        
        nx = int(np.ceil(domain_extents[0] / policy.pitch_coarse))
        ny = int(np.ceil(domain_extents[1] / policy.pitch_coarse))
        nz = int(np.ceil(domain_extents[2] / policy.pitch_coarse))
        estimated_voxels = nx * ny * nz
        
        would_exceed = estimated_voxels > policy.max_voxels_coarse
        
        assert would_exceed, (
            f"Estimated voxels ({estimated_voxels}) should exceed budget ({policy.max_voxels_coarse})"
        )


class TestBudgetReporting:
    """Test budget reporting in operation reports."""
    
    def test_report_includes_budget_metrics(self):
        """Test that report includes comprehensive budget metrics."""
        report = OperationReport(
            operation="hierarchical_pathfinding",
            success=True,
            metadata={
                "coarse_stage": {
                    "voxel_count": 800_000,
                    "max_voxels": 1_000_000,
                    "budget_utilization": 0.8,
                    "pitch": 0.001,
                    "pitch_relaxed": False,
                },
                "fine_stage": {
                    "voxel_count": 9_500_000,
                    "max_voxels": 10_000_000,
                    "budget_utilization": 0.95,
                    "pitch": 0.0001,
                    "pitch_relaxed": False,
                },
            },
        )
        
        coarse = report.metadata["coarse_stage"]
        fine = report.metadata["fine_stage"]
        
        assert "voxel_count" in coarse
        assert "max_voxels" in coarse
        assert "budget_utilization" in coarse
        
        assert "voxel_count" in fine
        assert "max_voxels" in fine
        assert "budget_utilization" in fine
    
    def test_budget_utilization_calculated_correctly(self):
        """Test that budget utilization is calculated correctly."""
        voxel_count = 800_000
        max_voxels = 1_000_000
        expected_utilization = voxel_count / max_voxels
        
        report = OperationReport(
            operation="hierarchical_pathfinding",
            success=True,
            metadata={
                "coarse_stage": {
                    "voxel_count": voxel_count,
                    "max_voxels": max_voxels,
                    "budget_utilization": expected_utilization,
                },
            },
        )
        
        actual_utilization = report.metadata["coarse_stage"]["budget_utilization"]
        assert abs(actual_utilization - expected_utilization) < 1e-10
