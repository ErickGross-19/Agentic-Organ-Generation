"""
Test that all modules can be imported without collisions.

This module validates that the key modules in the codebase can be
imported cleanly without circular dependencies or naming collisions.
"""

import pytest


class TestAOGPoliciesImport:
    """Test aog_policies package imports cleanly."""
    
    def test_aog_policies_import(self):
        """Test aog_policies package imports cleanly."""
        import aog_policies
        
        assert hasattr(aog_policies, "PortPlacementPolicy")
        assert hasattr(aog_policies, "ChannelPolicy")
        assert hasattr(aog_policies, "GrowthPolicy")
        assert hasattr(aog_policies, "TissueSamplingPolicy")
        assert hasattr(aog_policies, "CollisionPolicy")
        assert hasattr(aog_policies, "MeshSynthesisPolicy")
        assert hasattr(aog_policies, "MeshMergePolicy")
        assert hasattr(aog_policies, "EmbeddingPolicy")
        assert hasattr(aog_policies, "ValidationPolicy")
        assert hasattr(aog_policies, "RepairPolicy")
        assert hasattr(aog_policies, "OperationReport")
    
    def test_aog_policies_resolution_policy(self):
        """Test ResolutionPolicy is available."""
        from aog_policies import ResolutionPolicy
        
        assert ResolutionPolicy is not None
    
    def test_aog_policies_pathfinding_policy(self):
        """Test PathfindingPolicy is available."""
        from aog_policies import PathfindingPolicy
        
        assert PathfindingPolicy is not None
    
    def test_aog_policies_compose_policy(self):
        """Test ComposePolicy is available."""
        from aog_policies import ComposePolicy
        
        assert ComposePolicy is not None


class TestGenerationAPIImport:
    """Test generation.api package imports cleanly."""
    
    def test_generation_api_import(self):
        """Test generation.api package imports cleanly."""
        from generation.api import (
            generate_network,
            generate_void_mesh,
            build_component,
            embed_void,
            embed_void_mesh_as_negative_space,
        )
        
        assert callable(generate_network)
        assert callable(generate_void_mesh)
        assert callable(build_component)
        assert callable(embed_void)
        assert callable(embed_void_mesh_as_negative_space)


class TestValidityAPIImport:
    """Test validity.api package imports cleanly."""
    
    def test_validity_api_import(self):
        """Test validity.api package imports cleanly."""
        from validity.api import (
            validate_mesh,
            validate_network,
            validate_artifacts,
            repair_mesh,
            validate_repair_validate,
            run_full_pipeline,
        )
        
        assert callable(validate_mesh)
        assert callable(validate_network)
        assert callable(validate_artifacts)
        assert callable(repair_mesh)
        assert callable(validate_repair_validate)
        assert callable(run_full_pipeline)


class TestProgrammaticBackendImport:
    """Test programmatic backend imports cleanly."""
    
    def test_programmatic_backend_import(self):
        """Test programmatic backend imports cleanly."""
        from generation.backends.programmatic_backend import (
            ProgrammaticBackend,
            StepSpec,
            GenerationReport,
        )
        from aog_policies import ProgramPolicy, WaypointPolicy
        
        assert ProgrammaticBackend is not None
        assert ProgramPolicy is not None
        assert WaypointPolicy is not None
        assert StepSpec is not None
        assert GenerationReport is not None


class TestEmbeddingOpsImport:
    """Test embedding ops imports cleanly."""
    
    def test_embedding_ops_import(self):
        """Test embedding ops imports cleanly."""
        from generation.ops.embedding import (
            embed_with_port_preservation,
            embed_void_mesh_as_negative_space,
        )
        
        assert callable(embed_with_port_preservation)
        assert callable(embed_void_mesh_as_negative_space)


class TestVoxelRecarveImport:
    """Test voxel recarve imports cleanly."""
    
    def test_voxel_recarve_import(self):
        """Test voxel recarve imports cleanly."""
        from generation.ops.embedding.enhanced_embedding import (
            voxel_recarve_ports,
            RecarveReport,
            PortRecarveResult,
        )
        
        assert callable(voxel_recarve_ports)
        assert RecarveReport is not None
        assert PortRecarveResult is not None


class TestDesignSpecImport:
    """Test designspec package imports cleanly."""
    
    def test_designspec_spec_import(self):
        """Test designspec.spec imports cleanly."""
        from designspec.spec import DesignSpec
        
        assert DesignSpec is not None
    
    def test_designspec_runner_import(self):
        """Test designspec.runner imports cleanly."""
        from designspec.runner import DesignSpecRunner, RunnerResult
        
        assert DesignSpecRunner is not None
        assert RunnerResult is not None
    
    def test_designspec_plan_import(self):
        """Test designspec.plan imports cleanly."""
        from designspec.plan import ExecutionPlan
        
        assert ExecutionPlan is not None
