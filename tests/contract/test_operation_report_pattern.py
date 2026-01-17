"""
Test that public API functions return OperationReport with requested/effective policy.

This module validates the contract that all public API functions return
OperationReport objects containing both the requested and effective policies.
"""

import pytest


class TestGenerateNetworkReturnsOperationReport:
    """Test generate_network returns OperationReport."""
    
    def test_generate_network_returns_operation_report(self):
        """Test generate_network returns OperationReport."""
        from generation.api import generate_network
        from generation.core.domain import CylinderDomain
        from aog_policies import GrowthPolicy, OperationReport
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        ports = {
            "inlets": [{"position": (0, 0, 0.0015), "radius": 0.0005, "direction": (0, 0, -1), "vessel_type": "arterial"}],
            "outlets": [{"position": (0, 0, -0.0015), "radius": 0.0003, "direction": (0, 0, 1), "vessel_type": "venous"}],
        }
        growth_policy = GrowthPolicy(backend="cco_hybrid", max_iterations=10, target_terminals=5)
        
        network, report = generate_network(
            generator_kind="cco_hybrid",
            domain=domain,
            ports=ports,
            growth_policy=growth_policy,
        )
        
        assert isinstance(report, OperationReport)
        assert report.operation == "generate_network"
        assert report.requested_policy is not None
        assert report.effective_policy is not None
        assert isinstance(report.requested_policy, dict)
        assert isinstance(report.effective_policy, dict)
    
    def test_generate_network_report_is_json_serializable(self):
        """Test generate_network report is JSON-serializable."""
        import json
        from generation.api import generate_network
        from generation.core.domain import CylinderDomain
        from aog_policies import GrowthPolicy
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        ports = {
            "inlets": [{"position": (0, 0, 0.0015), "radius": 0.0005, "direction": (0, 0, -1), "vessel_type": "arterial"}],
            "outlets": [{"position": (0, 0, -0.0015), "radius": 0.0003, "direction": (0, 0, 1), "vessel_type": "venous"}],
        }
        growth_policy = GrowthPolicy(backend="cco_hybrid", max_iterations=10, target_terminals=5)
        
        network, report = generate_network(
            generator_kind="cco_hybrid",
            domain=domain,
            ports=ports,
            growth_policy=growth_policy,
        )
        
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)
        restored = json.loads(json_str)
        
        assert restored["operation"] == "generate_network"


class TestEmbedVoidReturnsOperationReport:
    """Test embed_void returns OperationReport."""
    
    def test_embed_void_returns_operation_report(self):
        """Test embed_void returns OperationReport."""
        from generation.api import embed_void
        from generation.core.domain import CylinderDomain
        from aog_policies import EmbeddingPolicy, OperationReport
        import trimesh
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        void_mesh = trimesh.creation.cylinder(radius=0.001, height=0.002)
        
        policy = EmbeddingPolicy(
            voxel_pitch=5e-4,
            preserve_ports_enabled=False,
        )
        
        solid, void_out, shell, report = embed_void(
            domain=domain,
            void_mesh=void_mesh,
            embedding_policy=policy,
        )
        
        assert isinstance(report, OperationReport)
        assert report.operation == "embed_void"
        assert report.requested_policy is not None
        assert report.effective_policy is not None
    
    def test_embed_void_report_is_json_serializable(self):
        """Test embed_void report is JSON-serializable."""
        import json
        from generation.api import embed_void
        from generation.core.domain import CylinderDomain
        from aog_policies import EmbeddingPolicy
        import trimesh
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        void_mesh = trimesh.creation.cylinder(radius=0.001, height=0.002)
        
        policy = EmbeddingPolicy(
            voxel_pitch=5e-4,
            preserve_ports_enabled=False,
        )
        
        solid, void_out, shell, report = embed_void(
            domain=domain,
            void_mesh=void_mesh,
            embedding_policy=policy,
        )
        
        report_dict = report.to_dict()
        json_str = json.dumps(report_dict)
        restored = json.loads(json_str)
        
        assert restored["operation"] == "embed_void"


class TestBuildComponentReturnsOperationReport:
    """Test build_component returns OperationReport."""
    
    def test_build_component_returns_operation_report(self):
        """Test build_component returns OperationReport."""
        from generation.api import build_component
        from generation.core.domain import CylinderDomain
        from aog_policies import GrowthPolicy, OperationReport
        
        domain = CylinderDomain(radius=0.005, height=0.003)
        ports = {
            "inlets": [{"position": (0, 0, 0.0015), "radius": 0.0005, "direction": (0, 0, -1), "vessel_type": "arterial"}],
            "outlets": [{"position": (0, 0, -0.0015), "radius": 0.0003, "direction": (0, 0, 1), "vessel_type": "venous"}],
        }
        growth_policy = GrowthPolicy(backend="cco_hybrid", max_iterations=10, target_terminals=5)
        
        component, report = build_component(
            domain=domain,
            ports=ports,
            growth_policy=growth_policy,
        )
        
        assert isinstance(report, OperationReport)
        assert report.requested_policy is not None
        assert report.effective_policy is not None


class TestOperationReportContract:
    """Test OperationReport contract."""
    
    def test_operation_report_has_required_fields(self):
        """Test OperationReport has required fields."""
        from aog_policies import OperationReport
        
        report = OperationReport(
            operation="test_op",
            requested_policy={"key": "value"},
            effective_policy={"key": "value", "extra": "field"},
        )
        
        assert hasattr(report, "operation")
        assert hasattr(report, "requested_policy")
        assert hasattr(report, "effective_policy")
    
    def test_operation_report_to_dict(self):
        """Test OperationReport.to_dict() method."""
        from aog_policies import OperationReport
        
        report = OperationReport(
            operation="test_op",
            requested_policy={"key": "value"},
            effective_policy={"key": "value", "extra": "field"},
        )
        
        d = report.to_dict()
        
        assert d["operation"] == "test_op"
        assert d["requested_policy"]["key"] == "value"
        assert d["effective_policy"]["extra"] == "field"
    
    def test_operation_report_json_round_trip(self):
        """Test OperationReport JSON round-trip."""
        import json
        from aog_policies import OperationReport
        
        report = OperationReport(
            operation="test_op",
            requested_policy={"key": "value"},
            effective_policy={"key": "value", "extra": "field"},
            metrics={"duration_s": 1.5, "voxel_count": 1000},
        )
        
        d = report.to_dict()
        json_str = json.dumps(d)
        restored = json.loads(json_str)
        
        assert restored["operation"] == "test_op"
        assert restored["metrics"]["duration_s"] == 1.5
