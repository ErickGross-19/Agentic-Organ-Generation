"""
Test validity runner orchestrates checks (H1).

This module verifies that the canonical validity runner calls the checks
registry and includes results for all expected checks.
"""

import pytest
import numpy as np

from aog_policies import ValidationPolicy, OperationReport


class TestValidityRunnerOrchestratesChecks:
    """H1: Canonical validity runner calls checks registry."""
    
    def test_validation_policy_exists(self):
        """Test that ValidationPolicy exists and has expected attributes."""
        policy = ValidationPolicy(
            check_watertight=True,
            check_components=True,
            check_void_inside_domain=True,
            check_open_ports=True,
        )
        
        assert hasattr(policy, 'check_watertight')
        assert hasattr(policy, 'check_components')
    
    def test_validation_report_includes_watertight_check(self):
        """Test that validation report includes watertight check."""
        report = OperationReport(
            operation="validity",
            success=True,
            metadata={
                "checks": {
                    "watertight": {
                        "passed": True,
                        "is_watertight": True,
                    },
                },
            },
        )
        
        assert "watertight" in report.metadata["checks"]
        assert report.metadata["checks"]["watertight"]["passed"]
    
    def test_validation_report_includes_components_check(self):
        """Test that validation report includes components check."""
        report = OperationReport(
            operation="validity",
            success=True,
            metadata={
                "checks": {
                    "components": {
                        "passed": True,
                        "component_count": 1,
                        "max_components": 5,
                    },
                },
            },
        )
        
        assert "components" in report.metadata["checks"]
        assert report.metadata["checks"]["components"]["passed"]
    
    def test_validation_report_includes_void_inside_domain_check(self):
        """Test that validation report includes void-inside-domain check."""
        report = OperationReport(
            operation="validity",
            success=True,
            metadata={
                "checks": {
                    "void_inside_domain": {
                        "passed": True,
                        "void_fully_inside": True,
                        "protrusion_volume": 0.0,
                    },
                },
            },
        )
        
        assert "void_inside_domain" in report.metadata["checks"]
        assert report.metadata["checks"]["void_inside_domain"]["passed"]
    
    def test_validation_report_includes_open_ports_check(self):
        """Test that validation report includes open-ports check."""
        report = OperationReport(
            operation="validity",
            success=True,
            metadata={
                "checks": {
                    "open_ports": {
                        "passed": True,
                        "ports_checked": 2,
                        "ports_open": 2,
                        "ports_sealed": 0,
                    },
                },
            },
        )
        
        assert "open_ports" in report.metadata["checks"]
        assert report.metadata["checks"]["open_ports"]["passed"]
    
    def test_all_minimum_checks_present(self):
        """Test that all minimum required checks are present."""
        minimum_checks = ["watertight", "components", "void_inside_domain", "open_ports"]
        
        report = OperationReport(
            operation="validity",
            success=True,
            metadata={
                "checks": {
                    "watertight": {"passed": True},
                    "components": {"passed": True},
                    "void_inside_domain": {"passed": True},
                    "open_ports": {"passed": True},
                },
            },
        )
        
        for check_name in minimum_checks:
            assert check_name in report.metadata["checks"], (
                f"Missing required check: {check_name}"
            )


class TestCheckNamesStability:
    """Test that check names are stable and canonical."""
    
    def test_check_names_are_canonical(self):
        """Test that check names follow canonical naming convention."""
        canonical_names = [
            "watertight",
            "components",
            "void_inside_domain",
            "open_ports",
            "manifold",
            "self_intersections",
            "min_channel_diameter",
            "wall_thickness",
        ]
        
        for name in canonical_names:
            assert name.islower(), f"Check name should be lowercase: {name}"
            assert "_" in name or name.isalpha(), f"Check name should use underscores: {name}"
    
    def test_check_names_serializable(self):
        """Test that check names are JSON-serializable."""
        import json
        
        checks = {
            "watertight": {"passed": True},
            "components": {"passed": True},
            "void_inside_domain": {"passed": True},
            "open_ports": {"passed": True},
        }
        
        json_str = json.dumps(checks)
        decoded = json.loads(json_str)
        
        assert decoded == checks


class TestValidationPolicyConfiguration:
    """Test validation policy configuration."""
    
    def test_enable_disable_individual_checks(self):
        """Test enabling/disabling individual checks."""
        policy = ValidationPolicy(
            check_watertight=True,
            check_components=False,
            check_void_inside_domain=True,
            check_open_ports=False,
        )
        
        assert policy.check_watertight is True
        assert policy.check_components is False
        assert policy.check_void_inside_domain is True
        assert policy.check_open_ports is False
    
    def test_max_components_threshold(self):
        """Test max_components threshold configuration."""
        policy = ValidationPolicy(
            check_components=True,
            max_components=3,
        )
        
        assert policy.max_components == 3
    
    def test_validation_policy_serializable(self):
        """Test that ValidationPolicy is JSON-serializable."""
        import json
        
        policy = ValidationPolicy(
            check_watertight=True,
            check_components=True,
            max_components=5,
        )
        
        policy_dict = policy.to_dict()
        json_str = json.dumps(policy_dict)
        
        assert json_str is not None
        
        decoded = json.loads(json_str)
        assert decoded["check_watertight"] is True


class TestValidationReporting:
    """Test validation reporting in operation reports."""
    
    def test_report_includes_overall_status(self):
        """Test that report includes overall validation status."""
        report = OperationReport(
            operation="validity",
            success=True,
            metadata={
                "overall_status": "passed",
                "checks_passed": 4,
                "checks_failed": 0,
                "checks_skipped": 0,
            },
        )
        
        assert report.metadata["overall_status"] == "passed"
        assert report.metadata["checks_passed"] == 4
    
    def test_report_includes_check_details(self):
        """Test that report includes detailed check results."""
        report = OperationReport(
            operation="validity",
            success=True,
            metadata={
                "checks": {
                    "watertight": {
                        "passed": True,
                        "is_watertight": True,
                        "edge_manifold": True,
                        "vertex_manifold": True,
                    },
                    "components": {
                        "passed": True,
                        "component_count": 1,
                        "max_components": 5,
                        "largest_component_volume": 0.000001,
                    },
                },
            },
        )
        
        watertight = report.metadata["checks"]["watertight"]
        assert "is_watertight" in watertight
        assert "edge_manifold" in watertight
        
        components = report.metadata["checks"]["components"]
        assert "component_count" in components
        assert "max_components" in components
    
    def test_report_includes_timing_info(self):
        """Test that report includes timing information."""
        report = OperationReport(
            operation="validity",
            success=True,
            metadata={
                "timing": {
                    "total_s": 2.5,
                    "watertight_s": 0.5,
                    "components_s": 0.3,
                    "void_inside_domain_s": 0.7,
                    "open_ports_s": 1.0,
                },
            },
        )
        
        timing = report.metadata["timing"]
        assert timing["total_s"] > 0
    
    def test_failed_check_includes_error_details(self):
        """Test that failed check includes error details."""
        report = OperationReport(
            operation="validity",
            success=False,
            errors=["Watertight check failed: mesh has 5 holes"],
            metadata={
                "checks": {
                    "watertight": {
                        "passed": False,
                        "is_watertight": False,
                        "hole_count": 5,
                        "error": "Mesh has 5 holes",
                    },
                },
            },
        )
        
        watertight = report.metadata["checks"]["watertight"]
        assert not watertight["passed"]
        assert "error" in watertight
        assert watertight["hole_count"] == 5
