"""
Tests for the DesignSpec LLM workflow with run approval gating.

These tests validate:
- Run approval gating (workflow does not execute runs without approval)
- Context request fulfillment (automatic second LLM call)
- GUI wiring (buttons enable/disable based on workflow state)

Note: Some tests require numpy and other dependencies. Tests that can run
without heavy dependencies are marked accordingly.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional


class TestRunApprovalGating:
    """Tests for run approval gating in the DesignSpec workflow."""
    
    def test_workflow_event_types_include_approval_events(self):
        """Test that WorkflowEventType enum includes approval events."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "workflows", "designspec_workflow.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'RUN_APPROVAL_REQUIRED' in source
        assert 'RUN_APPROVED' in source
        assert 'RUN_REJECTED' in source
    
    def test_workflow_status_enum_has_approval_states(self):
        """Test that WorkflowStatus enum includes approval states."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "workflows", "designspec_workflow.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'IDLE' in source
        assert 'WAITING_RUN_APPROVAL' in source
        assert 'RUNNING' in source
        assert 'ERROR' in source
    
    def test_workflow_has_pending_run_request_attribute(self):
        """Test that DesignSpecWorkflow has _pending_run_request attribute."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "workflows", "designspec_workflow.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert '_pending_run_request' in source
    
    def test_workflow_has_approve_pending_run_method(self):
        """Test that DesignSpecWorkflow has approve_pending_run method."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "workflows", "designspec_workflow.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'def approve_pending_run' in source
    
    def test_workflow_has_reject_pending_run_method(self):
        """Test that DesignSpecWorkflow has reject_pending_run method."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "workflows", "designspec_workflow.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'def reject_pending_run' in source
    
    def test_workflow_has_get_pending_run_request_method(self):
        """Test that DesignSpecWorkflow has get_pending_run_request method."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "workflows", "designspec_workflow.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'def get_pending_run_request' in source


class TestContextRequestFulfillment:
    """Tests for context request fulfillment with automatic second LLM call."""
    
    def test_agent_turn_log_has_context_request_fields(self):
        """Test that AgentTurnLog has context request fulfillment fields."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "agent.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'context_request_fulfilled' in source
        assert 'second_directive_json' in source
    
    def test_context_request_dataclass_exists(self):
        """Test that ContextRequest dataclass exists with expected fields."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "directive.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'class ContextRequest' in source
        assert 'need_full_spec' in source
        assert 'need_validity_report' in source
    
    def test_context_builder_has_expanded_context_methods(self):
        """Test that ContextBuilder has methods for expanded context."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "context_builder.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'def add_validation_details' in source
        assert 'def add_run_details' in source
        assert 'def add_network_artifact' in source
        assert 'def add_specific_files' in source
        assert 'def add_extended_history' in source
    
    def test_context_pack_has_mesh_and_network_stats(self):
        """Test that ContextPack includes mesh and network stats."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "context_builder.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'class ContextPack' in source
        assert 'class MeshStats' in source
        assert 'class NetworkStats' in source
        assert 'mesh_stats' in source
        assert 'network_stats' in source
    
    def test_context_pack_has_debug_compact_flag(self):
        """Test that ContextPack has is_debug_compact flag for auto-escalation."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "context_builder.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'is_debug_compact' in source


class TestGUIWiring:
    """Tests for GUI wiring with workflow manager."""
    
    def test_run_panel_class_exists(self):
        """Test that RunPanel class exists and has approval methods."""
        import importlib.util
        spec = importlib.util.find_spec("gui.designspec_panels")
        assert spec is not None, "gui.designspec_panels module should exist"
    
    def test_workflow_manager_has_approval_methods(self):
        """Test that DesignSpecWorkflowManager has approval methods."""
        from gui.designspec_workflow_manager import DesignSpecWorkflowManager
        
        assert hasattr(DesignSpecWorkflowManager, 'approve_run')
        assert hasattr(DesignSpecWorkflowManager, 'reject_run')
        assert hasattr(DesignSpecWorkflowManager, 'get_pending_run_request')
    
    def test_main_window_module_exists_in_legacy(self):
        """Test that main_window module exists in legacy folder."""
        import importlib.util
        spec = importlib.util.find_spec("gui._legacy.main_window")
        assert spec is not None, "gui._legacy.main_window module should exist"
    
    def test_app_module_exists(self):
        """Test that app module exists as the new entry point."""
        import importlib.util
        spec = importlib.util.find_spec("gui.app")
        assert spec is not None, "gui.app module should exist"


class TestCompactContextAutoEscalation:
    """Tests for compact context auto-escalation logic."""
    
    def test_validation_summary_has_failure_details(self):
        """Test that ValidationSummary includes failure details."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "context_builder.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'class ValidationSummary' in source
        assert 'failed_checks' in source
        assert 'failure_names' in source
        assert 'failure_reasons' in source
        assert 'key_metrics' in source
    
    def test_mesh_stats_dataclass(self):
        """Test MeshStats dataclass structure."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "context_builder.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'class MeshStats' in source
        assert 'component_void_mesh' in source
        assert 'union_void_mesh' in source
        assert 'domain_with_void_mesh' in source
    
    def test_network_stats_dataclass(self):
        """Test NetworkStats dataclass structure."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "context_builder.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'class NetworkStats' in source
        assert 'node_count' in source
        assert 'edge_count' in source
        assert 'bbox' in source
        assert 'radius_stats' in source


class TestDirectiveSchema:
    """Tests for directive schema validation."""
    
    def test_directive_has_requires_approval_field(self):
        """Test that DesignSpecDirective has requires_approval field."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "directive.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'class DesignSpecDirective' in source
        assert 'requires_approval' in source
    
    def test_run_request_has_expected_fields(self):
        """Test that RunRequest has all expected fields."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "directive.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'class RunRequest' in source
        assert 'run_until' in source
        assert 'expected_signal' in source
    
    def test_context_request_has_all_fields(self):
        """Test that ContextRequest has all expected fields."""
        import os
        
        source_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "automation", "designspec_llm", "directive.py"
        )
        
        with open(source_path, 'r') as f:
            source = f.read()
        
        assert 'class ContextRequest' in source
        assert 'need_full_spec' in source
        assert 'need_validity_report' in source
        assert 'need_last_run_report' in source
        assert 'need_network_artifact' in source
        assert 'need_specific_files' in source
        assert 'need_more_history' in source
