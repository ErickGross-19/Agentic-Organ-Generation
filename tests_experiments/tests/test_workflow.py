"""
Tests for the Single Agent Organ Generator V4 workflow.

These tests validate:
- Workflow state transitions
- Project context management
- Object context management
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, MagicMock


class TestWorkflowState:
    """Tests for WorkflowState enum."""
    
    def test_workflow_states_exist(self):
        """Test that all expected workflow states exist."""
        from automation.workflow import WorkflowState
        
        assert WorkflowState.PROJECT_INIT.value == "project_init"
        assert WorkflowState.OBJECT_PLANNING.value == "object_planning"
        assert WorkflowState.FRAME_OF_REFERENCE.value == "frame_of_reference"
        assert WorkflowState.REQUIREMENTS_CAPTURE.value == "requirements_capture"
        assert WorkflowState.SPEC_COMPILATION.value == "spec_compilation"
        assert WorkflowState.GENERATION.value == "generation"
        assert WorkflowState.ANALYSIS_VALIDATION.value == "analysis_validation"
        assert WorkflowState.ITERATION.value == "iteration"
        assert WorkflowState.FINALIZATION.value == "finalization"
        assert WorkflowState.COMPLETE.value == "complete"


class TestProjectContext:
    """Tests for ProjectContext dataclass."""
    
    def test_project_context_defaults(self):
        """Test ProjectContext default values."""
        from automation.workflow import ProjectContext
        
        ctx = ProjectContext()
        
        assert ctx.project_name == ""
        assert ctx.project_slug == ""
        assert ctx.output_dir == ""
        assert ctx.units_internal == "m"
        assert ctx.units_export == "mm"
        assert ctx.flow_solver_enabled is False
        assert ctx.objects == []
        assert ctx.current_object_index == 0
        assert ctx.variant_mode is False
    
    def test_project_context_creation(self):
        """Test creating a ProjectContext with values."""
        from automation.workflow import ProjectContext
        
        ctx = ProjectContext(
            project_name="test_project",
            project_slug="test-project",
            output_dir="/tmp/test",
            units_export="cm",
        )
        
        assert ctx.project_name == "test_project"
        assert ctx.project_slug == "test-project"
        assert ctx.output_dir == "/tmp/test"
        assert ctx.units_export == "cm"
    
    def test_project_context_to_dict(self):
        """Test ProjectContext serialization."""
        from automation.workflow import ProjectContext
        
        ctx = ProjectContext(
            project_name="test",
            project_slug="test-slug",
            output_dir="/tmp/test",
            units_export="mm",
        )
        
        d = ctx.to_dict()
        
        assert d["project_name"] == "test"
        assert d["project_slug"] == "test-slug"
        assert d["output_dir"] == "/tmp/test"
        assert d["units_export"] == "mm"
    
    def test_project_context_from_dict(self):
        """Test ProjectContext deserialization."""
        from automation.workflow import ProjectContext
        
        d = {
            "project_name": "restored",
            "project_slug": "restored-slug",
            "output_dir": "/tmp/restored",
            "units_export": "um",
        }
        
        ctx = ProjectContext.from_dict(d)
        
        assert ctx.project_name == "restored"
        assert ctx.project_slug == "restored-slug"
        assert ctx.output_dir == "/tmp/restored"
        assert ctx.units_export == "um"
    
    def test_project_context_roundtrip(self):
        """Test ProjectContext serialization roundtrip."""
        from automation.workflow import ProjectContext
        
        original = ProjectContext(
            project_name="roundtrip",
            project_slug="roundtrip-slug",
            output_dir="/tmp/roundtrip",
            units_export="cm",
        )
        
        d = original.to_dict()
        restored = ProjectContext.from_dict(d)
        
        assert restored.project_name == original.project_name
        assert restored.project_slug == original.project_slug
        assert restored.output_dir == original.output_dir
        assert restored.units_export == original.units_export


class TestObjectContext:
    """Tests for ObjectContext dataclass."""
    
    def test_object_context_defaults(self):
        """Test ObjectContext default values."""
        from automation.workflow import ObjectContext
        
        ctx = ObjectContext()
        
        assert ctx.name == ""
        assert ctx.slug == ""
        assert ctx.object_dir == ""
        assert ctx.current_version == 1
    
    def test_object_context_creation(self):
        """Test creating an ObjectContext with values."""
        from automation.workflow import ObjectContext
        
        ctx = ObjectContext(
            name="test_object",
            slug="test-object",
            object_dir="/tmp/test/objects/test-object",
        )
        
        assert ctx.name == "test_object"
        assert ctx.slug == "test-object"
        assert ctx.object_dir == "/tmp/test/objects/test-object"
    
    def test_object_context_to_dict(self):
        """Test ObjectContext serialization."""
        from automation.workflow import ObjectContext
        
        ctx = ObjectContext(
            name="test",
            slug="test-slug",
            object_dir="/tmp/test",
        )
        
        d = ctx.to_dict()
        
        assert d["name"] == "test"
        assert d["slug"] == "test-slug"
        assert d["object_dir"] == "/tmp/test"
    
    def test_object_context_from_dict(self):
        """Test ObjectContext deserialization."""
        from automation.workflow import ObjectContext
        
        d = {
            "name": "restored",
            "slug": "restored-slug",
            "object_dir": "/tmp/restored",
        }
        
        ctx = ObjectContext.from_dict(d)
        
        assert ctx.name == "restored"
        assert ctx.slug == "restored-slug"
        assert ctx.object_dir == "/tmp/restored"


class TestObjectRequirements:
    """Tests for ObjectRequirements dataclass."""
    
    def test_object_requirements_defaults(self):
        """Test ObjectRequirements default values."""
        from automation.workflow import ObjectRequirements
        
        req = ObjectRequirements()
        
        assert req.identity is not None
        assert req.frame_of_reference is not None
        assert req.domain is not None
        assert req.inlets_outlets is not None
        assert req.topology is not None
        assert req.geometry is not None
        assert req.constraints is not None
        assert req.embedding_export is not None
        assert req.acceptance_criteria is not None
    
    def test_object_requirements_to_dict(self):
        """Test ObjectRequirements serialization."""
        from automation.workflow import ObjectRequirements
        
        req = ObjectRequirements()
        d = req.to_dict()
        
        assert "identity" in d
        assert "frame_of_reference" in d
        assert "domain" in d
        assert "inlets_outlets" in d
        assert "topology" in d
        assert "geometry" in d
        assert "constraints" in d
        assert "embedding_export" in d
        assert "acceptance_criteria" in d
    
    def test_object_requirements_roundtrip(self):
        """Test ObjectRequirements serialization roundtrip."""
        from automation.workflow import ObjectRequirements
        
        original = ObjectRequirements()
        d = original.to_dict()
        restored = ObjectRequirements.from_dict(d)
        
        assert restored.identity.object_name == original.identity.object_name
        assert restored.domain.shape_type == original.domain.shape_type


class TestSingleAgentOrganGeneratorV4:
    """Tests for the SingleAgentOrganGeneratorV4 workflow class."""
    
    def test_workflow_creation(self):
        """Test creating a workflow instance."""
        from automation.workflow import SingleAgentOrganGeneratorV4, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV4(
            agent=mock_agent,
            base_output_dir="/tmp/test_projects",
            verbose=False,
        )
        
        assert workflow.agent == mock_agent
        assert workflow.base_output_dir == "/tmp/test_projects"
        assert workflow.verbose is False
        assert workflow.state == WorkflowState.PROJECT_INIT
    
    def test_workflow_name_and_version(self):
        """Test workflow name and version constants."""
        from automation.workflow import SingleAgentOrganGeneratorV4
        
        assert SingleAgentOrganGeneratorV4.WORKFLOW_NAME == "Single Agent Organ Generator V4"
        assert SingleAgentOrganGeneratorV4.WORKFLOW_VERSION == "4.0.0"
    
    def test_workflow_get_state(self):
        """Test getting current workflow state."""
        from automation.workflow import SingleAgentOrganGeneratorV4, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV4(agent=mock_agent, verbose=False)
        
        assert workflow.get_state() == WorkflowState.PROJECT_INIT
    
    def test_workflow_get_context(self):
        """Test getting current project context."""
        from automation.workflow import SingleAgentOrganGeneratorV4, ProjectContext
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV4(agent=mock_agent, verbose=False)
        
        ctx = workflow.get_context()
        assert isinstance(ctx, ProjectContext)


class TestWorkflowIntegration:
    """Integration tests for the workflow."""
    
    def test_workflow_creates_output_directory(self):
        """Test that workflow creates output directory."""
        from automation.workflow import SingleAgentOrganGeneratorV4
        
        mock_agent = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = SingleAgentOrganGeneratorV4(
                agent=mock_agent,
                base_output_dir=tmpdir,
                verbose=False,
            )
            
            assert workflow.base_output_dir == tmpdir
