"""
Tests for the Single Agent Organ Generator V1 workflow.

These tests validate:
- Workflow state transitions
- Project context management
- Workflow serialization/deserialization
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
        
        assert WorkflowState.INIT.value == "init"
        assert WorkflowState.REQUIREMENTS.value == "requirements"
        assert WorkflowState.GENERATING.value == "generating"
        assert WorkflowState.VISUALIZING.value == "visualizing"
        assert WorkflowState.REVIEW.value == "review"
        assert WorkflowState.CLARIFYING.value == "clarifying"
        assert WorkflowState.FINALIZING.value == "finalizing"
        assert WorkflowState.COMPLETE.value == "complete"


class TestProjectContext:
    """Tests for ProjectContext dataclass."""
    
    def test_project_context_defaults(self):
        """Test ProjectContext default values."""
        from automation.workflow import ProjectContext
        
        ctx = ProjectContext()
        
        assert ctx.project_name == ""
        assert ctx.output_dir == ""
        assert ctx.description == ""
        assert ctx.output_units == "mm"
        assert ctx.spec_json is None
        assert ctx.network_json is None
        assert ctx.stl_path is None
        assert ctx.embedded_stl_path is None
        assert ctx.code_path is None
        assert ctx.iteration == 0
        assert ctx.feedback_history == []
    
    def test_project_context_creation(self):
        """Test creating a ProjectContext with values."""
        from automation.workflow import ProjectContext
        
        ctx = ProjectContext(
            project_name="test_project",
            output_dir="/tmp/test",
            description="Test description",
            output_units="cm",
            iteration=2,
        )
        
        assert ctx.project_name == "test_project"
        assert ctx.output_dir == "/tmp/test"
        assert ctx.description == "Test description"
        assert ctx.output_units == "cm"
        assert ctx.iteration == 2
    
    def test_project_context_to_dict(self):
        """Test ProjectContext serialization."""
        from automation.workflow import ProjectContext
        
        ctx = ProjectContext(
            project_name="test",
            output_dir="/tmp/test",
            description="desc",
            output_units="mm",
            iteration=1,
            feedback_history=["feedback1", "feedback2"],
        )
        
        d = ctx.to_dict()
        
        assert d["project_name"] == "test"
        assert d["output_dir"] == "/tmp/test"
        assert d["description"] == "desc"
        assert d["output_units"] == "mm"
        assert d["iteration"] == 1
        assert d["feedback_history"] == ["feedback1", "feedback2"]
    
    def test_project_context_from_dict(self):
        """Test ProjectContext deserialization."""
        from automation.workflow import ProjectContext
        
        d = {
            "project_name": "restored",
            "output_dir": "/tmp/restored",
            "description": "restored desc",
            "output_units": "um",
            "iteration": 3,
            "feedback_history": ["fb1"],
        }
        
        ctx = ProjectContext.from_dict(d)
        
        assert ctx.project_name == "restored"
        assert ctx.output_dir == "/tmp/restored"
        assert ctx.description == "restored desc"
        assert ctx.output_units == "um"
        assert ctx.iteration == 3
        assert ctx.feedback_history == ["fb1"]
    
    def test_project_context_roundtrip(self):
        """Test ProjectContext serialization roundtrip."""
        from automation.workflow import ProjectContext
        
        original = ProjectContext(
            project_name="roundtrip",
            output_dir="/tmp/roundtrip",
            description="roundtrip test",
            output_units="cm",
            spec_json="/tmp/spec.json",
            stl_path="/tmp/mesh.stl",
            iteration=5,
            feedback_history=["a", "b", "c"],
        )
        
        d = original.to_dict()
        restored = ProjectContext.from_dict(d)
        
        assert restored.project_name == original.project_name
        assert restored.output_dir == original.output_dir
        assert restored.description == original.description
        assert restored.output_units == original.output_units
        assert restored.spec_json == original.spec_json
        assert restored.stl_path == original.stl_path
        assert restored.iteration == original.iteration
        assert restored.feedback_history == original.feedback_history


class TestSingleAgentOrganGeneratorV1:
    """Tests for the SingleAgentOrganGeneratorV1 workflow class."""
    
    def test_workflow_creation(self):
        """Test creating a workflow instance."""
        from automation.workflow import SingleAgentOrganGeneratorV1, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(
            agent=mock_agent,
            base_output_dir="/tmp/test_projects",
            verbose=False,
        )
        
        assert workflow.agent == mock_agent
        assert workflow.base_output_dir == "/tmp/test_projects"
        assert workflow.verbose is False
        assert workflow.state == WorkflowState.INIT
    
    def test_workflow_name_and_version(self):
        """Test workflow name and version constants."""
        from automation.workflow import SingleAgentOrganGeneratorV1
        
        assert SingleAgentOrganGeneratorV1.WORKFLOW_NAME == "Single Agent Organ Generator V1"
        assert SingleAgentOrganGeneratorV1.WORKFLOW_VERSION == "1.0.0"
    
    def test_workflow_get_state(self):
        """Test getting current workflow state."""
        from automation.workflow import SingleAgentOrganGeneratorV1, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(agent=mock_agent, verbose=False)
        
        assert workflow.get_state() == WorkflowState.INIT
    
    def test_workflow_get_context(self):
        """Test getting current project context."""
        from automation.workflow import SingleAgentOrganGeneratorV1, ProjectContext
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(agent=mock_agent, verbose=False)
        
        ctx = workflow.get_context()
        assert isinstance(ctx, ProjectContext)
    
    def test_workflow_save_and_load_state(self):
        """Test saving and loading workflow state."""
        from automation.workflow import SingleAgentOrganGeneratorV1, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(agent=mock_agent, verbose=False)
        
        workflow.state = WorkflowState.REQUIREMENTS
        workflow.context.project_name = "saved_project"
        workflow.context.description = "saved description"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            workflow.save_state(filepath)
            
            workflow2 = SingleAgentOrganGeneratorV1(agent=mock_agent, verbose=False)
            workflow2.load_state(filepath)
            
            assert workflow2.state == WorkflowState.REQUIREMENTS
            assert workflow2.context.project_name == "saved_project"
            assert workflow2.context.description == "saved description"
        finally:
            os.unlink(filepath)
    
    def test_workflow_step_init_to_requirements(self):
        """Test workflow step from INIT to REQUIREMENTS."""
        from automation.workflow import SingleAgentOrganGeneratorV1, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(
            agent=mock_agent,
            base_output_dir="/tmp/test_projects",
            verbose=False,
        )
        
        new_state, message = workflow.step("my_project")
        
        assert new_state == WorkflowState.REQUIREMENTS
        assert workflow.context.project_name == "my_project"
        assert "my_project" in message
    
    def test_workflow_step_requirements_to_generating(self):
        """Test workflow step from REQUIREMENTS to GENERATING."""
        from automation.workflow import SingleAgentOrganGeneratorV1, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(agent=mock_agent, verbose=False)
        
        workflow.step("test_project")
        
        new_state, message = workflow.step("Generate a liver vascular network")
        
        assert new_state == WorkflowState.GENERATING
        assert workflow.context.description == "Generate a liver vascular network"
    
    def test_workflow_step_review_yes(self):
        """Test workflow step from REVIEW with 'yes' response."""
        from automation.workflow import SingleAgentOrganGeneratorV1, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(agent=mock_agent, verbose=False)
        
        workflow.state = WorkflowState.REVIEW
        
        new_state, message = workflow.step("yes")
        
        assert new_state == WorkflowState.FINALIZING
    
    def test_workflow_step_review_no(self):
        """Test workflow step from REVIEW with 'no' response."""
        from automation.workflow import SingleAgentOrganGeneratorV1, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(agent=mock_agent, verbose=False)
        
        workflow.state = WorkflowState.REVIEW
        
        new_state, message = workflow.step("no")
        
        assert new_state == WorkflowState.CLARIFYING
    
    def test_workflow_step_clarifying_to_generating(self):
        """Test workflow step from CLARIFYING to GENERATING."""
        from automation.workflow import SingleAgentOrganGeneratorV1, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(agent=mock_agent, verbose=False)
        
        workflow.state = WorkflowState.CLARIFYING
        
        new_state, message = workflow.step("Make the vessels thicker")
        
        assert new_state == WorkflowState.GENERATING
        assert "Make the vessels thicker" in workflow.context.feedback_history
    
    def test_workflow_step_quit(self):
        """Test workflow step with 'quit' command."""
        from automation.workflow import SingleAgentOrganGeneratorV1, WorkflowState
        
        mock_agent = Mock()
        workflow = SingleAgentOrganGeneratorV1(agent=mock_agent, verbose=False)
        
        new_state, message = workflow.step("quit")
        
        assert new_state == WorkflowState.COMPLETE
        assert "terminated" in message.lower()


class TestWorkflowIntegration:
    """Integration tests for the workflow."""
    
    def test_workflow_creates_output_directory(self):
        """Test that workflow creates output directory."""
        from automation.workflow import SingleAgentOrganGeneratorV1
        
        mock_agent = Mock()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            workflow = SingleAgentOrganGeneratorV1(
                agent=mock_agent,
                base_output_dir=tmpdir,
                verbose=False,
            )
            
            workflow.step("test_project")
            
            expected_dir = os.path.join(tmpdir, "test_project")
            assert os.path.exists(expected_dir)
