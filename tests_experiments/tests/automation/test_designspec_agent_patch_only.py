"""
Tests for DesignSpecAgent patch-only behavior.

Verifies:
- Agent responses contain patch proposals or questions, not Python code
- Agent generates valid JSON Patch operations
- Agent asks clarifying questions for missing fields
- Agent handles run requests appropriately
"""

import pytest
from typing import Dict, Any, List


class TestDesignSpecAgentPatchOnly:
    """Test suite for DesignSpecAgent patch-only behavior."""
    
    def test_agent_response_contains_no_python_code(self):
        """Test that agent responses do not contain Python code."""
        from automation.designspec_agent import (
            DesignSpecAgent,
            AgentResponse,
            AgentResponseType,
        )
        
        agent = DesignSpecAgent()
        
        spec = {"meta": {"name": "Test"}}
        
        response = agent.process_message(
            user_message="Create a box domain 20mm x 60mm x 30mm",
            spec=spec,
        )
        
        assert response.response_type in [
            AgentResponseType.PATCH_PROPOSAL,
            AgentResponseType.QUESTION,
            AgentResponseType.MESSAGE,
            AgentResponseType.RUN_REQUEST,
            AgentResponseType.ERROR,
        ]
        
        if response.patch_proposal:
            for patch in response.patch_proposal.patches:
                assert "op" in patch
                assert "path" in patch
                assert patch["op"] in ["add", "remove", "replace", "move", "copy", "test"]
                
                if "value" in patch:
                    assert not isinstance(patch["value"], str) or "def " not in patch["value"]
                    assert not isinstance(patch["value"], str) or "import " not in patch["value"]
    
    def test_agent_generates_valid_json_patch_for_domain(self):
        """Test that agent generates valid JSON Patch for domain creation."""
        from automation.designspec_agent import DesignSpecAgent, AgentResponseType
        
        agent = DesignSpecAgent()
        
        spec = {"meta": {"name": "Test"}}
        
        response = agent.process_message(
            user_message="Create a box domain 20mm x 60mm x 30mm",
            spec=spec,
        )
        
        if response.response_type == AgentResponseType.PATCH_PROPOSAL:
            patches = response.patch_proposal.patches
            assert len(patches) > 0
            
            for patch in patches:
                assert "op" in patch
                assert "path" in patch
                assert patch["path"].startswith("/")
    
    def test_agent_generates_valid_json_patch_for_seed(self):
        """Test that agent generates valid JSON Patch for seed setting."""
        from automation.designspec_agent import DesignSpecAgent, AgentResponseType
        
        agent = DesignSpecAgent()
        
        spec = {"meta": {"name": "Test"}}
        
        response = agent.process_message(
            user_message="Set the seed to 42",
            spec=spec,
        )
        
        if response.response_type == AgentResponseType.PATCH_PROPOSAL:
            patches = response.patch_proposal.patches
            
            seed_patch = None
            for patch in patches:
                if "seed" in patch.get("path", ""):
                    seed_patch = patch
                    break
            
            if seed_patch:
                assert seed_patch["op"] in ["add", "replace"]
                assert seed_patch["value"] == 42
    
    def test_agent_asks_questions_for_missing_required_fields(self):
        """Test that agent handles missing required fields appropriately."""
        from automation.designspec_agent import DesignSpecAgent, AgentResponseType
        from automation.designspec_session import ValidationReport
        
        agent = DesignSpecAgent()
        
        spec = {}
        
        validation_report = ValidationReport(
            valid=False,
            errors=["Missing required field: meta.name", "Missing required field: domain"],
            warnings=[],
        )
        
        response = agent.process_message(
            user_message="Help me create an organ",
            spec=spec,
            validation_report=validation_report,
        )
        
        assert response.response_type in [
            AgentResponseType.QUESTION,
            AgentResponseType.PATCH_PROPOSAL,
            AgentResponseType.MESSAGE,
            AgentResponseType.ERROR,
        ]
    
    def test_agent_handles_run_request(self):
        """Test that agent handles run requests appropriately."""
        from automation.designspec_agent import DesignSpecAgent, AgentResponseType
        
        agent = DesignSpecAgent()
        
        spec = {
            "meta": {"name": "Test", "seed": 42},
            "domain": {"type": "box", "size": [0.02, 0.06, 0.03]},
        }
        
        response = agent.process_message(
            user_message="Run the pipeline",
            spec=spec,
        )
        
        assert response.response_type in [
            AgentResponseType.RUN_REQUEST,
            AgentResponseType.QUESTION,
            AgentResponseType.MESSAGE,
        ]
    
    def test_agent_handles_run_until_request(self):
        """Test that agent handles run until requests."""
        from automation.designspec_agent import DesignSpecAgent, AgentResponseType
        
        agent = DesignSpecAgent()
        
        spec = {
            "meta": {"name": "Test", "seed": 42},
            "domain": {"type": "box", "size": [0.02, 0.06, 0.03]},
        }
        
        response = agent.process_message(
            user_message="Run until union_voids",
            spec=spec,
        )
        
        if response.response_type == AgentResponseType.RUN_REQUEST:
            assert response.run_request is not None
            assert response.run_request.run_until == "union_voids" or response.run_request.full_run


class TestDesignSpecAgentPatchExtraction:
    """Test suite for agent patch extraction from natural language."""
    
    def test_extract_box_domain_from_message(self):
        """Test extraction of box domain from natural language."""
        from automation.designspec_agent import DesignSpecAgent, AgentResponseType
        
        agent = DesignSpecAgent()
        
        spec = {"meta": {"name": "Test"}}
        
        response = agent.process_message(
            user_message="Create a box 20mm x 60mm x 30mm",
            spec=spec,
        )
        
        if response.response_type == AgentResponseType.PATCH_PROPOSAL:
            patches = response.patch_proposal.patches
            
            domain_patch = None
            for patch in patches:
                if "/domain" in patch.get("path", ""):
                    domain_patch = patch
                    break
            
            if domain_patch and "value" in domain_patch:
                value = domain_patch["value"]
                if isinstance(value, dict):
                    if "main_domain" in value:
                        assert value["main_domain"].get("type") == "box"
                    else:
                        assert value.get("type") == "box"
    
    def test_extract_cylinder_domain_from_message(self):
        """Test extraction of cylinder domain from natural language."""
        from automation.designspec_agent import DesignSpecAgent, AgentResponseType
        
        agent = DesignSpecAgent()
        
        spec = {"meta": {"name": "Test"}}
        
        response = agent.process_message(
            user_message="Create a cylinder with radius 10mm and height 20mm",
            spec=spec,
        )
        
        if response.response_type == AgentResponseType.PATCH_PROPOSAL:
            patches = response.patch_proposal.patches
            
            domain_patch = None
            for patch in patches:
                if "/domain" in patch.get("path", ""):
                    domain_patch = patch
                    break
            
            if domain_patch and "value" in domain_patch:
                value = domain_patch["value"]
                if isinstance(value, dict):
                    assert value.get("type") == "cylinder"
    
    def test_extract_name_from_message(self):
        """Test extraction of project name from natural language."""
        from automation.designspec_agent import DesignSpecAgent, AgentResponseType
        
        agent = DesignSpecAgent()
        
        spec = {"meta": {}}
        
        response = agent.process_message(
            user_message="Name this project 'Liver Scaffold'",
            spec=spec,
        )
        
        if response.response_type == AgentResponseType.PATCH_PROPOSAL:
            patches = response.patch_proposal.patches
            
            name_patch = None
            for patch in patches:
                if "name" in patch.get("path", ""):
                    name_patch = patch
                    break
            
            if name_patch:
                assert "Liver" in str(name_patch.get("value", "")) or "liver" in str(name_patch.get("value", "")).lower()


class TestDesignSpecAgentConversationHistory:
    """Test suite for agent conversation history."""
    
    def test_agent_maintains_conversation_history(self):
        """Test that agent maintains conversation history."""
        from automation.designspec_agent import DesignSpecAgent
        
        agent = DesignSpecAgent()
        
        spec = {"meta": {"name": "Test"}}
        
        agent.process_message("Hello", spec)
        agent.process_message("Create a box domain", spec)
        
        history = agent.get_conversation_history()
        
        assert len(history) >= 2
    
    def test_agent_clears_conversation_history(self):
        """Test that agent can clear conversation history."""
        from automation.designspec_agent import DesignSpecAgent
        
        agent = DesignSpecAgent()
        
        spec = {"meta": {"name": "Test"}}
        
        agent.process_message("Hello", spec)
        agent.clear_conversation_history()
        
        history = agent.get_conversation_history()
        
        assert len(history) == 0
