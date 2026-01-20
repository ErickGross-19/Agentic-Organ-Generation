"""
Tests for DesignSpec LLM Workflow Integration.

Tests cover:
1. Agent creation and basic functionality
2. Prompt builder functionality
3. Create LLM agent factory function
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Optional

import pytest

from automation.designspec_llm.directive import (
    DesignSpecDirective,
    Question,
    RunRequest,
    ContextRequest,
)
from automation.designspec_llm.agent import (
    DesignSpecLLMAgent,
    AgentTurnLog,
    create_llm_agent,
)
from automation.designspec_llm.context_builder import ContextBuilder, ContextPack, SpecSummary
from automation.designspec_llm.prompt_builder import PromptBuilder, get_system_prompt, build_user_prompt


class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_count = 0
        self.last_messages = None
    
    def chat(self, messages: list, **kwargs) -> str:
        self.last_messages = messages
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return response
        return json.dumps({
            "assistant_message": "Default response",
            "confidence": 0.5,
            "requires_approval": False,
            "stop": False,
        })


class TestDesignSpecLLMAgent:
    """Tests for DesignSpecLLMAgent class."""
    
    def test_agent_creation(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        mock_llm = MockLLMClient()
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=mock_llm,
        )
        
        assert agent.session == mock_session
        assert agent.llm_client == mock_llm
        assert agent.has_llm is True
    
    def test_agent_without_llm(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=None,
        )
        
        assert agent.has_llm is False
    
    def test_agent_conversation_history(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=None,
        )
        
        history = agent.get_conversation_history()
        assert isinstance(history, list)
        assert len(history) == 0
    
    def test_agent_clear_conversation_history(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=None,
        )
        
        agent.add_assistant_message("Test message")
        assert len(agent.get_conversation_history()) > 0
        
        agent.clear_conversation_history()
        assert len(agent.get_conversation_history()) == 0


class TestAgentTurnLog:
    """Tests for AgentTurnLog dataclass."""
    
    def test_turn_log_creation(self):
        log = AgentTurnLog(
            timestamp=1704067200.0,
            user_message="Create a domain",
            context_pack_hash="abc123",
            directive_json={"assistant_message": "OK"},
            patch_applied=False,
            run_executed=False,
        )
        
        assert log.timestamp == 1704067200.0
        assert log.user_message == "Create a domain"
        assert log.patch_applied is False
    
    def test_turn_log_to_dict(self):
        log = AgentTurnLog(
            timestamp=1704067200.0,
            user_message="Test",
            context_pack_hash="hash",
            directive_json={"assistant_message": "Response"},
        )
        
        d = log.to_dict()
        assert d["timestamp"] == 1704067200.0
        assert d["user_message"] == "Test"
        assert "directive_json" in d


class TestCreateLLMAgent:
    """Tests for create_llm_agent factory function."""
    
    def test_create_agent_with_llm(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        mock_llm = MockLLMClient()
        
        agent = create_llm_agent(
            session=mock_session,
            llm_client=mock_llm,
        )
        
        assert isinstance(agent, DesignSpecLLMAgent)
        assert agent.has_llm is True
    
    def test_create_agent_without_llm(self):
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        agent = create_llm_agent(
            session=mock_session,
            llm_client=None,
        )
        
        assert isinstance(agent, DesignSpecLLMAgent)
        assert agent.has_llm is False


class TestPromptBuilder:
    """Tests for PromptBuilder class."""
    
    def test_get_system_prompt(self):
        prompt = get_system_prompt()
        
        assert "DesignSpec" in prompt
        assert "JSON" in prompt
        assert len(prompt) > 100
    
    def test_build_user_prompt(self):
        context_pack = ContextPack(
            spec_summary=SpecSummary(has_domains=True, domain_count=1, domain_names=["main"]),
            compile_success=True,
        )
        
        prompt = build_user_prompt(
            user_message="Create a network component",
            context_pack=context_pack,
        )
        
        assert "Create a network component" in prompt
    
    def test_prompt_builder_class(self):
        builder = PromptBuilder()
        
        system_prompt = builder.system_prompt
        assert len(system_prompt) > 0
        assert "DesignSpec" in system_prompt
    
    def test_prompt_builder_user_prompt(self):
        builder = PromptBuilder()
        
        context_pack = ContextPack(
            spec_summary=SpecSummary(has_domains=False, domain_count=0),
            compile_success=False,
        )
        
        user_prompt = builder.build_user_prompt(
            user_message="Help me create a spec",
            context_pack=context_pack,
        )
        
        assert "Help me create a spec" in user_prompt


class TestDirectiveIntegration:
    """Tests for directive creation and validation."""
    
    def test_directive_creation_minimal(self):
        directive = DesignSpecDirective(
            assistant_message="I'll help you create a domain.",
            confidence=0.9,
            requires_approval=False,
        )
        
        assert directive.assistant_message == "I'll help you create a domain."
        assert directive.has_patches() is False
        assert directive.has_questions() is False
    
    def test_directive_with_patches(self):
        patches = [
            {"op": "add", "path": "/domains/main", "value": {"type": "box"}},
        ]
        directive = DesignSpecDirective(
            assistant_message="Adding domain",
            proposed_patches=patches,
            requires_approval=True,
        )
        
        assert directive.has_patches() is True
        assert len(directive.proposed_patches) == 1
    
    def test_directive_with_run_request(self):
        directive = DesignSpecDirective(
            assistant_message="Let's run the pipeline",
            run_request=RunRequest(run=True, run_until="validity"),
            requires_approval=True,
        )
        
        assert directive.has_run_request() is True
        assert directive.run_request.run_until == "validity"
    
    def test_directive_to_dict_roundtrip(self):
        original = DesignSpecDirective(
            assistant_message="Test message",
            questions=[Question(id="q1", question="Test?")],
            proposed_patches=[{"op": "add", "path": "/test", "value": 1}],
            confidence=0.85,
            requires_approval=True,
        )
        
        d = original.to_dict()
        restored = DesignSpecDirective.from_dict(d)
        
        assert restored.assistant_message == original.assistant_message
        assert len(restored.questions) == len(original.questions)
        assert len(restored.proposed_patches) == len(original.proposed_patches)
