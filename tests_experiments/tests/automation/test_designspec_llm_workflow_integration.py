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


class MockLLMClientThatRaises:
    """Mock LLM client that raises an exception."""
    
    def __init__(self, exception_type=Exception, exception_message="Test error"):
        self.exception_type = exception_type
        self.exception_message = exception_message
        self.call_count = 0
    
    def chat(self, messages: list, **kwargs) -> str:
        self.call_count += 1
        raise self.exception_type(self.exception_message)


class MockLegacyAgent:
    """Mock legacy agent for testing fallback behavior."""
    
    def __init__(self, response_type="PATCH_PROPOSAL"):
        self.response_type = response_type
        self.process_message_called = False
        self._conversation_history = []
    
    def process_message(self, user_message, spec, validation_report=None, compile_report=None):
        from automation.designspec_agent import AgentResponse, AgentResponseType
        from automation.designspec_session import PatchProposal
        
        self.process_message_called = True
        
        if self.response_type == "PATCH_PROPOSAL":
            return AgentResponse(
                response_type=AgentResponseType.PATCH_PROPOSAL,
                message="Legacy agent response",
                patch_proposal=PatchProposal(
                    patch_id="test_patch",
                    patches=[{"op": "add", "path": "/test", "value": 1}],
                    explanation="Legacy patch explanation",
                ),
            )
        elif self.response_type == "QUESTION":
            from automation.designspec_agent import Question
            return AgentResponse(
                response_type=AgentResponseType.QUESTION,
                message="Legacy question response",
                questions=[Question(
                    field_path="/test",
                    question_text="What value?",
                )],
            )
        else:
            return AgentResponse(
                response_type=AgentResponseType.MESSAGE,
                message="Legacy message response",
            )


class TestLLMFallbackWarning:
    """Tests for LLM fallback warning behavior."""
    
    def test_fallback_notice_in_message_on_llm_exception(self):
        """When LLM raises exception, fallback notice should appear in response message."""
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        mock_llm = MockLLMClientThatRaises(
            exception_type=Exception,
            exception_message="API connection failed"
        )
        mock_legacy = MockLegacyAgent(response_type="MESSAGE")
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=mock_llm,
            legacy_agent=mock_legacy,
        )
        
        response = agent.process_message("Create a domain")
        
        # Verify fallback notice is in the message
        assert "LLM failed" in response.message
        assert "Falling back to legacy parser" in response.message
        assert "Exception" in response.message
        assert mock_legacy.process_message_called
    
    def test_fallback_notice_in_patch_proposal_explanation(self):
        """When LLM fails and legacy returns patch proposal, notice should be in explanation."""
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        mock_llm = MockLLMClientThatRaises(
            exception_type=ValueError,
            exception_message="Invalid response format"
        )
        mock_legacy = MockLegacyAgent(response_type="PATCH_PROPOSAL")
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=mock_llm,
            legacy_agent=mock_legacy,
        )
        
        response = agent.process_message("Add a component")
        
        # Verify fallback notice is in both message and patch explanation
        assert "LLM failed" in response.message
        assert response.patch_proposal is not None
        assert "LLM failed" in response.patch_proposal.explanation
        assert "Falling back to legacy parser" in response.patch_proposal.explanation
    
    def test_fallback_notice_sanitizes_long_error_messages(self):
        """Error messages should be truncated to avoid leaking too much info."""
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        # Create a very long error message
        long_error = "A" * 500
        mock_llm = MockLLMClientThatRaises(
            exception_type=Exception,
            exception_message=long_error
        )
        mock_legacy = MockLegacyAgent(response_type="MESSAGE")
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=mock_llm,
            legacy_agent=mock_legacy,
        )
        
        response = agent.process_message("Test message")
        
        # The error message should be truncated (200 chars max + "...")
        assert "LLM failed" in response.message
        # The full 500-char error should not appear
        assert long_error not in response.message
        assert "..." in response.message  # Should be truncated
    
    def test_fallback_notice_redacts_potential_api_keys(self):
        """Potential API keys in error messages should be redacted."""
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        # Error message containing something that looks like an API key
        api_key_like = "sk-" + "a" * 48  # 51 chars total, looks like OpenAI key
        mock_llm = MockLLMClientThatRaises(
            exception_type=Exception,
            exception_message=f"Auth failed with key {api_key_like}"
        )
        mock_legacy = MockLegacyAgent(response_type="MESSAGE")
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=mock_llm,
            legacy_agent=mock_legacy,
        )
        
        response = agent.process_message("Test message")
        
        # The API key should be redacted
        assert api_key_like not in response.message
        assert "[REDACTED]" in response.message
    
    def test_turn_log_saved_on_fallback(self):
        """Turn logs should be saved even when falling back to legacy agent."""
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        mock_llm = MockLLMClientThatRaises()
        mock_legacy = MockLegacyAgent(response_type="MESSAGE")
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=mock_llm,
            legacy_agent=mock_legacy,
        )
        
        # Mock _save_turn_log to track if it's called
        save_called = []
        original_save = agent._save_turn_log
        def mock_save(turn_log):
            save_called.append(turn_log)
        agent._save_turn_log = mock_save
        
        response = agent.process_message("Test message")
        
        # Verify turn log was saved
        assert len(save_called) == 1
        assert save_called[0].user_message == "Test message"
        assert len(save_called[0].errors) > 0  # Should have recorded the error
    
    def test_conversation_history_synced_to_legacy_agent(self):
        """Conversation history should be synced to legacy agent on fallback."""
        mock_session = MagicMock()
        mock_session.get_spec.return_value = {}
        mock_session.project_dir = Path("/tmp/test")
        
        mock_llm = MockLLMClientThatRaises()
        mock_legacy = MockLegacyAgent(response_type="MESSAGE")
        
        agent = DesignSpecLLMAgent(
            session=mock_session,
            llm_client=mock_llm,
            legacy_agent=mock_legacy,
        )
        
        # Add some history before the failing call
        agent._conversation_history = [
            {"role": "user", "content": "Previous message"},
            {"role": "assistant", "content": "Previous response"},
        ]
        
        response = agent.process_message("New message")
        
        # Legacy agent should have received the conversation history
        # (including the new message that was added before fallback)
        assert len(mock_legacy._conversation_history) >= 2
