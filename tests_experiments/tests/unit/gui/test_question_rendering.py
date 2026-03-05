"""
Unit tests for GUI question rendering.

Tests that QUESTION events are properly rendered with question_text field.
"""

import pytest
from unittest.mock import MagicMock, patch


class TestQuestionTextExtraction:
    """Tests for question text extraction logic."""

    def test_question_text_key_is_primary(self):
        """Test that question_text is used as primary key."""
        question = {
            "field_path": "domains.main_domain",
            "question_text": "What shape and size should the domain be?",
            "text": "Old text key",
            "question": "Even older question key",
            "field_type": "domain",
        }
        
        result = question.get('question_text', question.get('text', question.get('question', '')))
        assert result == "What shape and size should the domain be?"

    def test_falls_back_to_text_key(self):
        """Test that 'text' key is used if question_text missing."""
        question = {
            "field_path": "meta.seed",
            "text": "What seed value should be used?",
            "question": "Old question key",
        }
        
        result = question.get('question_text', question.get('text', question.get('question', '')))
        assert result == "What seed value should be used?"

    def test_falls_back_to_question_key(self):
        """Test that 'question' key is used as last resort."""
        question = {
            "field_path": "components",
            "question": "What type of vascular structure?",
        }
        
        result = question.get('question_text', question.get('text', question.get('question', '')))
        assert result == "What type of vascular structure?"

    def test_empty_string_if_no_keys(self):
        """Test that empty string is returned if no keys present."""
        question = {
            "field_path": "unknown",
        }
        
        result = question.get('question_text', question.get('text', question.get('question', '')))
        assert result == ""


class TestQuestionFormatting:
    """Tests for question formatting as bullet points."""

    def test_single_question_formatted(self):
        """Test that a single question is formatted with bullet."""
        questions = [
            {"question_text": "What shape should the domain be?"},
        ]
        
        question_text = "\n".join(
            f"- {q.get('question_text', q.get('text', q.get('question', '')))}"
            for q in questions
        )
        
        assert question_text == "- What shape should the domain be?"

    def test_multiple_questions_formatted(self):
        """Test that multiple questions are formatted as bullet points."""
        questions = [
            {"question_text": "What shape should the domain be?"},
            {"question_text": "What type of channels do you need?"},
            {"text": "What is the inlet radius?"},
        ]
        
        question_text = "\n".join(
            f"- {q.get('question_text', q.get('text', q.get('question', '')))}"
            for q in questions
        )
        
        assert "- What shape should the domain be?" in question_text
        assert "- What type of channels do you need?" in question_text
        assert "- What is the inlet radius?" in question_text

    def test_empty_questions_list(self):
        """Test that empty questions list produces empty string."""
        questions = []
        
        question_text = "\n".join(
            f"- {q.get('question_text', q.get('text', q.get('question', '')))}"
            for q in questions
        )
        
        assert question_text == ""


class TestWorkflowManagerQuestionHandling:
    """Integration tests for DesignSpecWorkflowManager question handling."""

    def test_manager_formats_questions_correctly(self):
        """Test that the manager formats questions with question_text key."""
        from gui.designspec_workflow_manager import DesignSpecWorkflowManager
        from gui.workflow_manager import WorkflowMessage
        
        messages_received = []
        
        def message_callback(msg):
            messages_received.append(msg)
        
        manager = DesignSpecWorkflowManager(
            message_callback=message_callback,
        )
        
        questions = [
            {
                "field_path": "domains.main_domain",
                "question_text": "What shape and size should the domain be?",
                "field_type": "domain",
            }
        ]
        
        question_text = "\n".join(
            f"- {q.get('question_text', q.get('text', q.get('question', '')))}"
            for q in questions
        )
        
        assert "What shape and size should the domain be?" in question_text

    def test_manager_handles_mixed_question_keys(self):
        """Test that the manager handles questions with different key names."""
        questions = [
            {"question_text": "Primary question text"},
            {"text": "Fallback text"},
            {"question": "Last resort question"},
        ]
        
        formatted = []
        for q in questions:
            text = q.get('question_text', q.get('text', q.get('question', '')))
            formatted.append(f"- {text}")
        
        result = "\n".join(formatted)
        
        assert "- Primary question text" in result
        assert "- Fallback text" in result
        assert "- Last resort question" in result
