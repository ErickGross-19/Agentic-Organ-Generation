"""
Tests for DesignSpec LLM Directive parsing and validation.

Tests cover:
1. Valid JSON output parsing
2. Invalid JSON triggers appropriate errors
3. Invalid patch structure triggers errors
4. Directive field validation
"""

import json
import pytest

from automation.designspec_llm.directive import (
    DesignSpecDirective,
    Question,
    RunRequest,
    ContextRequest,
    DirectiveParseError,
    from_json,
    extract_json_from_text,
    validate_directive,
    validate_patches,
    validate_json_patch,
    PIPELINE_STAGES,
    VALID_PATCH_OPS,
)


class TestDirectiveDataclasses:
    """Tests for directive dataclass creation and serialization."""
    
    def test_question_creation(self):
        q = Question(
            id="test_q",
            question="What size?",
            why_needed="Need to determine dimensions",
            default=10,
        )
        assert q.id == "test_q"
        assert q.question == "What size?"
        assert q.default == 10
        
        d = q.to_dict()
        assert d["id"] == "test_q"
        assert d["question"] == "What size?"
        
    def test_question_from_dict(self):
        d = {
            "id": "q1",
            "question": "How many?",
            "why_needed": "Need count",
            "default": 5,
        }
        q = Question.from_dict(d)
        assert q.id == "q1"
        assert q.question == "How many?"
        assert q.default == 5
    
    def test_run_request_creation(self):
        rr = RunRequest(
            run=True,
            run_until="component_mesh",
            reason="Test mesh generation",
            expected_signal="Mesh should have > 1000 faces",
        )
        assert rr.run is True
        assert rr.run_until == "component_mesh"
        
        d = rr.to_dict()
        assert d["run"] is True
        assert d["run_until"] == "component_mesh"
    
    def test_context_request_has_requests(self):
        cr = ContextRequest()
        assert cr.has_requests() is False
        
        cr.need_full_spec = True
        assert cr.has_requests() is True
        
        cr2 = ContextRequest(need_specific_files=["mesh.stl"])
        assert cr2.has_requests() is True
    
    def test_directive_creation(self):
        directive = DesignSpecDirective(
            assistant_message="I'll help you create a domain.",
            confidence=0.9,
            requires_approval=True,
        )
        assert directive.assistant_message == "I'll help you create a domain."
        assert directive.confidence == 0.9
        assert directive.has_patches() is False
        assert directive.has_questions() is False
        assert directive.has_run_request() is False
    
    def test_directive_with_patches(self):
        patches = [
            {"op": "add", "path": "/domains/main", "value": {"type": "box"}},
        ]
        directive = DesignSpecDirective(
            assistant_message="Adding domain",
            proposed_patches=patches,
        )
        assert directive.has_patches() is True
        assert len(directive.proposed_patches) == 1
    
    def test_directive_with_questions(self):
        questions = [
            Question(id="q1", question="What size?"),
        ]
        directive = DesignSpecDirective(
            assistant_message="I need more info",
            questions=questions,
        )
        assert directive.has_questions() is True
    
    def test_directive_with_run_request(self):
        directive = DesignSpecDirective(
            assistant_message="Let's run the pipeline",
            run_request=RunRequest(run=True, run_until="validity"),
        )
        assert directive.has_run_request() is True
    
    def test_directive_to_dict_and_back(self):
        original = DesignSpecDirective(
            assistant_message="Test message",
            questions=[Question(id="q1", question="Test?")],
            proposed_patches=[{"op": "add", "path": "/test", "value": 1}],
            run_request=RunRequest(run=True, run_until="full"),
            context_requests=ContextRequest(need_full_spec=True),
            confidence=0.85,
            requires_approval=True,
            stop=False,
        )
        
        d = original.to_dict()
        restored = DesignSpecDirective.from_dict(d)
        
        assert restored.assistant_message == original.assistant_message
        assert len(restored.questions) == len(original.questions)
        assert len(restored.proposed_patches) == len(original.proposed_patches)
        assert restored.confidence == original.confidence


class TestJsonExtraction:
    """Tests for extracting JSON from LLM output text."""
    
    def test_pure_json(self):
        text = '{"assistant_message": "Hello"}'
        result = extract_json_from_text(text)
        assert json.loads(result)["assistant_message"] == "Hello"
    
    def test_json_with_whitespace(self):
        text = '  \n  {"assistant_message": "Hello"}  \n  '
        result = extract_json_from_text(text)
        assert json.loads(result)["assistant_message"] == "Hello"
    
    def test_json_in_markdown_code_block(self):
        text = '''Here is my response:
```json
{"assistant_message": "Hello from markdown"}
```
'''
        result = extract_json_from_text(text)
        assert json.loads(result)["assistant_message"] == "Hello from markdown"
    
    def test_json_in_plain_code_block(self):
        text = '''```
{"assistant_message": "Plain block"}
```'''
        result = extract_json_from_text(text)
        assert json.loads(result)["assistant_message"] == "Plain block"
    
    def test_json_with_leading_text(self):
        text = 'Here is my analysis: {"assistant_message": "Analysis result"}'
        result = extract_json_from_text(text)
        assert json.loads(result)["assistant_message"] == "Analysis result"
    
    def test_json_with_trailing_text(self):
        text = '{"assistant_message": "Result"} Let me know if you need more.'
        result = extract_json_from_text(text)
        assert json.loads(result)["assistant_message"] == "Result"
    
    def test_nested_json(self):
        text = '{"assistant_message": "Test", "run_request": {"run": true, "run_until": "full"}}'
        result = extract_json_from_text(text)
        parsed = json.loads(result)
        assert parsed["run_request"]["run"] is True
    
    def test_no_json_raises_error(self):
        text = "This is just plain text with no JSON."
        with pytest.raises(DirectiveParseError) as exc_info:
            extract_json_from_text(text)
        assert "Could not extract valid JSON" in str(exc_info.value)
    
    def test_invalid_json_raises_error(self):
        text = '{"assistant_message": "Missing closing brace"'
        with pytest.raises(DirectiveParseError):
            extract_json_from_text(text)


class TestPatchValidation:
    """Tests for JSON Patch validation."""
    
    def test_valid_add_patch(self):
        patch = {"op": "add", "path": "/domains/main", "value": {"type": "box"}}
        errors = validate_json_patch(patch)
        assert len(errors) == 0
    
    def test_valid_replace_patch(self):
        patch = {"op": "replace", "path": "/meta/seed", "value": 42}
        errors = validate_json_patch(patch)
        assert len(errors) == 0
    
    def test_valid_remove_patch(self):
        patch = {"op": "remove", "path": "/components/0"}
        errors = validate_json_patch(patch)
        assert len(errors) == 0
    
    def test_valid_move_patch(self):
        patch = {"op": "move", "from": "/old/path", "path": "/new/path"}
        errors = validate_json_patch(patch)
        assert len(errors) == 0
    
    def test_valid_copy_patch(self):
        patch = {"op": "copy", "from": "/source", "path": "/dest"}
        errors = validate_json_patch(patch)
        assert len(errors) == 0
    
    def test_valid_test_patch(self):
        patch = {"op": "test", "path": "/meta/seed", "value": 42}
        errors = validate_json_patch(patch)
        assert len(errors) == 0
    
    def test_missing_op(self):
        patch = {"path": "/test", "value": 1}
        errors = validate_json_patch(patch)
        assert any("missing 'op'" in e for e in errors)
    
    def test_invalid_op(self):
        patch = {"op": "invalid_op", "path": "/test"}
        errors = validate_json_patch(patch)
        assert any("Invalid patch operation" in e for e in errors)
    
    def test_missing_path(self):
        patch = {"op": "add", "value": 1}
        errors = validate_json_patch(patch)
        assert any("missing 'path'" in e for e in errors)
    
    def test_path_not_starting_with_slash(self):
        patch = {"op": "add", "path": "no_slash", "value": 1}
        errors = validate_json_patch(patch)
        assert any("must start with '/'" in e for e in errors)
    
    def test_add_missing_value(self):
        patch = {"op": "add", "path": "/test"}
        errors = validate_json_patch(patch)
        assert any("requires 'value'" in e for e in errors)
    
    def test_move_missing_from(self):
        patch = {"op": "move", "path": "/dest"}
        errors = validate_json_patch(patch)
        assert any("requires 'from'" in e for e in errors)
    
    def test_validate_patches_list(self):
        patches = [
            {"op": "add", "path": "/test", "value": 1},
            {"op": "invalid"},
        ]
        errors = validate_patches(patches)
        assert len(errors) > 0
        assert any("Patch 1" in e for e in errors)


class TestDirectiveValidation:
    """Tests for directive structure validation."""
    
    def test_valid_minimal_directive(self):
        d = {"assistant_message": "Hello"}
        errors = validate_directive(d)
        assert len(errors) == 0
    
    def test_missing_assistant_message(self):
        d = {"confidence": 0.5}
        errors = validate_directive(d)
        assert any("assistant_message" in e for e in errors)
    
    def test_invalid_assistant_message_type(self):
        d = {"assistant_message": 123}
        errors = validate_directive(d)
        assert any("must be a string" in e for e in errors)
    
    def test_invalid_questions_type(self):
        d = {"assistant_message": "Test", "questions": "not a list"}
        errors = validate_directive(d)
        assert any("must be a list" in e for e in errors)
    
    def test_invalid_question_item(self):
        d = {"assistant_message": "Test", "questions": ["not a dict"]}
        errors = validate_directive(d)
        assert any("must be a dictionary" in e for e in errors)
    
    def test_question_missing_question_field(self):
        d = {"assistant_message": "Test", "questions": [{"id": "q1"}]}
        errors = validate_directive(d)
        assert any("missing 'question'" in e for e in errors)
    
    def test_invalid_run_until_stage(self):
        d = {
            "assistant_message": "Test",
            "run_request": {"run": True, "run_until": "invalid_stage"},
        }
        errors = validate_directive(d)
        assert any("Invalid run_until stage" in e for e in errors)
    
    def test_valid_run_until_stage(self):
        d = {
            "assistant_message": "Test",
            "run_request": {"run": True, "run_until": "component_mesh"},
        }
        errors = validate_directive(d)
        assert len(errors) == 0
    
    def test_invalid_confidence_range(self):
        d = {"assistant_message": "Test", "confidence": 1.5}
        errors = validate_directive(d)
        assert any("between 0 and 1" in e for e in errors)
    
    def test_invalid_confidence_type(self):
        d = {"assistant_message": "Test", "confidence": "high"}
        errors = validate_directive(d)
        assert any("must be a number" in e for e in errors)
    
    def test_invalid_requires_approval_type(self):
        d = {"assistant_message": "Test", "requires_approval": "yes"}
        errors = validate_directive(d)
        assert any("must be a boolean" in e for e in errors)


class TestFromJson:
    """Tests for the main from_json parsing function."""
    
    def test_parse_valid_directive(self):
        text = json.dumps({
            "assistant_message": "I'll create a box domain for you.",
            "proposed_patches": [
                {"op": "add", "path": "/domains/main", "value": {"type": "box"}}
            ],
            "confidence": 0.9,
            "requires_approval": True,
            "stop": False,
        })
        
        directive = from_json(text)
        
        assert directive.assistant_message == "I'll create a box domain for you."
        assert len(directive.proposed_patches) == 1
        assert directive.confidence == 0.9
        assert directive.requires_approval is True
    
    def test_parse_directive_with_questions(self):
        text = json.dumps({
            "assistant_message": "I need more information.",
            "questions": [
                {
                    "id": "domain_size",
                    "question": "What size should the domain be?",
                    "why_needed": "Need dimensions for box domain",
                    "default": "20mm x 60mm x 30mm",
                }
            ],
            "confidence": 0.7,
            "requires_approval": False,
        })
        
        directive = from_json(text)
        
        assert directive.has_questions()
        assert len(directive.questions) == 1
        assert directive.questions[0].id == "domain_size"
    
    def test_parse_directive_with_run_request(self):
        text = json.dumps({
            "assistant_message": "Let's run the pipeline to test.",
            "run_request": {
                "run": True,
                "run_until": "validity",
                "reason": "Verify mesh quality",
                "expected_signal": "All validity checks should pass",
            },
            "confidence": 0.85,
            "requires_approval": True,
        })
        
        directive = from_json(text)
        
        assert directive.has_run_request()
        assert directive.run_request.run_until == "validity"
    
    def test_parse_directive_from_markdown(self):
        text = '''Based on your request, here's my response:

```json
{
    "assistant_message": "Creating domain as requested.",
    "confidence": 0.9,
    "requires_approval": false,
    "stop": false
}
```

Let me know if you need anything else.'''
        
        directive = from_json(text)
        assert directive.assistant_message == "Creating domain as requested."
    
    def test_parse_invalid_json_raises_error(self):
        text = "This is not JSON at all."
        with pytest.raises(DirectiveParseError) as exc_info:
            from_json(text)
        assert "Could not extract valid JSON" in str(exc_info.value)
    
    def test_parse_invalid_directive_raises_error(self):
        text = json.dumps({"confidence": 0.5})  # Missing assistant_message
        with pytest.raises(DirectiveParseError) as exc_info:
            from_json(text)
        assert "assistant_message" in str(exc_info.value)
    
    def test_parse_invalid_patch_raises_error(self):
        text = json.dumps({
            "assistant_message": "Test",
            "proposed_patches": [{"op": "invalid"}],
        })
        with pytest.raises(DirectiveParseError):
            from_json(text)


class TestPipelineStages:
    """Tests for pipeline stage constants."""
    
    def test_known_stages_exist(self):
        expected_stages = [
            "compile_policies",
            "compile_domains",
            "component_build",
            "component_mesh",
            "validity",
            "full",
        ]
        for stage in expected_stages:
            assert stage in PIPELINE_STAGES
    
    def test_valid_patch_ops(self):
        expected_ops = {"add", "remove", "replace", "move", "copy", "test"}
        assert VALID_PATCH_OPS == expected_ops
