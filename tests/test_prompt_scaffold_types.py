"""
Tests for scaffold type coverage in LLM agent prompts.

Verifies that the DesignSpec LLM agent system prompt and the backend
ScaffoldAgent prompt both document the manifold_generator build type
and all 41 generator types from the ManifoldBackend registry.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from automation.designspec_llm.prompt_builder import SYSTEM_PROMPT  # noqa: E402
from generation.backends.manifold_backend import (  # noqa: E402
    GENERATOR_CATEGORIES,
    GENERATOR_REGISTRY,
)

_backend_prompts_path = REPO_ROOT / "backend" / "app" / "llm" / "prompts.py"
_spec = importlib.util.spec_from_file_location("_backend_prompts", _backend_prompts_path)
_backend_prompts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_backend_prompts)
BACKEND_PROMPT: str = _backend_prompts.SYSTEM_PROMPT
SYSTEM_PROMPT_COMPACT: str = _backend_prompts.SYSTEM_PROMPT_COMPACT


class TestDesignSpecPromptManifoldGenerator:
    """Verify the DesignSpec LLM agent prompt includes manifold_generator."""

    def test_manifold_generator_build_type_mentioned(self):
        assert "manifold_generator" in SYSTEM_PROMPT

    def test_manifold_generator_listed_as_valid_build_type(self):
        assert '"manifold_generator"' in SYSTEM_PROMPT

    def test_generator_type_field_documented(self):
        assert "generator_type" in SYSTEM_PROMPT

    def test_generator_params_field_documented(self):
        assert "generator_params" in SYSTEM_PROMPT

    def test_example_manifold_generator_component(self):
        assert '"type": "manifold_generator"' in SYSTEM_PROMPT

    def test_all_generator_categories_mentioned(self):
        prompt_lower = SYSTEM_PROMPT.lower()
        for category in GENERATOR_CATEGORIES:
            if category == "original":
                continue
            lookup = category.replace("_", " ")
            assert lookup in prompt_lower, (
                f"Category '{category}' (as '{lookup}') not found in DesignSpec SYSTEM_PROMPT"
            )

    def test_all_registry_types_mentioned_in_prompt(self):
        missing = []
        for gen_type in GENERATOR_REGISTRY:
            if gen_type not in SYSTEM_PROMPT:
                missing.append(gen_type)
        assert not missing, (
            f"Generator types missing from DesignSpec SYSTEM_PROMPT: {missing}"
        )

    def test_guiding_questions_include_manifold(self):
        assert "Manifold generator" in SYSTEM_PROMPT


class TestBackendPromptScaffoldTypes:
    """Verify the backend ScaffoldAgent prompt includes all scaffold types."""

    def test_all_registry_types_mentioned(self):
        missing = []
        for gen_type in GENERATOR_REGISTRY:
            if gen_type not in BACKEND_PROMPT:
                missing.append(gen_type)
        assert not missing, (
            f"Generator types missing from backend SYSTEM_PROMPT: {missing}"
        )

    def test_scaffold_count_mentioned(self):
        assert "41" in BACKEND_PROMPT

    def test_categories_mentioned(self):
        expected_categories = [
            "Lattice",
            "Skeletal",
            "Organ",
            "Soft Tissue",
            "Tubular",
            "Dental",
            "Microfluidic",
        ]
        for cat in expected_categories:
            assert cat in BACKEND_PROMPT, (
                f"Category '{cat}' not found in backend SYSTEM_PROMPT"
            )

    def test_compact_prompt_has_all_types(self):
        missing = []
        for gen_type in GENERATOR_REGISTRY:
            if gen_type not in SYSTEM_PROMPT_COMPACT:
                missing.append(gen_type)
        assert not missing, (
            f"Generator types missing from compact prompt: {missing}"
        )

    def test_application_routing_updated(self):
        assert "trabecular_bone" in BACKEND_PROMPT
        assert "hepatic_lobule" in BACKEND_PROMPT
        assert "cardiac_patch" in BACKEND_PROMPT
        assert "organ_on_chip" in BACKEND_PROMPT
