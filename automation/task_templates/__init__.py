"""
Task Templates

Pre-built task prompts for common organ generation and validation operations.
These templates provide structured prompts that guide the LLM to produce
consistent, high-quality outputs.
"""

from .generate_structure import (
    generate_structure_prompt,
    generate_liver_prompt,
    generate_custom_organ_prompt,
)
from .validate_structure import (
    validate_structure_prompt,
    validate_pre_embedding_prompt,
    validate_post_embedding_prompt,
)
from .iterate_design import (
    iterate_design_prompt,
    fix_validation_issues_prompt,
    optimize_structure_prompt,
)

__all__ = [
    # Generation prompts
    "generate_structure_prompt",
    "generate_liver_prompt",
    "generate_custom_organ_prompt",
    # Validation prompts
    "validate_structure_prompt",
    "validate_pre_embedding_prompt",
    "validate_post_embedding_prompt",
    # Iteration prompts
    "iterate_design_prompt",
    "fix_validation_issues_prompt",
    "optimize_structure_prompt",
]
