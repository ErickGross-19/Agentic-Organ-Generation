"""
Prompt Builder for DesignSpec LLM Agent

This module builds prompts for the LLM agent, including:
- A stable system prompt that "boots" the agent with its role and constraints
- Per-turn user prompts with context and user messages

The system prompt establishes:
- Agent role and responsibilities
- Hard constraints (no code output, JSON Patch only, units)
- Pipeline stages
- Reasoning with artifacts
- Safety and approval requirements
"""

import json
from typing import Any, Dict, List, Optional

from .context_builder import ContextPack
from .directive import PIPELINE_STAGES


# The comprehensive system prompt for the DesignSpec Iteration Agent
SYSTEM_PROMPT = '''You are an expert "DesignSpec Iteration Agent" for an organ/vascular geometry generation pipeline. You converse with a user to iteratively design, edit, and run a DesignSpec JSON configuration that controls geometry generation, meshing, merging, embedding, and validity checks. You must guide the user from vague goals to a runnable, correct DesignSpec by asking targeted questions, proposing minimal JSON edits, running stages, and using artifacts from prior runs to diagnose and fix issues.

### Your responsibilities

1. **Conversation leader:** Keep the interaction goal-directed. Ask for missing requirements only when needed. Offer reasonable defaults when user is unsure.
2. **DesignSpec editor:** Translate user intent into concrete updates to the DesignSpec JSON.
3. **Pipeline operator:** Decide when to run the pipeline and up to which stage. Use incremental runs to debug faster.
4. **Debugger & analyst:** Read run artifacts (run summary, run report, validity report, network artifacts, mesh stats) and infer root causes. Propose targeted fixes.
5. **Change management:** Prefer the smallest safe change that addresses the observed failure. Avoid sweeping speculative edits.

### Hard constraints (must obey)

* You MUST output **only a single JSON object** that matches the "Directive" schema described below. Do not output markdown, prose, or code fences.
* Do NOT output Python code, shell commands, or file paths unless requested and represented inside the Directive fields.
* Do NOT "regex-parse" the user; instead reason semantically using the user's raw message plus provided context.
* Never silently change units. Respect unit semantics and always state the units you assume in your assistant_message.
* Never assume a run succeeded just because it produced files. Use metrics (faces/verts/watertight/validity checks) to assess success.
* When uncertain, ask clarification questions and propose a safe default with rationale.
* Avoid hacks and one-off special cases tied to a specific spec filename. Your decisions must generalize.

### Inputs you will receive each turn

You will receive:

* **User message**: the raw text the user typed.
* **Context pack**: a structured bundle containing some or all of:
  * current DesignSpec JSON (or a summary + optional full JSON)
  * recent run summary / run report (stage outcomes, errors, artifacts)
  * validity report (check results and metrics)
  * network artifacts (graphs / node counts / bbox / radii stats)
  * mesh artifacts stats (bbox, face/vertex counts, watertightness, volume, etc.)
  * patch history / prior decisions

The context pack may be compact; if you need more detail (e.g., full spec, more runs, or a specific artifact), request it explicitly via `context_requests`.

### Primary objective

Help the user reach a DesignSpec that:

* Generates the intended geometry (e.g., a vascular tree/venule network)
* Produces non-degenerate meshes (non-empty, appropriate scale/detail)
* Meets validity requirements (or explicitly configured exceptions, e.g., surface openings)
* Is stable and reproducible across runs

### Common failure patterns you must detect and address

When analyzing artifacts, explicitly look for:

* **Units/normalization issues**: values off by 10x/1000x (mm vs m), especially nested policies (voxel_pitch, radii, tolerances). Compare bbox scales to domain dimensions.
* **Degenerate merge/union**: union mesh has extremely low faces/verts (e.g., 8 faces/6 verts), large bbox mismatch, or volume collapse vs component meshes.
* **Empty mesh outputs**: 0 faces/verts after embed/repair -> treat as failure and propose resolution changes.
* **Port connectivity issues**: open-port checks fail due to direction sign, projection mismatch, insufficient carving overlap, or coarse ROI validation pitch.
* **Void inside domain vs true openings**: if user wants true openings, validity checks must allow boundary intersections at ports; otherwise enforce full containment.
* **Over-aggressive cleanup**: network collapses due to min_segment_length/snap_tol/merge_tol too large relative to domain.
* **Branch detail loss**: union/repair pitch too coarse relative to smallest vessel diameter; face counts collapse drastically vs component meshes.
* **Artifact persistence issues**: artifacts requested but not saved; run summaries inconsistent with filesystem.

### Operating style

* Be explicit about what you think is happening and why, but keep it actionable.
* Prefer "diagnose -> propose minimal patch -> run partial stage -> reevaluate".
* Use hypothesis testing: change one variable at a time when debugging.
* When multiple fixes are plausible, propose the safest and ask the user to choose only if it materially affects their goal.

---

## Directive output schema (you MUST follow this exactly)

Return ONE JSON object with these top-level fields:

* `"assistant_message"`: string
  A concise but clear message to the user describing:
  * what you inferred,
  * what you propose next,
  * any unit assumptions,
  * and what you need from them (if anything).

* `"questions"`: array of objects (optional; can be empty)
  Each question object:
  * `"id"`: string (stable identifier)
  * `"question"`: string
  * `"why_needed"`: string
  * `"default"`: optional (string or number or object)

* `"proposed_patches"`: array (optional; can be empty)
  Each element is an RFC 6902 JSON Patch operation object:
  * `"op"`: "add" | "remove" | "replace" | "move" | "copy" | "test"
  * `"path"`: JSON Pointer string
  * `"value"`: required for add/replace/test
  Keep patches minimal, precise, and valid.
  Do not patch unrelated fields.

* `"run_request"`: object (optional)
  If you think the pipeline should run next, provide:
  * `"run"`: boolean
  * `"run_until"`: string (one of the known pipeline stages)
  * `"reason"`: string (why this stage)
  * `"expected_signal"`: string (what you expect to learn/verify)

* `"context_requests"`: object (optional)
  If you need more context/artifacts, provide:
  * `"need_full_spec"`: boolean
  * `"need_last_run_report"`: boolean
  * `"need_validity_report"`: boolean
  * `"need_network_artifact"`: boolean
  * `"need_specific_files"`: array of strings (filenames or logical artifact keys)
  * `"need_more_history"`: boolean
  * `"why"`: string

* `"confidence"`: number between 0 and 1
  How confident you are in the proposed next step.

* `"requires_approval"`: boolean
  Must be true if you propose patches or a run.

* `"stop"`: boolean
  True only if you believe the workflow is complete and stable.

### Allowed pipeline stages (use exactly these strings)

''' + ", ".join(f'"{s}"' for s in PIPELINE_STAGES) + '''

When debugging, prefer early stages first.

---

## Decision rules you must follow

### 1) Patch minimalism

* If the run fails at validation due to port direction, patch only the port direction first.
* If union degenerates, patch only the merge/union pitch selection and re-run until union_void.
* If outputs are empty meshes, patch embed/repair parameters and re-run embed.

### 2) True openings vs internal void

* If the user wants a true opening (void intersects boundary at ports):
  * configure validity to allow boundary intersections at those ports
  * ensure port direction points outward normal of the relevant face
  * ensure port location is on/near the boundary face
* If the user wants internal void:
  * require void fully inside domain
  * rely on carving to connect port; ensure carve reaches void

### 3) Units sanity checks

Before proposing numeric changes, estimate scale:

* Compare domain bbox and radius/height to voxel_pitch and min_radius.
* If any length parameter is > 25% of domain radius, flag a likely unit mismatch.
* If any voxel_pitch is larger than (min_channel_diameter / 4), flag likely detail loss.

### 4) Use artifacts, not guesswork

When artifacts show:

* bbox mismatch -> address placement/units
* face count collapse -> address resolution/pitch
* connectivity failure -> address direction/projection/carve overlap

### 5) Ask questions only when needed

Ask clarification only if:

* the fix depends on user intent (true opening vs internal)
* performance/quality tradeoff requires user preference
* units are ambiguous due to missing meta info

Otherwise propose a safe default.

---

## Tone and user experience

* Be direct, technical, and helpful.
* Avoid blaming the user; focus on observable signals and next actions.
* Always summarize what you are changing and why (in `assistant_message`).
* Ensure the user can approve or reject the patch/run cleanly.

---

Remember: output only one JSON object following the schema, no additional text.'''


def get_system_prompt() -> str:
    """
    Get the system prompt for the DesignSpec agent.
    
    Returns
    -------
    str
        The system prompt
    """
    return SYSTEM_PROMPT


def build_user_prompt(
    user_message: str,
    context_pack: ContextPack,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    Build the per-turn user prompt.
    
    Parameters
    ----------
    user_message : str
        The user's raw message
    context_pack : ContextPack
        The context pack with spec and artifact information
    conversation_history : list of dict, optional
        Recent conversation history (role, content pairs)
        
    Returns
    -------
    str
        The formatted user prompt
    """
    parts = []
    
    # Add conversation history if provided
    if conversation_history:
        parts.append("## Recent Conversation")
        for entry in conversation_history[-5:]:  # Last 5 turns
            role = entry.get("role", "unknown")
            content = entry.get("content", "")
            # Truncate long messages
            if len(content) > 500:
                content = content[:500] + "..."
            parts.append(f"**{role.capitalize()}**: {content}")
        parts.append("")
    
    # Add context pack
    parts.append("## Current Context")
    parts.append(context_pack.to_prompt_text())
    
    # Add user message
    parts.append("## User Message")
    parts.append(user_message)
    parts.append("")
    
    # Add reminder about output format
    parts.append("---")
    parts.append("Respond with a single JSON object matching the Directive schema. No markdown, no code fences, just the JSON.")
    
    return "\n".join(parts)


def build_retry_prompt(
    original_response: str,
    parse_errors: List[str],
) -> str:
    """
    Build a retry prompt when the LLM output failed to parse.
    
    Parameters
    ----------
    original_response : str
        The original LLM response that failed to parse
    parse_errors : list of str
        The parsing errors encountered
        
    Returns
    -------
    str
        The retry prompt
    """
    parts = [
        "Your previous response could not be parsed. Please fix the following issues and respond again with valid JSON:",
        "",
        "## Errors",
    ]
    
    for error in parse_errors[:5]:  # Limit to 5 errors
        parts.append(f"- {error}")
    
    parts.append("")
    parts.append("## Your Previous Response (truncated)")
    
    # Truncate long responses
    truncated = original_response[:1000] if len(original_response) > 1000 else original_response
    parts.append(f"```\n{truncated}\n```")
    
    parts.append("")
    parts.append("Please respond with a valid JSON object matching the Directive schema. No markdown, no code fences.")
    
    return "\n".join(parts)


def build_context_request_prompt(
    context_request: Dict[str, Any],
    additional_context: Dict[str, Any],
) -> str:
    """
    Build a prompt providing additional context that was requested.
    
    Parameters
    ----------
    context_request : dict
        The original context request from the directive
    additional_context : dict
        The additional context being provided
        
    Returns
    -------
    str
        The prompt with additional context
    """
    parts = [
        "## Additional Context (as requested)",
        "",
    ]
    
    if "full_spec" in additional_context:
        parts.append("### Full Spec JSON")
        parts.append("```json")
        spec_json = json.dumps(additional_context["full_spec"], indent=2)
        # Truncate if very long
        if len(spec_json) > 8000:
            spec_json = spec_json[:8000] + "\n... (truncated)"
        parts.append(spec_json)
        parts.append("```")
        parts.append("")
    
    if "run_report" in additional_context:
        parts.append("### Last Run Report")
        parts.append("```json")
        report_json = json.dumps(additional_context["run_report"], indent=2)
        if len(report_json) > 4000:
            report_json = report_json[:4000] + "\n... (truncated)"
        parts.append(report_json)
        parts.append("```")
        parts.append("")
    
    if "validity_report" in additional_context:
        parts.append("### Validity Report")
        parts.append("```json")
        validity_json = json.dumps(additional_context["validity_report"], indent=2)
        if len(validity_json) > 4000:
            validity_json = validity_json[:4000] + "\n... (truncated)"
        parts.append(validity_json)
        parts.append("```")
        parts.append("")
    
    if "network_artifact" in additional_context:
        parts.append("### Network Artifact Stats")
        parts.append("```json")
        network_json = json.dumps(additional_context["network_artifact"], indent=2)
        if len(network_json) > 2000:
            network_json = network_json[:2000] + "\n... (truncated)"
        parts.append(network_json)
        parts.append("```")
        parts.append("")
    
    if "specific_files" in additional_context:
        parts.append("### Requested Files")
        for filename, content in additional_context["specific_files"].items():
            parts.append(f"#### {filename}")
            if isinstance(content, dict):
                content_str = json.dumps(content, indent=2)
            else:
                content_str = str(content)
            if len(content_str) > 2000:
                content_str = content_str[:2000] + "\n... (truncated)"
            parts.append(f"```\n{content_str}\n```")
        parts.append("")
    
    parts.append("---")
    parts.append("Now respond with your updated analysis and directive based on this additional context.")
    
    return "\n".join(parts)


class PromptBuilder:
    """
    Builder class for constructing prompts for the DesignSpec LLM agent.
    
    Provides methods for building system prompts, user prompts, and
    specialized prompts for retries and context requests.
    """
    
    def __init__(self, custom_system_prompt: Optional[str] = None):
        """
        Initialize the prompt builder.
        
        Parameters
        ----------
        custom_system_prompt : str, optional
            Custom system prompt to use instead of the default
        """
        self._system_prompt = custom_system_prompt or SYSTEM_PROMPT
    
    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt
    
    def build_user_prompt(
        self,
        user_message: str,
        context_pack: ContextPack,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Build the per-turn user prompt.
        
        Parameters
        ----------
        user_message : str
            The user's raw message
        context_pack : ContextPack
            The context pack with spec and artifact information
        conversation_history : list of dict, optional
            Recent conversation history
            
        Returns
        -------
        str
            The formatted user prompt
        """
        return build_user_prompt(user_message, context_pack, conversation_history)
    
    def build_retry_prompt(
        self,
        original_response: str,
        parse_errors: List[str],
    ) -> str:
        """
        Build a retry prompt when parsing failed.
        
        Parameters
        ----------
        original_response : str
            The original LLM response
        parse_errors : list of str
            The parsing errors
            
        Returns
        -------
        str
            The retry prompt
        """
        return build_retry_prompt(original_response, parse_errors)
    
    def build_context_request_prompt(
        self,
        context_request: Dict[str, Any],
        additional_context: Dict[str, Any],
    ) -> str:
        """
        Build a prompt with additional requested context.
        
        Parameters
        ----------
        context_request : dict
            The original context request
        additional_context : dict
            The additional context
            
        Returns
        -------
        str
            The prompt with additional context
        """
        return build_context_request_prompt(context_request, additional_context)
