"""
Brain Module - LLM Decision-Making for V5

The brain is responsible for:
1. Building prompts from user messages, world model facts, workspace summary, verification reports
2. Calling the LLM to decide what to do next
3. Parsing the LLM response into a structured Directive

The Directive describes the next actions:
- assistant_message: what to show the user
- questions: optional list of questions to ask
- workspace_update: optional code changes (master script, tools, registry)
- request_execution: whether to run the master script
- stop: whether the agent believes it's done
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .world_model import WorldModel
    from .workspace import WorkspaceManager, WorkspaceSummary
    from ...llm_client import LLMClient

logger = logging.getLogger(__name__)


def _apply_unified_diff(original: str, patch: str) -> str:
    """
    Apply a unified diff patch to original content.
    
    Pure-Python implementation that doesn't require external `patch` binary.
    
    Parameters
    ----------
    original : str
        The original file content
    patch : str
        The unified diff patch content
        
    Returns
    -------
    str
        The patched content
        
    Raises
    ------
    ValueError
        If patch cannot be applied (context mismatch, invalid format, etc.)
    """
    import re
    
    original_lines = original.splitlines(keepends=True)
    if original and not original.endswith('\n'):
        if original_lines:
            original_lines[-1] += '\n'
    
    result_lines = list(original_lines)
    
    hunk_pattern = re.compile(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@')
    
    patch_lines = patch.splitlines(keepends=True)
    
    i = 0
    offset = 0
    
    while i < len(patch_lines):
        line = patch_lines[i]
        
        if line.startswith('---') or line.startswith('+++'):
            i += 1
            continue
        
        match = hunk_pattern.match(line)
        if match:
            old_start = int(match.group(1))
            
            i += 1
            
            hunk_removes = []
            hunk_adds = []
            context_before_count = 0
            in_changes = False
            
            while i < len(patch_lines):
                hunk_line = patch_lines[i]
                
                if hunk_line.startswith('@@') or hunk_line.startswith('---') or hunk_line.startswith('+++'):
                    break
                
                if hunk_line.startswith('-'):
                    in_changes = True
                    hunk_removes.append(hunk_line[1:])
                elif hunk_line.startswith('+'):
                    in_changes = True
                    hunk_adds.append(hunk_line[1:])
                elif hunk_line.startswith(' ') or hunk_line == '\n':
                    if not in_changes:
                        context_before_count += 1
                elif hunk_line.startswith('\\'):
                    i += 1
                    continue
                else:
                    break
                
                i += 1
            
            apply_at = old_start - 1 + context_before_count + offset
            
            if apply_at < 0:
                apply_at = 0
            if apply_at > len(result_lines):
                apply_at = len(result_lines)
            
            num_to_remove = len(hunk_removes)
            
            del result_lines[apply_at:apply_at + num_to_remove]
            
            for j, add_line in enumerate(hunk_adds):
                if not add_line.endswith('\n'):
                    add_line += '\n'
                result_lines.insert(apply_at + j, add_line)
            
            offset += len(hunk_adds) - num_to_remove
        else:
            i += 1
    
    result = ''.join(result_lines)
    if result.endswith('\n') and not original.endswith('\n'):
        result = result[:-1]
    
    return result


@dataclass
class Question:
    """A question to ask the user."""
    id: str
    prompt: str
    question_type: str = "text"  # "text", "choice", "confirm"
    options: Optional[List[str]] = None
    default: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "type": self.question_type,
            "options": self.options,
            "default": self.default,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Question":
        return cls(
            id=d["id"],
            prompt=d["prompt"],
            question_type=d.get("type", "text"),
            options=d.get("options"),
            default=d.get("default"),
        )


@dataclass
class FileUpdate:
    """
    A file to create or update.
    
    P1 #7: Supports both full content and unified diff patches.
    """
    path: str  # Relative path within workspace (e.g., "master.py", "tools/my_tool.py")
    content: str  # Full content OR unified diff patch
    patch_style: Optional[str] = None  # None for full content, "unified_diff" for patch
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
            "patch_style": self.patch_style,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileUpdate":
        return cls(
            path=d["path"],
            content=d["content"],
            patch_style=d.get("patch_style"),
        )
    
    def is_patch(self) -> bool:
        """Check if this is a patch update rather than full content."""
        return self.patch_style == "unified_diff"
    
    def apply_patch(self, original_content: str) -> str:
        """
        Apply unified diff patch to original content.
        
        Uses pure-Python implementation (no external `patch` binary required).
        
        Parameters
        ----------
        original_content : str
            The original file content
            
        Returns
        -------
        str
            The patched content
            
        Raises
        ------
        ValueError
            If patch cannot be applied
        """
        if not self.is_patch():
            return self.content
        
        return _apply_unified_diff(original_content, self.content)


@dataclass
class RegistryUpdate:
    """An update to the tool registry."""
    action: str  # "add" or "remove"
    name: str
    module: Optional[str] = None
    entrypoints: Optional[List[str]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "name": self.name,
            "module": self.module,
            "entrypoints": self.entrypoints,
            "description": self.description,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegistryUpdate":
        return cls(
            action=d["action"],
            name=d["name"],
            module=d.get("module"),
            entrypoints=d.get("entrypoints"),
            description=d.get("description"),
        )


@dataclass
class WorkspaceUpdate:
    """Updates to the workspace (code changes)."""
    edit_master: bool = False
    files: List[FileUpdate] = field(default_factory=list)
    registry_updates: List[RegistryUpdate] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "edit_master": self.edit_master,
            "files": [f.to_dict() for f in self.files],
            "registry_updates": [r.to_dict() for r in self.registry_updates],
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WorkspaceUpdate":
        return cls(
            edit_master=d.get("edit_master", False),
            files=[FileUpdate.from_dict(f) for f in d.get("files", [])],
            registry_updates=[RegistryUpdate.from_dict(r) for r in d.get("registry_updates", [])],
        )


@dataclass
class FactUpdate:
    """
    P2 #17: A structured fact update from the LLM.
    
    Allows the LLM to cleanly update structured facts (domain, inlet/outlet, targets).
    """
    op: str  # "set", "delete", "append"
    path: str  # Fact path (e.g., "domain.type", "inlet.position")
    value: Any = None
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "op": self.op,
            "path": self.path,
            "value": self.value,
            "confidence": self.confidence,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FactUpdate":
        return cls(
            op=d.get("op", "set"),
            path=d["path"],
            value=d.get("value"),
            confidence=d.get("confidence", 1.0),
        )


@dataclass
class PlanBoardUpdate:
    """
    P2 #18: Update to the plan board.
    """
    add_objectives: List[str] = field(default_factory=list)
    add_assumptions: List[str] = field(default_factory=list)
    set_strategy: Optional[str] = None
    add_next_steps: List[str] = field(default_factory=list)
    complete_steps: List[str] = field(default_factory=list)
    add_risks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "add_objectives": self.add_objectives,
            "add_assumptions": self.add_assumptions,
            "set_strategy": self.set_strategy,
            "add_next_steps": self.add_next_steps,
            "complete_steps": self.complete_steps,
            "add_risks": self.add_risks,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlanBoardUpdate":
        return cls(
            add_objectives=d.get("add_objectives", []),
            add_assumptions=d.get("add_assumptions", []),
            set_strategy=d.get("set_strategy"),
            add_next_steps=d.get("add_next_steps", []),
            complete_steps=d.get("complete_steps", []),
            add_risks=d.get("add_risks", []),
        )


@dataclass
class Directive:
    """
    The LLM's decision about what to do next.
    
    This is the structured output from the brain that the controller acts on.
    
    P2 #17: Includes fact_updates for structured fact changes.
    P2 #18: Includes plan_board_update for plan changes.
    P2 #20: Includes preview_mode for quick preview runs.
    """
    assistant_message: str = ""
    questions: List[Question] = field(default_factory=list)
    workspace_update: Optional[WorkspaceUpdate] = None
    request_execution: bool = False
    stop: bool = False
    reasoning: str = ""  # Internal reasoning (for debugging/tracing)
    # P2 #17: Structured fact updates
    fact_updates: List[FactUpdate] = field(default_factory=list)
    # P2 #18: Plan board updates
    plan_board_update: Optional[PlanBoardUpdate] = None
    # P2 #20: Preview mode flag
    preview_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assistant_message": self.assistant_message,
            "questions": [q.to_dict() for q in self.questions],
            "workspace_update": self.workspace_update.to_dict() if self.workspace_update else None,
            "request_execution": self.request_execution,
            "stop": self.stop,
            "reasoning": self.reasoning,
            "fact_updates": [f.to_dict() for f in self.fact_updates],
            "plan_board_update": self.plan_board_update.to_dict() if self.plan_board_update else None,
            "preview_mode": self.preview_mode,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Directive":
        workspace_update = None
        if d.get("workspace_update"):
            workspace_update = WorkspaceUpdate.from_dict(d["workspace_update"])
        
        plan_board_update = None
        if d.get("plan_board_update"):
            plan_board_update = PlanBoardUpdate.from_dict(d["plan_board_update"])
        
        return cls(
            assistant_message=d.get("assistant_message", ""),
            questions=[Question.from_dict(q) for q in d.get("questions", [])],
            workspace_update=workspace_update,
            request_execution=d.get("request_execution", False),
            stop=d.get("stop", False),
            reasoning=d.get("reasoning", ""),
            fact_updates=[FactUpdate.from_dict(f) for f in d.get("fact_updates", [])],
            plan_board_update=plan_board_update,
            preview_mode=d.get("preview_mode", False),
        )


@dataclass
class GoalSatisfaction:
    """
    P2 #22: Goal satisfaction signals for the LLM.
    
    Soft signals that help the LLM understand what's missing.
    """
    has_domain: bool = False
    has_inlet: bool = False
    has_outlet: bool = False
    has_topology: bool = False
    has_master_script: bool = False
    has_successful_run: bool = False
    has_verified_mesh: bool = False
    missing_requirements: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_domain": self.has_domain,
            "has_inlet": self.has_inlet,
            "has_outlet": self.has_outlet,
            "has_topology": self.has_topology,
            "has_master_script": self.has_master_script,
            "has_successful_run": self.has_successful_run,
            "has_verified_mesh": self.has_verified_mesh,
            "missing_requirements": self.missing_requirements,
        }


@dataclass
class ErrorPacket:
    """
    P2 #23: Structured error-to-fix packet for the LLM.
    
    When a run fails, provides structured information to help the LLM fix it.
    """
    error_type: str  # "syntax", "import", "runtime", "verification", "timeout"
    error_message: str
    stderr_tail: List[str] = field(default_factory=list)
    verification_errors: List[str] = field(default_factory=list)
    mesh_stats: Optional[Dict[str, Any]] = None
    suggested_fix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stderr_tail": self.stderr_tail,
            "verification_errors": self.verification_errors,
            "mesh_stats": self.mesh_stats,
            "suggested_fix": self.suggested_fix,
        }


@dataclass
class ArtifactRequirements:
    """
    P0 #2: Required artifact paths injected into LLM prompt every run.
    
    Tells the LLM exactly what files must be created and their naming rules.
    """
    version: int = 1
    required_files: List[str] = field(default_factory=list)
    optional_files: List[str] = field(default_factory=list)
    naming_rules: Dict[str, str] = field(default_factory=dict)
    output_dir_env: str = "ORGAN_AGENT_OUTPUT_DIR"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "required_files": self.required_files,
            "optional_files": self.optional_files,
            "naming_rules": self.naming_rules,
            "output_dir_env": self.output_dir_env,
        }
    
    @classmethod
    def for_generation(cls, version: int) -> "ArtifactRequirements":
        v = f"v{version:03d}"
        return cls(
            version=version,
            required_files=[
                f"04_outputs/network_{v}.json",
                f"05_mesh/mesh_network_{v}.stl",
            ],
            optional_files=[
                f"06_analysis/analysis_{v}.json",
                f"06_analysis/analysis_{v}.txt",
            ],
            naming_rules={
                "version_format": "v{version:03d}",
                "network_pattern": "04_outputs/network_v{version:03d}.json",
                "mesh_pattern": "05_mesh/mesh_network_v{version:03d}.stl",
            },
        )


@dataclass 
class PlanBoard:
    """
    P2 #18: Living plan structure that the LLM updates.
    
    Contains objectives, assumptions, strategy, steps, and risks.
    """
    objectives: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    chosen_strategy: str = ""
    next_steps: List[str] = field(default_factory=list)
    done_steps: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "objectives": self.objectives,
            "assumptions": self.assumptions,
            "chosen_strategy": self.chosen_strategy,
            "next_steps": self.next_steps,
            "done_steps": self.done_steps,
            "risks": self.risks,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlanBoard":
        return cls(
            objectives=d.get("objectives", []),
            assumptions=d.get("assumptions", []),
            chosen_strategy=d.get("chosen_strategy", ""),
            next_steps=d.get("next_steps", []),
            done_steps=d.get("done_steps", []),
            risks=d.get("risks", []),
        )


@dataclass
class ObservationPacket:
    """
    The observation packet sent to the LLM for decision-making.
    
    Contains all context the LLM needs to decide what to do next.
    """
    user_message: Optional[str] = None
    world_model_summary: Dict[str, Any] = field(default_factory=dict)
    workspace_summary: Dict[str, Any] = field(default_factory=dict)
    master_script_content: str = ""
    tool_registry_description: str = ""
    last_run_result: Optional[Dict[str, Any]] = None
    verification_report: Optional[Dict[str, Any]] = None
    goal_progress: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    # P2 #22: Goal satisfaction signals
    goal_satisfaction: Optional[GoalSatisfaction] = None
    # P2 #23: Error-to-fix packet
    error_packet: Optional[ErrorPacket] = None
    # P0 #2: Required artifact paths
    artifact_requirements: Optional[ArtifactRequirements] = None
    # P2 #18: Plan board
    plan_board: Optional[PlanBoard] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_message": self.user_message,
            "world_model_summary": self.world_model_summary,
            "workspace_summary": self.workspace_summary,
            "master_script_content": self.master_script_content,
            "tool_registry_description": self.tool_registry_description,
            "last_run_result": self.last_run_result,
            "verification_report": self.verification_report,
            "goal_progress": self.goal_progress,
            "conversation_history": self.conversation_history,
            "goal_satisfaction": self.goal_satisfaction.to_dict() if self.goal_satisfaction else None,
            "error_packet": self.error_packet.to_dict() if self.error_packet else None,
            "artifact_requirements": self.artifact_requirements.to_dict() if self.artifact_requirements else None,
            "plan_board": self.plan_board.to_dict() if self.plan_board else None,
        }


# System prompt for the brain
BRAIN_SYSTEM_PROMPT = """You are an AI agent that generates vascular network scaffolds for organ engineering.

Your job is to:
1. Understand what the user wants to create
2. Write and iterate on a master Python script that generates the scaffold
3. Use available tools from the tool registry
4. Request execution when ready
5. Verify results and fix issues

## Your Capabilities

You can:
- Ask questions to clarify requirements
- Write/edit the master.py script
- Create helper tool modules in tools/
- Request execution of the master script (requires user approval)
- Analyze verification results and fix issues

## Output Format

You MUST respond with a JSON object containing:
```json
{
  "reasoning": "Your internal reasoning about what to do next",
  "assistant_message": "Message to show the user",
  "questions": [
    {"id": "q1", "prompt": "Question text", "type": "text", "options": null, "default": null}
  ],
  "workspace_update": {
    "edit_master": true,
    "files": [
      {"path": "master.py", "content": "...full script content..."}
    ],
    "registry_updates": []
  },
  "request_execution": false,
  "stop": false
}
```

## Important Guidelines

1. **Prefer minimal edits**: If the master script exists and works, only make necessary changes
2. **Reuse existing tools**: Check the tool registry before creating new tools
3. **Only create tools when**: A component is reused, or master is getting complex
4. **Always include full file content**: When editing files, include the complete content
5. **Request execution when ready**: Set request_execution=true when the script is ready to run
6. **Stop when done**: Set stop=true when generation is complete and verified

## Master Script Contract

The master script must:
1. Compute WORKSPACE_DIR from __file__ (the script's location)
2. Load spec.json and tool_registry.json from WORKSPACE_DIR
3. Write outputs to OUTPUT_DIR (from environment variable ORGAN_AGENT_OUTPUT_DIR)
4. Create REQUIRED output files with EXACT paths (verifier checks these):
   - `04_outputs/network_v{VERSION:03d}.json` - Network data
   - `05_mesh/mesh_network_v{VERSION:03d}.stl` - STL mesh
5. Print ARTIFACTS_JSON footer with created files and metrics

CRITICAL: The verifier checks for EXACT file paths. Use the version provided in Required Artifacts (artifact_requirements.version in the observation packet).

Example structure:
```python
import os
import sys
import json

# CRITICAL: Compute workspace dir from script location, NOT cwd
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get('ORGAN_AGENT_OUTPUT_DIR', os.getcwd())

# Add workspace/tools to path for generated tools
sys.path.insert(0, os.path.join(WORKSPACE_DIR, 'tools'))

def load_spec():
    spec_path = os.path.join(WORKSPACE_DIR, 'spec.json')
    with open(spec_path) as f:
        return json.load(f)

def load_registry():
    registry_path = os.path.join(WORKSPACE_DIR, 'tool_registry.json')
    with open(registry_path) as f:
        return json.load(f)

def main():
    spec = load_spec()
    version = spec.get('run_version', 1)
    v_str = f'v{version:03d}'
    
    # Create required output directories
    os.makedirs(os.path.join(OUTPUT_DIR, '04_outputs'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, '05_mesh'), exist_ok=True)
    
    # ... generation logic using tools from generation.ops and generation.core ...
    
    # REQUIRED output files (verifier checks these exact paths!)
    network_path = os.path.join(OUTPUT_DIR, '04_outputs', f'network_{v_str}.json')
    mesh_path = os.path.join(OUTPUT_DIR, '05_mesh', f'mesh_network_{v_str}.stl')
    
    # ... save network to network_path ...
    # ... export mesh to mesh_path ...
    
    # Print ARTIFACTS_JSON footer (verifier parses this)
    artifacts = {
        "files": [network_path, mesh_path],
        "metrics": {"node_count": 0, "segment_count": 0},
        "status": "success"
    }
    print(f'ARTIFACTS_JSON: {json.dumps(artifacts)}')

if __name__ == "__main__":
    main()
```
"""


class Brain:
    """
    The LLM brain for V5 decision-making.
    
    Builds prompts, calls the LLM, and parses responses into Directives.
    """
    
    def __init__(
        self,
        llm_client: "LLMClient",
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the brain.
        
        Parameters
        ----------
        llm_client : LLMClient
            The LLM client to use for decisions
        system_prompt : str, optional
            Custom system prompt (defaults to BRAIN_SYSTEM_PROMPT)
        """
        self.llm_client = llm_client
        self.system_prompt = system_prompt or BRAIN_SYSTEM_PROMPT
        self._conversation_history: List[Dict[str, str]] = []
    
    def build_observation_packet(
        self,
        user_message: Optional[str],
        world_model: "WorldModel",
        workspace: "WorkspaceManager",
        goal_progress: Dict[str, Any],
        last_run_result: Optional[Dict[str, Any]] = None,
        verification_report: Optional[Dict[str, Any]] = None,
        goal_satisfaction: Optional[GoalSatisfaction] = None,
        error_packet: Optional[ErrorPacket] = None,
        plan_board: Optional[PlanBoard] = None,
    ) -> ObservationPacket:
        """
        Build an observation packet for the LLM.
        
        Parameters
        ----------
        user_message : str, optional
            The latest user message
        world_model : WorldModel
            The current world model
        workspace : WorkspaceManager
            The workspace manager
        goal_progress : dict
            Goal progress summary from GoalTracker
        last_run_result : dict, optional
            Result from the last script run
        verification_report : dict, optional
            Verification report from artifact verifier
        goal_satisfaction : GoalSatisfaction, optional
            P2 #22: Goal satisfaction signals
        error_packet : ErrorPacket, optional
            P2 #23: Error-to-fix packet
        plan_board : PlanBoard, optional
            P2 #18: Living plan structure
            
        Returns
        -------
        ObservationPacket
            The observation packet for the LLM
        """
        # P0 #2: Get artifact requirements for next version
        # Use peek_next_run_version() to get the version the LLM should target
        # without incrementing the counter (that happens when run actually starts)
        run_version = workspace.peek_next_run_version()
        artifact_requirements = ArtifactRequirements.for_generation(run_version)
        
        return ObservationPacket(
            user_message=user_message,
            world_model_summary=world_model.get_living_spec_summary(),
            workspace_summary=workspace.get_summary().to_dict(),
            master_script_content=workspace.get_master_script_for_prompt(),
            tool_registry_description=workspace.get_tool_registry_for_prompt(),
            last_run_result=last_run_result,
            verification_report=verification_report,
            goal_progress=goal_progress,
            conversation_history=list(self._conversation_history),
            goal_satisfaction=goal_satisfaction,
            error_packet=error_packet,
            artifact_requirements=artifact_requirements,
            plan_board=plan_board,
        )
    
    def build_prompt(self, observation: ObservationPacket) -> str:
        """
        Build the prompt for the LLM from an observation packet.
        
        Parameters
        ----------
        observation : ObservationPacket
            The observation packet
            
        Returns
        -------
        str
            The formatted prompt
        """
        sections = []
        
        # User message
        if observation.user_message:
            sections.append(f"## User Message\n{observation.user_message}")
        
        # World model summary (living spec)
        if observation.world_model_summary:
            facts = observation.world_model_summary.get("facts", {})
            if facts:
                facts_str = json.dumps(facts, indent=2, default=str)
                sections.append(f"## Current Specification\n```json\n{facts_str}\n```")
        
        # P0 #2: Required artifact paths - inject every run
        if observation.artifact_requirements:
            ar = observation.artifact_requirements
            ar_lines = [
                "## Required Artifacts (CRITICAL)",
                f"Version: {ar.version}",
                "",
                "**REQUIRED files (verifier checks these exact paths):**",
            ]
            for f in ar.required_files:
                ar_lines.append(f"- `$OUTPUT_DIR/{f}`")
            ar_lines.append("")
            ar_lines.append("**Optional files:**")
            for f in ar.optional_files:
                ar_lines.append(f"- `$OUTPUT_DIR/{f}`")
            ar_lines.append("")
            ar_lines.append(f"**Output directory env var:** `{ar.output_dir_env}`")
            ar_lines.append("")
            ar_lines.append("**Naming rules:**")
            for key, pattern in ar.naming_rules.items():
                ar_lines.append(f"- {key}: `{pattern}`")
            sections.append('\n'.join(ar_lines))
        
        # Workspace summary
        ws = observation.workspace_summary
        if ws:
            ws_lines = [
                "## Workspace Status",
                f"- Master script exists: {ws.get('master_script_exists', False)}",
                f"- Master script lines: {ws.get('master_script_lines', 0)}",
                f"- Generated tools: {ws.get('tool_count', 0)}",
                f"- Run count: {ws.get('run_count', 0)}",
            ]
            if ws.get('last_run_status'):
                ws_lines.append(f"- Last run status: {ws.get('last_run_status')}")
            # P1 #9: Include file hashes and modified times
            # WorkspaceSummary.to_dict() outputs 'tool_files' with keys 'path', 'hash', 'modified_time'
            if ws.get('tool_files'):
                ws_lines.append("")
                ws_lines.append("**File details:**")
                for file_info in ws.get('tool_files', []):
                    path = file_info.get('path', 'unknown')
                    name = path.split('/')[-1] if '/' in path else path
                    hash_val = file_info.get('hash', 'N/A')[:8] if file_info.get('hash') else 'N/A'
                    modified = file_info.get('modified_time', 'N/A')
                    ws_lines.append(f"- {name}: hash={hash_val}... modified={modified}")
            sections.append('\n'.join(ws_lines))
        
        # Master script content
        if observation.master_script_content and observation.master_script_content != "[No master script exists yet]":
            sections.append(f"## Current Master Script\n```python\n{observation.master_script_content}\n```")
        
        # Tool registry
        if observation.tool_registry_description:
            sections.append(f"## Tool Registry\n{observation.tool_registry_description}")
        
        # P2 #18: Plan board
        if observation.plan_board:
            pb = observation.plan_board
            pb_lines = ["## Plan Board"]
            if pb.objectives:
                pb_lines.append("**Objectives:**")
                for obj in pb.objectives:
                    pb_lines.append(f"- {obj}")
            if pb.assumptions:
                pb_lines.append("**Assumptions:**")
                for a in pb.assumptions:
                    pb_lines.append(f"- {a}")
            if pb.chosen_strategy:
                pb_lines.append(f"**Strategy:** {pb.chosen_strategy}")
            if pb.done_steps:
                pb_lines.append("**Done:**")
                for s in pb.done_steps:
                    pb_lines.append(f"- [x] {s}")
            if pb.next_steps:
                pb_lines.append("**Next:**")
                for s in pb.next_steps:
                    pb_lines.append(f"- [ ] {s}")
            if pb.risks:
                pb_lines.append("**Risks:**")
                for r in pb.risks:
                    pb_lines.append(f"- {r}")
            sections.append('\n'.join(pb_lines))
        
        # P2 #22: Goal satisfaction signals
        if observation.goal_satisfaction:
            gs = observation.goal_satisfaction
            gs_lines = [
                "## Goal Satisfaction Signals",
                f"- has_domain: {gs.has_domain}",
                f"- has_inlet: {gs.has_inlet}",
                f"- has_outlet: {gs.has_outlet}",
                f"- has_topology: {gs.has_topology}",
                f"- has_master_script: {gs.has_master_script}",
                f"- has_successful_run: {gs.has_successful_run}",
                f"- has_verified_mesh: {gs.has_verified_mesh}",
            ]
            if gs.missing_requirements:
                gs_lines.append("**Missing requirements:**")
                for req in gs.missing_requirements:
                    gs_lines.append(f"- {req}")
            sections.append('\n'.join(gs_lines))
        
        # Last run result
        if observation.last_run_result:
            run_str = json.dumps(observation.last_run_result, indent=2, default=str)
            sections.append(f"## Last Run Result\n```json\n{run_str}\n```")
        
        # P2 #23: Error-to-fix packet
        if observation.error_packet:
            ep = observation.error_packet
            ep_lines = [
                "## Error Analysis (FIX THIS)",
                f"**Exception:** {ep.exception_type}: {ep.exception_message}",
            ]
            if ep.stderr_tail:
                ep_lines.append(f"**Stderr tail:**\n```\n{ep.stderr_tail}\n```")
            if ep.verifier_summary:
                ep_lines.append(f"**Verifier summary:** {ep.verifier_summary}")
            if ep.mesh_stats:
                ep_lines.append(f"**Mesh stats:** {json.dumps(ep.mesh_stats)}")
            if ep.suggested_fix:
                ep_lines.append(f"**Suggested fix:** {ep.suggested_fix}")
            ep_lines.append("")
            ep_lines.append("**Instructions:** Make a MINIMAL edit to fix the root cause. Do not rewrite the entire script.")
            sections.append('\n'.join(ep_lines))
        
        # Verification report
        if observation.verification_report:
            ver_str = json.dumps(observation.verification_report, indent=2, default=str)
            sections.append(f"## Verification Report\n```json\n{ver_str}\n```")
        
        # Goal progress
        if observation.goal_progress:
            goal_str = json.dumps(observation.goal_progress, indent=2, default=str)
            sections.append(f"## Goal Progress\n```json\n{goal_str}\n```")
        
        # P2 #24: Self-check instructions
        self_check = """## Self-Check (BEFORE requesting execution)

Before setting request_execution=true, verify:
1. Script computes WORKSPACE_DIR from `__file__` (NOT from cwd)
2. Script loads spec.json and tool_registry.json from WORKSPACE_DIR
3. Script writes to OUTPUT_DIR (from env var ORGAN_AGENT_OUTPUT_DIR)
4. Script creates ALL required output files with EXACT paths
5. Script prints ARTIFACTS_JSON footer with created files and metrics

If any of these are missing, fix them first."""
        
        # Instructions
        instructions = """## Your Task

Based on the above context, decide what to do next. Respond with a JSON object as specified in the system prompt.

Remember:
- If you need more information, ask questions (propose defaults per P2 #19)
- If the master script needs changes, include the full updated content
- If the script is ready to run, set request_execution=true
- If generation is complete and verified, set stop=true
- Prefer minimal changes to existing working code
- Update the plan_board if your strategy changes"""
        
        sections.append(self_check)
        sections.append(instructions)
        
        return '\n\n'.join(sections)
    
    def decide_next(
        self,
        observation: ObservationPacket,
    ) -> Directive:
        """
        Decide what to do next based on the observation.
        
        Parameters
        ----------
        observation : ObservationPacket
            The current observation
            
        Returns
        -------
        Directive
            The decision about what to do next
        """
        prompt = self.build_prompt(observation)
        
        # Add user message to conversation history
        if observation.user_message:
            self._conversation_history.append({
                "role": "user",
                "content": observation.user_message,
            })
        
        try:
            response = self.llm_client.chat(
                message=prompt,
                system_prompt=self.system_prompt,
                temperature=0.7,
            )
            
            directive = self._parse_response(response.content)
            
            # Add assistant response to conversation history
            if directive.assistant_message:
                self._conversation_history.append({
                    "role": "assistant",
                    "content": directive.assistant_message,
                })
            
            return directive
            
        except Exception as e:
            logger.exception("Error calling LLM")
            return Directive(
                assistant_message=f"I encountered an error: {str(e)}. Please try again.",
                reasoning=f"LLM call failed: {str(e)}",
            )
    
    def _parse_response(self, response_text: str) -> Directive:
        """
        Parse the LLM response into a Directive.
        
        Uses fenced-json parsing (```json ... ```) as the primary method.
        This is more robust than regex matching braces, which breaks on
        code strings containing braces.
        
        Parameters
        ----------
        response_text : str
            The raw LLM response
            
        Returns
        -------
        Directive
            The parsed directive
        """
        import re
        
        # Method 1: Try to parse the entire response as JSON (clean output)
        try:
            data = json.loads(response_text.strip())
            return Directive.from_dict(data)
        except json.JSONDecodeError:
            pass
        
        # Method 2: Find fenced JSON block (```json ... ``` or ``` ... ```)
        # This is the recommended format and most reliable
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return Directive.from_dict(data)
            except json.JSONDecodeError as e:
                logger.warning(f"Found fenced block but failed to parse: {e}")
        
        # Method 3: Find JSON starting with { and ending with } at line boundaries
        # More robust than regex matching nested braces
        lines = response_text.split('\n')
        json_start = None
        brace_count = 0
        json_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if json_start is None and stripped.startswith('{'):
                json_start = i
                brace_count = 0
            
            if json_start is not None:
                json_lines.append(line)
                brace_count += stripped.count('{') - stripped.count('}')
                
                if brace_count == 0 and stripped.endswith('}'):
                    # Found complete JSON object
                    try:
                        json_text = '\n'.join(json_lines)
                        data = json.loads(json_text)
                        return Directive.from_dict(data)
                    except json.JSONDecodeError:
                        # Reset and continue looking
                        json_start = None
                        json_lines = []
                        brace_count = 0
        
        # Fallback: treat the entire response as the assistant message
        logger.warning("Could not parse LLM response as JSON, using as plain message")
        return Directive(
            assistant_message=response_text,
            reasoning="Failed to parse structured response, using raw text",
        )
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self._conversation_history.clear()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return list(self._conversation_history)


def create_initial_master_script(spec_data: Dict[str, Any]) -> str:
    """
    Create an initial master script template based on the spec.
    
    Parameters
    ----------
    spec_data : dict
        The spec data from spec.json
        
    Returns
    -------
    str
        The initial master script content
    """
    facts = spec_data.get("facts", {})
    
    # Extract key parameters
    domain_type = facts.get("domain.type", {}).get("value", "box")
    domain_size = facts.get("domain.size", {}).get("value", [0.02, 0.06, 0.03])
    topology_kind = facts.get("topology.kind", {}).get("value", "tree")
    inlet_radius = facts.get("inlet.radius", {}).get("value", 0.002)
    outlet_radius = facts.get("outlet.radius", {}).get("value", 0.002)
    
    return f'''#!/usr/bin/env python3
"""
Master Script - Vascular Network Generation

Auto-generated initial template. The LLM will iterate on this script.
"""

import os
import sys
import json

# Output directory from environment (set by subprocess_runner to run_dir)
OUTPUT_DIR = os.environ.get('ORGAN_AGENT_OUTPUT_DIR', os.getcwd())

# Workspace directory: use WORKSPACE_PATH env var, or derive from __file__
# master.py lives in workspace root, so dirname(__file__) is the workspace
WORKSPACE_DIR = os.environ.get('WORKSPACE_PATH', os.path.dirname(os.path.abspath(__file__)))

# Tools directory for generated tool imports
TOOLS_DIR = os.path.join(WORKSPACE_DIR, 'tools')
if os.path.isdir(TOOLS_DIR):
    sys.path.insert(0, TOOLS_DIR)


def load_spec():
    """Load the specification from spec.json."""
    spec_path = os.path.join(WORKSPACE_DIR, 'spec.json')
    if os.path.exists(spec_path):
        with open(spec_path, 'r') as f:
            return json.load(f)
    return {{"facts": {{}}}}


def main():
    """Main generation function."""
    spec = load_spec()
    facts = spec.get("facts", {{}})
    
    # Extract parameters from spec
    domain_type = facts.get("domain.type", {{}}).get("value", "{domain_type}")
    domain_size = facts.get("domain.size", {{}}).get("value", {domain_size})
    topology_kind = facts.get("topology.kind", {{}}).get("value", "{topology_kind}")
    inlet_radius = facts.get("inlet.radius", {{}}).get("value", {inlet_radius})
    outlet_radius = facts.get("outlet.radius", {{}}).get("value", {outlet_radius})
    
    print(f"Generating {{topology_kind}} network in {{domain_type}} domain")
    print(f"Domain size: {{domain_size}}")
    print(f"Inlet radius: {{inlet_radius}}, Outlet radius: {{outlet_radius}}")
    
    # TODO: Implement generation logic using tools from registry
    # Example:
    # from generation.ops.space_colonization import space_colonization_step
    # from generation.core.domain import create_box_domain
    # from generation.core.network import VascularNetwork
    # 
    # domain = create_box_domain(size=domain_size)
    # network = VascularNetwork()
    # # ... use space_colonization_step to grow the network ...
    
    # Placeholder outputs
    created_files = []
    metrics = {{
        "domain_type": domain_type,
        "topology_kind": topology_kind,
        "status": "template_only",
    }}
    
    # Print artifacts JSON footer
    artifacts = {{
        "files": created_files,
        "metrics": metrics,
        "status": "success"
    }}
    print(f"ARTIFACTS_JSON: {{json.dumps(artifacts)}}")


if __name__ == "__main__":
    main()
'''
