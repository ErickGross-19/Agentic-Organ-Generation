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
    """A file to create or update."""
    path: str  # Relative path within workspace (e.g., "master.py", "tools/my_tool.py")
    content: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "content": self.content,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FileUpdate":
        return cls(
            path=d["path"],
            content=d["content"],
        )


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
class Directive:
    """
    The LLM's decision about what to do next.
    
    This is the structured output from the brain that the controller acts on.
    """
    assistant_message: str = ""
    questions: List[Question] = field(default_factory=list)
    workspace_update: Optional[WorkspaceUpdate] = None
    request_execution: bool = False
    stop: bool = False
    reasoning: str = ""  # Internal reasoning (for debugging/tracing)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "assistant_message": self.assistant_message,
            "questions": [q.to_dict() for q in self.questions],
            "workspace_update": self.workspace_update.to_dict() if self.workspace_update else None,
            "request_execution": self.request_execution,
            "stop": self.stop,
            "reasoning": self.reasoning,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Directive":
        workspace_update = None
        if d.get("workspace_update"):
            workspace_update = WorkspaceUpdate.from_dict(d["workspace_update"])
        
        return cls(
            assistant_message=d.get("assistant_message", ""),
            questions=[Question.from_dict(q) for q in d.get("questions", [])],
            workspace_update=workspace_update,
            request_execution=d.get("request_execution", False),
            stop=d.get("stop", False),
            reasoning=d.get("reasoning", ""),
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
1. Load spec.json for parameters
2. Use tools from the registry
3. Write outputs to OUTPUT_DIR (from environment)
4. Print ARTIFACTS_JSON footer with created files and metrics

Example structure:
```python
import os
import json

OUTPUT_DIR = os.environ.get('ORGAN_AGENT_OUTPUT_DIR', os.getcwd())

def load_spec():
    with open(os.path.join(OUTPUT_DIR, '..', 'agent_workspace', 'spec.json')) as f:
        return json.load(f)

def main():
    spec = load_spec()
    # ... generation logic using tools ...
    # ... write outputs ...
    print(f'ARTIFACTS_JSON: {json.dumps({"files": [...], "metrics": {...}, "status": "success"})}')

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
            
        Returns
        -------
        ObservationPacket
            The observation packet for the LLM
        """
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
            sections.append('\n'.join(ws_lines))
        
        # Master script content
        if observation.master_script_content and observation.master_script_content != "[No master script exists yet]":
            sections.append(f"## Current Master Script\n```python\n{observation.master_script_content}\n```")
        
        # Tool registry
        if observation.tool_registry_description:
            sections.append(f"## Tool Registry\n{observation.tool_registry_description}")
        
        # Last run result
        if observation.last_run_result:
            run_str = json.dumps(observation.last_run_result, indent=2, default=str)
            sections.append(f"## Last Run Result\n```json\n{run_str}\n```")
        
        # Verification report
        if observation.verification_report:
            ver_str = json.dumps(observation.verification_report, indent=2, default=str)
            sections.append(f"## Verification Report\n```json\n{ver_str}\n```")
        
        # Goal progress
        if observation.goal_progress:
            goal_str = json.dumps(observation.goal_progress, indent=2, default=str)
            sections.append(f"## Goal Progress\n```json\n{goal_str}\n```")
        
        # Instructions
        sections.append("""## Your Task

Based on the above context, decide what to do next. Respond with a JSON object as specified in the system prompt.

Remember:
- If you need more information, ask questions
- If the master script needs changes, include the full updated content
- If the script is ready to run, set request_execution=true
- If generation is complete and verified, set stop=true
- Prefer minimal changes to existing working code""")
        
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
        
        Parameters
        ----------
        response_text : str
            The raw LLM response
            
        Returns
        -------
        Directive
            The parsed directive
        """
        # Try to extract JSON from the response
        try:
            # First, try to parse the entire response as JSON
            data = json.loads(response_text)
            return Directive.from_dict(data)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON block in markdown
        import re
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return Directive.from_dict(data)
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object anywhere in the response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return Directive.from_dict(data)
            except json.JSONDecodeError:
                pass
        
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

# Output directory from environment
OUTPUT_DIR = os.environ.get('ORGAN_AGENT_OUTPUT_DIR', os.getcwd())
WORKSPACE_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), 'agent_workspace')

# Add repo root to path for imports
REPO_ROOT = os.environ.get('PYTHONPATH', '').split(':')[0]
if REPO_ROOT:
    sys.path.insert(0, REPO_ROOT)


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
    # from generation.ops.space_colonization import grow_network, SpaceColonizationParams
    # from generation.domain import BoxDomain
    # 
    # domain = BoxDomain(size=domain_size)
    # params = SpaceColonizationParams(...)
    # network = grow_network(domain, params)
    
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
