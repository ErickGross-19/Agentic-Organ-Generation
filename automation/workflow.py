"""
Single Agent Organ Generator V1 Workflow

This module implements the "Single Agent Organ Generator V1" workflow - a stateful,
interactive workflow for organ structure generation using LLM agents.

The workflow follows these steps:
1. INIT: Ask user for project name and file name
2. REQUIREMENTS: Ask user for description of the system
3. GENERATING: Use the library to generate structure and code
4. VISUALIZING: Output necessary files/code and visualization of 3D object
5. REVIEW: Ask user if it's what they wanted
6. CLARIFYING: If no, ask for clarification and what's wrong (then loop back to GENERATING)
7. FINALIZING: If yes, output embedded structure, STL mesh, and code used to generate
8. COMPLETE: Close project

Usage:
    from automation.workflow import SingleAgentOrganGeneratorV1
    from automation.agent_runner import create_agent
    
    agent = create_agent(provider="openai", model="gpt-4")
    workflow = SingleAgentOrganGeneratorV1(agent)
    workflow.run()
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from pathlib import Path
import json
import time
import os

from .agent_runner import AgentRunner, TaskResult, TaskStatus


class WorkflowState(Enum):
    """States in the Single Agent Organ Generator V1 workflow."""
    INIT = "init"                    # Ask project name/filename
    REQUIREMENTS = "requirements"     # Ask for description
    GENERATING = "generating"         # Generate structure
    VISUALIZING = "visualizing"       # Show 3D visualization
    REVIEW = "review"                 # Ask if satisfied
    CLARIFYING = "clarifying"         # Ask what's wrong
    FINALIZING = "finalizing"         # Output final files
    COMPLETE = "complete"             # Done


@dataclass
class ProjectContext:
    """
    Tracks project state across the workflow.
    
    Attributes
    ----------
    project_name : str
        Name of the project
    output_dir : str
        Directory for output files
    description : str
        User's description of the desired structure
    output_units : str
        Units for output files (mm, cm, m, um)
    spec_json : str, optional
        Path to generated design spec JSON
    network_json : str, optional
        Path to generated network JSON
    stl_path : str, optional
        Path to generated STL file
    embedded_stl_path : str, optional
        Path to embedded structure STL
    code_path : str, optional
        Path to generated Python code
    iteration : int
        Current iteration number
    feedback_history : List[str]
        History of user feedback
    """
    project_name: str = ""
    output_dir: str = ""
    description: str = ""
    output_units: str = "mm"
    spec_json: Optional[str] = None
    network_json: Optional[str] = None
    stl_path: Optional[str] = None
    embedded_stl_path: Optional[str] = None
    code_path: Optional[str] = None
    iteration: int = 0
    feedback_history: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "project_name": self.project_name,
            "output_dir": self.output_dir,
            "description": self.description,
            "output_units": self.output_units,
            "spec_json": self.spec_json,
            "network_json": self.network_json,
            "stl_path": self.stl_path,
            "embedded_stl_path": self.embedded_stl_path,
            "code_path": self.code_path,
            "iteration": self.iteration,
            "feedback_history": self.feedback_history,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectContext":
        """Create from dictionary."""
        return cls(
            project_name=d.get("project_name", ""),
            output_dir=d.get("output_dir", ""),
            description=d.get("description", ""),
            output_units=d.get("output_units", "mm"),
            spec_json=d.get("spec_json"),
            network_json=d.get("network_json"),
            stl_path=d.get("stl_path"),
            embedded_stl_path=d.get("embedded_stl_path"),
            code_path=d.get("code_path"),
            iteration=d.get("iteration", 0),
            feedback_history=d.get("feedback_history", []),
        )


class SingleAgentOrganGeneratorV1:
    """
    Single Agent Organ Generator V1 - Stateful workflow for organ structure generation.
    
    This workflow implements an interactive, LLM-driven process for generating
    organ vascular structures. It guides the user through project setup,
    requirements gathering, generation, review, and finalization.
    
    Parameters
    ----------
    agent : AgentRunner
        The agent runner to use for LLM interactions
    base_output_dir : str
        Base directory for project outputs
    verbose : bool
        Whether to print detailed progress
        
    Examples
    --------
    >>> from automation.workflow import SingleAgentOrganGeneratorV1
    >>> from automation.agent_runner import create_agent
    >>> 
    >>> agent = create_agent(provider="openai", model="gpt-4")
    >>> workflow = SingleAgentOrganGeneratorV1(agent)
    >>> workflow.run()
    """
    
    WORKFLOW_NAME = "Single Agent Organ Generator V1"
    WORKFLOW_VERSION = "1.0.0"
    
    def __init__(
        self,
        agent: AgentRunner,
        base_output_dir: str = "./projects",
        verbose: bool = True,
    ):
        self.agent = agent
        self.base_output_dir = base_output_dir
        self.verbose = verbose
        self.state = WorkflowState.INIT
        self.context = ProjectContext()
        
        os.makedirs(base_output_dir, exist_ok=True)
    
    def run(self) -> ProjectContext:
        """
        Run the complete workflow interactively.
        
        Returns
        -------
        ProjectContext
            Final project context with all generated artifacts
        """
        self._print_header()
        
        while self.state != WorkflowState.COMPLETE:
            self._run_state()
        
        self._print_completion()
        return self.context
    
    def step(self, user_input: str) -> Tuple[WorkflowState, str]:
        """
        Process user input and advance workflow by one step.
        
        This method is useful for programmatic control of the workflow.
        
        Parameters
        ----------
        user_input : str
            User's input for the current state
            
        Returns
        -------
        Tuple[WorkflowState, str]
            New state and response message
        """
        return self._process_input(user_input)
    
    def get_state(self) -> WorkflowState:
        """Get current workflow state."""
        return self.state
    
    def get_context(self) -> ProjectContext:
        """Get current project context."""
        return self.context
    
    def save_state(self, filepath: str) -> None:
        """
        Save workflow state to file for later resumption.
        
        Parameters
        ----------
        filepath : str
            Path to save state JSON
        """
        state_data = {
            "workflow_name": self.WORKFLOW_NAME,
            "workflow_version": self.WORKFLOW_VERSION,
            "state": self.state.value,
            "context": self.context.to_dict(),
            "timestamp": time.time(),
        }
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2)
        if self.verbose:
            print(f"State saved to: {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """
        Load workflow state from file.
        
        Parameters
        ----------
        filepath : str
            Path to state JSON file
        """
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        self.state = WorkflowState(state_data["state"])
        self.context = ProjectContext.from_dict(state_data["context"])
        if self.verbose:
            print(f"State loaded from: {filepath}")
            print(f"Resuming at state: {self.state.value}")
    
    def _print_header(self) -> None:
        """Print workflow header."""
        print("=" * 60)
        print(f"  {self.WORKFLOW_NAME}")
        print(f"  Version: {self.WORKFLOW_VERSION}")
        print("=" * 60)
        print()
        print("This workflow will guide you through generating an organ")
        print("vascular structure. You can type 'quit' at any time to exit.")
        print()
    
    def _print_completion(self) -> None:
        """Print completion message."""
        print()
        print("=" * 60)
        print("  Project Complete!")
        print("=" * 60)
        print()
        print(f"Project: {self.context.project_name}")
        print(f"Output directory: {self.context.output_dir}")
        print()
        print("Generated artifacts:")
        if self.context.spec_json:
            print(f"  - Design spec: {self.context.spec_json}")
        if self.context.network_json:
            print(f"  - Network: {self.context.network_json}")
        if self.context.stl_path:
            print(f"  - STL mesh: {self.context.stl_path}")
        if self.context.embedded_stl_path:
            print(f"  - Embedded structure: {self.context.embedded_stl_path}")
        if self.context.code_path:
            print(f"  - Generation code: {self.context.code_path}")
        print()
        print("Thank you for using the Single Agent Organ Generator V1!")
    
    def _run_state(self) -> None:
        """Run the current state's logic."""
        if self.state == WorkflowState.INIT:
            self._run_init()
        elif self.state == WorkflowState.REQUIREMENTS:
            self._run_requirements()
        elif self.state == WorkflowState.GENERATING:
            self._run_generating()
        elif self.state == WorkflowState.VISUALIZING:
            self._run_visualizing()
        elif self.state == WorkflowState.REVIEW:
            self._run_review()
        elif self.state == WorkflowState.CLARIFYING:
            self._run_clarifying()
        elif self.state == WorkflowState.FINALIZING:
            self._run_finalizing()
    
    def _run_init(self) -> None:
        """Run INIT state: Ask for project name and filename."""
        print("-" * 40)
        print("Step 1: Project Setup")
        print("-" * 40)
        
        project_name = input("Enter project name: ").strip()
        if project_name.lower() in ("quit", "exit"):
            self.state = WorkflowState.COMPLETE
            return
        
        if not project_name:
            project_name = f"project_{int(time.time())}"
            print(f"Using default project name: {project_name}")
        
        self.context.project_name = project_name
        self.context.output_dir = os.path.join(self.base_output_dir, project_name)
        os.makedirs(self.context.output_dir, exist_ok=True)
        
        output_units = input("Enter output units (mm/cm/m/um) [default: mm]: ").strip().lower()
        if output_units in ("quit", "exit"):
            self.state = WorkflowState.COMPLETE
            return
        if output_units not in ("mm", "cm", "m", "um"):
            output_units = "mm"
            print(f"Using default units: {output_units}")
        self.context.output_units = output_units
        
        print(f"\nProject '{project_name}' created at: {self.context.output_dir}")
        print(f"Output units: {self.context.output_units}")
        
        self.state = WorkflowState.REQUIREMENTS
    
    def _run_requirements(self) -> None:
        """Run REQUIREMENTS state: Ask for system description."""
        print()
        print("-" * 40)
        print("Step 2: System Description")
        print("-" * 40)
        print()
        print("Please describe the organ vascular structure you want to generate.")
        print("Include details like:")
        print("  - Organ type (liver, kidney, etc.)")
        print("  - Size and dimensions")
        print("  - Number of vessels/segments")
        print("  - Any specific constraints")
        print()
        
        description = input("Description: ").strip()
        if description.lower() in ("quit", "exit"):
            self.state = WorkflowState.COMPLETE
            return
        
        if not description:
            print("Error: Description is required.")
            return
        
        self.context.description = description
        self.state = WorkflowState.GENERATING
    
    def _run_generating(self) -> None:
        """Run GENERATING state: Generate structure using LLM."""
        print()
        print("-" * 40)
        print(f"Step 3: Generating Structure (Iteration {self.context.iteration + 1})")
        print("-" * 40)
        print()
        print("Generating structure based on your description...")
        print("This may take a moment...")
        print()
        
        feedback_context = ""
        if self.context.feedback_history:
            feedback_context = "\n\nPrevious feedback from user:\n"
            for i, fb in enumerate(self.context.feedback_history, 1):
                feedback_context += f"{i}. {fb}\n"
        
        task = f"""Generate a vascular network structure based on this description:

{self.context.description}
{feedback_context}

Requirements:
1. Use the generation library to create the structure
2. Save the design spec to: {self.context.output_dir}/design_spec.json
3. Save the network to: {self.context.output_dir}/network.json
4. Export STL mesh to: {self.context.output_dir}/structure.stl
5. Use output_units="{self.context.output_units}" for all exports
6. Save the Python code used to: {self.context.output_dir}/generation_code.py

Provide complete, runnable Python code that uses the generation library.
After generating, report the paths to all created files."""

        result = self.agent.run_task(
            task=task,
            context={
                "project_name": self.context.project_name,
                "output_dir": self.context.output_dir,
                "output_units": self.context.output_units,
                "iteration": self.context.iteration,
            }
        )
        
        if result.status == TaskStatus.COMPLETED:
            self.context.spec_json = os.path.join(self.context.output_dir, "design_spec.json")
            self.context.network_json = os.path.join(self.context.output_dir, "network.json")
            self.context.stl_path = os.path.join(self.context.output_dir, "structure.stl")
            self.context.code_path = os.path.join(self.context.output_dir, "generation_code.py")
            self.context.iteration += 1
            
            print("\nGeneration complete!")
            print(f"\nAgent response:\n{result.output[:1000]}...")
            
            self.state = WorkflowState.VISUALIZING
        else:
            print(f"\nGeneration failed: {result.error}")
            print("Please try again with a different description.")
            self.state = WorkflowState.REQUIREMENTS
    
    def _run_visualizing(self) -> None:
        """Run VISUALIZING state: Show visualization and files."""
        print()
        print("-" * 40)
        print("Step 4: Visualization")
        print("-" * 40)
        print()
        print("Generated files:")
        
        if self.context.spec_json and os.path.exists(self.context.spec_json):
            print(f"  [OK] Design spec: {self.context.spec_json}")
        else:
            print(f"  [--] Design spec: Not generated")
        
        if self.context.network_json and os.path.exists(self.context.network_json):
            print(f"  [OK] Network: {self.context.network_json}")
        else:
            print(f"  [--] Network: Not generated")
        
        if self.context.stl_path and os.path.exists(self.context.stl_path):
            print(f"  [OK] STL mesh: {self.context.stl_path}")
        else:
            print(f"  [--] STL mesh: Not generated")
        
        if self.context.code_path and os.path.exists(self.context.code_path):
            print(f"  [OK] Generation code: {self.context.code_path}")
        else:
            print(f"  [--] Generation code: Not generated")
        
        print()
        print("To visualize the 3D structure, you can:")
        print(f"  1. Open {self.context.stl_path} in a 3D viewer (MeshLab, Blender, etc.)")
        print("  2. Use the generation library's visualization tools")
        print()
        
        self.state = WorkflowState.REVIEW
    
    def _run_review(self) -> None:
        """Run REVIEW state: Ask if user is satisfied."""
        print()
        print("-" * 40)
        print("Step 5: Review")
        print("-" * 40)
        print()
        
        response = input("Is this structure what you wanted? (yes/no): ").strip().lower()
        
        if response in ("quit", "exit"):
            self.state = WorkflowState.COMPLETE
            return
        
        if response in ("yes", "y"):
            self.state = WorkflowState.FINALIZING
        elif response in ("no", "n"):
            self.state = WorkflowState.CLARIFYING
        else:
            print("Please answer 'yes' or 'no'.")
    
    def _run_clarifying(self) -> None:
        """Run CLARIFYING state: Ask what's wrong."""
        print()
        print("-" * 40)
        print("Step 6: Clarification")
        print("-" * 40)
        print()
        print("Please describe what's wrong with the structure and what changes you'd like:")
        
        feedback = input("Feedback: ").strip()
        
        if feedback.lower() in ("quit", "exit"):
            self.state = WorkflowState.COMPLETE
            return
        
        if not feedback:
            print("Error: Feedback is required to improve the structure.")
            return
        
        self.context.feedback_history.append(feedback)
        print("\nThank you for your feedback. Regenerating structure...")
        
        self.state = WorkflowState.GENERATING
    
    def _run_finalizing(self) -> None:
        """Run FINALIZING state: Generate final outputs."""
        print()
        print("-" * 40)
        print("Step 7: Finalizing")
        print("-" * 40)
        print()
        print("Generating final outputs (embedded structure, final STL, code)...")
        
        task = f"""Finalize the organ structure project:

1. Create an embedded structure (domain with void carved out) using:
   - embed_tree_as_negative_space() from generation.ops.embedding
   - Save to: {self.context.output_dir}/embedded_structure.stl
   - Use output_units="{self.context.output_units}"

2. Ensure all final files are properly saved:
   - Embedded structure STL
   - Surface mesh STL  
   - Design specification JSON
   - Network JSON
   - Generation code Python file

3. Create a summary JSON file at: {self.context.output_dir}/project_summary.json
   Include: project name, description, all file paths, units, iterations

Use the existing structure from: {self.context.stl_path}
Report all final file paths when complete."""

        result = self.agent.run_task(
            task=task,
            context={
                "project_name": self.context.project_name,
                "output_dir": self.context.output_dir,
                "output_units": self.context.output_units,
                "stl_path": self.context.stl_path,
            }
        )
        
        if result.status == TaskStatus.COMPLETED:
            self.context.embedded_stl_path = os.path.join(
                self.context.output_dir, "embedded_structure.stl"
            )
            print("\nFinalization complete!")
            print(f"\nAgent response:\n{result.output[:1000]}...")
        else:
            print(f"\nFinalization had issues: {result.error}")
            print("Some final outputs may not have been generated.")
        
        self._save_project_summary()
        
        self.state = WorkflowState.COMPLETE
    
    def _save_project_summary(self) -> None:
        """Save project summary JSON."""
        summary = {
            "workflow_name": self.WORKFLOW_NAME,
            "workflow_version": self.WORKFLOW_VERSION,
            "project_name": self.context.project_name,
            "description": self.context.description,
            "output_units": self.context.output_units,
            "iterations": self.context.iteration,
            "feedback_history": self.context.feedback_history,
            "artifacts": {
                "design_spec": self.context.spec_json,
                "network": self.context.network_json,
                "stl_mesh": self.context.stl_path,
                "embedded_structure": self.context.embedded_stl_path,
                "generation_code": self.context.code_path,
            },
            "timestamp": time.time(),
        }
        
        summary_path = os.path.join(self.context.output_dir, "project_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        if self.verbose:
            print(f"\nProject summary saved to: {summary_path}")
    
    def _process_input(self, user_input: str) -> Tuple[WorkflowState, str]:
        """
        Process user input for programmatic workflow control.
        
        Parameters
        ----------
        user_input : str
            User's input
            
        Returns
        -------
        Tuple[WorkflowState, str]
            New state and response message
        """
        if user_input.lower() in ("quit", "exit"):
            self.state = WorkflowState.COMPLETE
            return self.state, "Workflow terminated by user."
        
        if self.state == WorkflowState.INIT:
            self.context.project_name = user_input
            self.context.output_dir = os.path.join(self.base_output_dir, user_input)
            os.makedirs(self.context.output_dir, exist_ok=True)
            self.state = WorkflowState.REQUIREMENTS
            return self.state, f"Project '{user_input}' created. Please provide a description."
        
        elif self.state == WorkflowState.REQUIREMENTS:
            self.context.description = user_input
            self.state = WorkflowState.GENERATING
            return self.state, "Description received. Starting generation..."
        
        elif self.state == WorkflowState.REVIEW:
            if user_input.lower() in ("yes", "y"):
                self.state = WorkflowState.FINALIZING
                return self.state, "Great! Finalizing project..."
            elif user_input.lower() in ("no", "n"):
                self.state = WorkflowState.CLARIFYING
                return self.state, "Please describe what's wrong."
            else:
                return self.state, "Please answer 'yes' or 'no'."
        
        elif self.state == WorkflowState.CLARIFYING:
            self.context.feedback_history.append(user_input)
            self.state = WorkflowState.GENERATING
            return self.state, "Feedback received. Regenerating..."
        
        return self.state, "Unexpected state. Please restart the workflow."


def run_single_agent_workflow(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_output_dir: str = "./projects",
    **kwargs,
) -> ProjectContext:
    """
    Convenience function to run the Single Agent Organ Generator V1 workflow.
    
    Parameters
    ----------
    provider : str
        LLM provider ("openai", "anthropic", "local")
    api_key : str, optional
        API key (or set via environment variable)
    model : str, optional
        Model name
    base_output_dir : str
        Base directory for project outputs
    **kwargs
        Additional arguments for AgentConfig
        
    Returns
    -------
    ProjectContext
        Final project context with all generated artifacts
        
    Examples
    --------
    >>> from automation.workflow import run_single_agent_workflow
    >>> 
    >>> context = run_single_agent_workflow(
    ...     provider="openai",
    ...     model="gpt-4",
    ...     base_output_dir="./my_projects"
    ... )
    >>> print(f"Project: {context.project_name}")
    >>> print(f"STL: {context.stl_path}")
    """
    from .agent_runner import create_agent
    
    agent = create_agent(
        provider=provider,
        api_key=api_key,
        model=model,
        **kwargs,
    )
    
    workflow = SingleAgentOrganGeneratorV1(
        agent=agent,
        base_output_dir=base_output_dir,
    )
    
    return workflow.run()
