"""
Agent Runner

Generic agent runner that orchestrates LLM interactions for organ structure
generation and validation tasks. Works with standard LLM APIs.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
from enum import Enum
import json
import time
import os

from .llm_client import LLMClient, LLMConfig, LLMResponse


class TaskStatus(Enum):
    """Status of a task execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_INPUT = "needs_input"


@dataclass
class AgentConfig:
    """
    Configuration for the agent runner.
    
    Attributes
    ----------
    repo_path : str
        Path to the Agentic-Organ-Generation repository
    output_dir : str
        Directory for output files
    max_iterations : int
        Maximum number of LLM interaction iterations
    auto_execute_code : bool
        Whether to automatically execute generated code
    verbose : bool
        Whether to print detailed progress
    save_conversation : bool
        Whether to save conversation history to file
    """
    repo_path: str = "."
    output_dir: str = "./output"
    max_iterations: int = 10
    auto_execute_code: bool = False
    verbose: bool = True
    save_conversation: bool = True


@dataclass
class TaskResult:
    """
    Result of a task execution.
    
    Attributes
    ----------
    status : TaskStatus
        Final status of the task
    output : Any
        Task output (e.g., generated structure, validation report)
    messages : List[Dict]
        Conversation history
    artifacts : Dict[str, str]
        Generated artifacts (file paths)
    error : str, optional
        Error message if task failed
    iterations : int
        Number of iterations taken
    total_tokens : int
        Total tokens used
    """
    status: TaskStatus
    output: Any
    messages: List[Dict[str, str]]
    artifacts: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    iterations: int = 0
    total_tokens: int = 0


class AgentRunner:
    """
    Agent runner for orchestrating LLM-driven organ structure generation.
    
    This class provides a high-level interface for running generation and
    validation tasks using LLM APIs. It handles conversation management,
    code execution, and artifact generation.
    
    Parameters
    ----------
    client : LLMClient
        LLM client for API interactions
    config : AgentConfig, optional
        Agent configuration
        
    Examples
    --------
    >>> from automation import AgentRunner, LLMClient
    >>> from automation.task_templates import generate_structure_prompt
    >>>
    >>> # Initialize
    >>> client = LLMClient(provider="openai", api_key="sk-...")
    >>> runner = AgentRunner(client=client)
    >>>
    >>> # Run a generation task
    >>> result = runner.run_task(
    ...     task="Generate a liver vascular network with 500 segments"
    ... )
    >>> print(f"Status: {result.status}")
    >>> print(f"Artifacts: {result.artifacts}")
    """
    
    # System prompt for organ generation tasks
    DEFAULT_SYSTEM_PROMPT = '''You are an expert biomedical engineer specializing in organ structure generation.
You have access to the Agentic-Organ-Generation library which provides:

GENERATION (Part A):
- generation.api.design_from_spec(): Build networks from specifications
- generation.api.run_experiment(): Complete design -> evaluate -> export workflow
- generation.ops: Low-level operations (create_network, add_inlet, space_colonization_step, etc.)
- generation.organ_generators.liver: Liver-specific network generator

VALIDATION (Part B):
- validity.run_pre_embedding_validation(): Validate structure before embedding
- validity.run_post_embedding_validation(): Validate after embedding into domain
- validity.pre_embedding: Mesh checks, graph checks, flow checks
- validity.post_embedding: Connectivity, printability, domain checks

When generating structures:
1. First understand the user's requirements (organ type, constraints, output format)
2. Choose appropriate generation method (high-level spec or low-level ops)
3. Generate the structure
4. Run validation checks
5. Export to requested format (domain-with-void scaffold + surface mesh)

Always provide complete, runnable Python code. Include all necessary imports.
Explain your approach and any trade-offs made.'''
    
    def __init__(
        self,
        client: LLMClient,
        config: Optional[AgentConfig] = None,
    ):
        self.client = client
        self.config = config or AgentConfig()
        
        # Set system prompt if not already set
        if self.client.config.system_prompt is None:
            self.client.config.system_prompt = self.DEFAULT_SYSTEM_PROMPT
            self.client.clear_history()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
    
    def run_task(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
        on_message: Optional[Callable[[str, str], None]] = None,
    ) -> TaskResult:
        """
        Run a task using the LLM.
        
        Parameters
        ----------
        task : str
            Task description or prompt
        context : dict, optional
            Additional context to include in the prompt
        on_message : callable, optional
            Callback for each message (role, content)
            
        Returns
        -------
        TaskResult
            Result of the task execution
        """
        messages = []
        artifacts = {}
        total_tokens = 0
        
        # Build initial prompt
        prompt = self._build_prompt(task, context)
        
        if self.config.verbose:
            print(f"Starting task: {task[:100]}...")
        
        for iteration in range(self.config.max_iterations):
            if self.config.verbose:
                print(f"Iteration {iteration + 1}/{self.config.max_iterations}")
            
            try:
                # Get LLM response
                response = self.client.chat(
                    message=prompt,
                    continue_conversation=(iteration > 0),
                )
                
                total_tokens += response.usage.get("total_tokens", 0)
                
                # Record message
                messages.append({"role": "user", "content": prompt})
                messages.append({"role": "assistant", "content": response.content})
                
                if on_message:
                    on_message("assistant", response.content)
                
                # Check if task is complete
                if self._is_task_complete(response.content):
                    # Extract artifacts from response
                    artifacts = self._extract_artifacts(response.content)
                    
                    if self.config.save_conversation:
                        self._save_conversation(messages)
                    
                    return TaskResult(
                        status=TaskStatus.COMPLETED,
                        output=response.content,
                        messages=messages,
                        artifacts=artifacts,
                        iterations=iteration + 1,
                        total_tokens=total_tokens,
                    )
                
                # Check if LLM needs input
                if self._needs_user_input(response.content):
                    return TaskResult(
                        status=TaskStatus.NEEDS_INPUT,
                        output=response.content,
                        messages=messages,
                        iterations=iteration + 1,
                        total_tokens=total_tokens,
                    )
                
                # Prepare next prompt (continue conversation)
                prompt = "Please continue with the task."
                
            except Exception as e:
                return TaskResult(
                    status=TaskStatus.FAILED,
                    output=None,
                    messages=messages,
                    error=str(e),
                    iterations=iteration + 1,
                    total_tokens=total_tokens,
                )
        
        # Max iterations reached
        return TaskResult(
            status=TaskStatus.FAILED,
            output=messages[-1]["content"] if messages else None,
            messages=messages,
            error="Max iterations reached",
            iterations=self.config.max_iterations,
            total_tokens=total_tokens,
        )
    
    def run_interactive(
        self,
        initial_task: Optional[str] = None,
    ) -> None:
        """
        Run an interactive session with the agent.
        
        Parameters
        ----------
        initial_task : str, optional
            Initial task to start with
        """
        print("Agentic Organ Generation - Interactive Mode")
        print("Type 'quit' or 'exit' to end the session")
        print("Type 'clear' to clear conversation history")
        print("-" * 50)
        
        if initial_task:
            print(f"Initial task: {initial_task}")
            result = self.run_task(initial_task)
            print(f"\nAssistant: {result.output}\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break
                
                if user_input.lower() == "clear":
                    self.client.clear_history()
                    print("Conversation history cleared.")
                    continue
                
                if not user_input:
                    continue
                
                response = self.client.chat(
                    message=user_input,
                    continue_conversation=True,
                )
                
                print(f"\nAssistant: {response.content}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    def _build_prompt(
        self,
        task: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the initial prompt with task and context."""
        prompt_parts = [task]
        
        if context:
            prompt_parts.append("\nContext:")
            for key, value in context.items():
                if isinstance(value, dict):
                    prompt_parts.append(f"- {key}: {json.dumps(value, indent=2)}")
                else:
                    prompt_parts.append(f"- {key}: {value}")
        
        prompt_parts.append(f"\nRepository path: {self.config.repo_path}")
        prompt_parts.append(f"Output directory: {self.config.output_dir}")
        
        return "\n".join(prompt_parts)
    
    def _is_task_complete(self, response: str) -> bool:
        """Check if the task appears to be complete."""
        completion_indicators = [
            "task complete",
            "task is complete",
            "completed successfully",
            "generation complete",
            "validation complete",
            "here is the final",
            "the structure has been generated",
            "the output files are",
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in completion_indicators)
    
    def _needs_user_input(self, response: str) -> bool:
        """Check if the LLM is asking for user input."""
        input_indicators = [
            "please provide",
            "could you specify",
            "what would you like",
            "please clarify",
            "i need more information",
            "which option",
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in input_indicators)
    
    def _extract_artifacts(self, response: str) -> Dict[str, str]:
        """Extract file paths and artifacts from response."""
        artifacts = {}
        
        # Look for file paths in the response
        import re
        
        # Match common file extensions
        file_patterns = [
            r'[\w/.-]+\.stl',
            r'[\w/.-]+\.json',
            r'[\w/.-]+\.py',
            r'[\w/.-]+\.obj',
        ]
        
        for pattern in file_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                ext = match.split('.')[-1]
                if ext not in artifacts:
                    artifacts[ext] = []
                if isinstance(artifacts[ext], list):
                    artifacts[ext].append(match)
                else:
                    artifacts[ext] = [artifacts[ext], match]
        
        return artifacts
    
    def _save_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Save conversation history to file."""
        timestamp = int(time.time())
        filename = f"conversation_{timestamp}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(messages, f, indent=2)
        
        if self.config.verbose:
            print(f"Conversation saved to: {filepath}")
        
        return filepath


def create_agent(
    provider: str = "openai",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    repo_path: str = ".",
    output_dir: str = "./output",
    **kwargs,
) -> AgentRunner:
    """
    Convenience function to create an agent runner.
    
    Parameters
    ----------
    provider : str
        LLM provider ("openai", "anthropic", "local")
    api_key : str, optional
        API key (or set via environment variable)
    model : str, optional
        Model name
    repo_path : str
        Path to repository
    output_dir : str
        Output directory
    **kwargs
        Additional arguments for AgentConfig
        
    Returns
    -------
    AgentRunner
        Configured agent runner
        
    Examples
    --------
    >>> from automation.agent_runner import create_agent
    >>> 
    >>> agent = create_agent(
    ...     provider="openai",
    ...     model="gpt-4",
    ...     output_dir="./my_output"
    ... )
    >>> result = agent.run_task("Generate a liver network")
    """
    # Set default model based on provider
    if model is None:
        if provider == "openai":
            model = "gpt-4"
        elif provider == "anthropic":
            model = "claude-3-opus-20240229"
        else:
            model = "default"
    
    client = LLMClient(
        provider=provider,
        api_key=api_key,
        model=model,
    )
    
    config = AgentConfig(
        repo_path=repo_path,
        output_dir=output_dir,
        **kwargs,
    )
    
    return AgentRunner(client=client, config=config)
