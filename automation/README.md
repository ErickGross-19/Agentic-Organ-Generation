# Automation Scripts (Part C)

The Automation module provides scripts for LLM API calls and agent behavior. It enables programmatic interaction with LLM APIs (OpenAI, Anthropic, etc.) to generate and validate organ structures using natural language instructions.

## Overview

The automation module consists of four main components:

1. **LLM Client**: A unified client for interacting with various LLM APIs
2. **Agent Runner**: An orchestrator for running generation and validation tasks
3. **Task Templates**: Pre-built prompts for common operations
4. **Single Agent Organ Generator V1**: An interactive workflow for guided organ structure generation

## Directory Structure

```
automation/
├── __init__.py              # Main entry point
├── llm_client.py            # LLM API client
├── agent_runner.py          # Agent orchestration
├── workflow.py              # Single Agent Organ Generator V1 workflow
├── cli.py                   # Command-line interface
└── task_templates/          # Pre-built task prompts
    ├── __init__.py
    ├── generate_structure.py   # Generation prompts
    ├── validate_structure.py   # Validation prompts
    └── iterate_design.py       # Iteration prompts
```

## Quick Start

### Basic Usage

```python
from automation import AgentRunner, LLMClient
from automation.task_templates import generate_structure_prompt

# Initialize client
client = LLMClient(provider="openai", api_key="sk-...")

# Create agent runner
runner = AgentRunner(client=client, output_dir="./output")

# Run a generation task
result = runner.run_task(
    task=generate_structure_prompt(
        organ_type="liver",
        constraints={"plate_size": (200, 200, 200)}
    )
)

print(f"Status: {result.status}")
print(f"Artifacts: {result.artifacts}")
```

### Using the CLI

```bash
# Generate a liver structure
python -m automation.cli generate --organ liver --segments 500 --output ./output

# Validate a structure
python -m automation.cli validate --input structure.stl --stage both

# Iteratively improve a design
python -m automation.cli iterate --input design.stl --max-iterations 5

# Start interactive session
python -m automation.cli interactive
```

## LLM Client

### Supported Providers

| Provider | Model Examples | Environment Variable |
|----------|---------------|---------------------|
| OpenAI | gpt-4, gpt-3.5-turbo, gpt-5 | `OPENAI_API_KEY` |
| Anthropic | claude-3-opus, claude-3-sonnet | `ANTHROPIC_API_KEY` |
| xAI/Grok | grok-beta, grok-2 | `XAI_API_KEY` |
| Google/Gemini | gemini-pro, gemini-1.5-pro | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| Mistral | mistral-large, mistral-medium | `MISTRAL_API_KEY` |
| Groq | llama-3.1-70b, mixtral-8x7b | `GROQ_API_KEY` |
| Devin | devin (session-based) | `DEVIN_API_KEY` |
| Local | Any OpenAI-compatible API | N/A (use `api_base`) |

### Configuration

```python
from automation.llm_client import LLMClient, LLMConfig

# Using environment variables
client = LLMClient(provider="openai")

# Explicit configuration
config = LLMConfig(
    provider="openai",
    model="gpt-4",
    api_key="sk-...",
    max_tokens=4096,
    temperature=0.7,
    system_prompt="You are an expert biomedical engineer...",
)
client = LLMClient(config=config)

# Using Anthropic
client = LLMClient(
    provider="anthropic",
    model="claude-3-opus-20240229",
)

# Using local model (OpenAI-compatible API)
client = LLMClient(
    provider="local",
    api_base="http://localhost:8000/v1",
    model="local-model",
)
```

### Chat Interface

```python
# Single message
response = client.chat("Generate a vascular network design spec")
print(response.content)
print(f"Tokens used: {response.usage['total_tokens']}")

# Conversation mode
response1 = client.chat("Create a liver network", continue_conversation=True)
response2 = client.chat("Now add a venous tree", continue_conversation=True)

# Clear history
client.clear_history()

# Get conversation history
history = client.get_history()
```

## Agent Runner

### Configuration

```python
from automation.agent_runner import AgentRunner, AgentConfig

config = AgentConfig(
    repo_path="/path/to/Agentic-Organ-Generation",
    output_dir="./output",
    max_iterations=10,
    auto_execute_code=False,  # Safety: don't auto-execute generated code
    verbose=True,
    save_conversation=True,
)

runner = AgentRunner(client=client, config=config)
```

### Running Tasks

```python
# Run a task
result = runner.run_task(
    task="Generate a liver vascular network with 500 segments",
    context={
        "constraints": {"plate_size": (200, 200, 200)},
        "output_format": "domain_with_void",
    },
)

# Check result
if result.status == TaskStatus.COMPLETED:
    print("Task completed successfully!")
    print(f"Output: {result.output}")
    print(f"Artifacts: {result.artifacts}")
elif result.status == TaskStatus.NEEDS_INPUT:
    print("Agent needs more information:")
    print(result.output)
elif result.status == TaskStatus.FAILED:
    print(f"Task failed: {result.error}")

# Get statistics
print(f"Iterations: {result.iterations}")
print(f"Total tokens: {result.total_tokens}")
```

### Interactive Mode

```python
# Start interactive session
runner.run_interactive(initial_task="Help me design a kidney network")

# Or from CLI
# python -m automation.cli interactive --task "Design a kidney network"
```

### Convenience Function

```python
from automation.agent_runner import create_agent

# Quick setup
agent = create_agent(
    provider="openai",
    model="gpt-4",
    output_dir="./my_output",
    verbose=True,
)

result = agent.run_task("Generate a liver network")
```

## Task Templates

### Generation Templates

```python
from automation.task_templates import (
    generate_structure_prompt,
    generate_liver_prompt,
    generate_custom_organ_prompt,
)

# Generic organ generation
prompt = generate_structure_prompt(
    organ_type="kidney",
    constraints={
        "plate_size": (150, 150, 150),
        "min_channel_diameter": 0.5,
        "num_segments": 300,
    },
    output_format="domain_with_void",
)

# Liver-specific generation
prompt = generate_liver_prompt(
    arterial_segments=500,
    venous_segments=500,
    plate_size=(200, 200, 200),
    seed=42,
)

# Custom organ with specific domain
prompt = generate_custom_organ_prompt(
    organ_name="pancreas",
    domain_shape="ellipsoid",
    domain_dimensions=(80, 40, 30),
    inlet_positions=[(0, 0, 30)],
    outlet_positions=[(0, 0, -30)],
)
```

### Validation Templates

```python
from automation.task_templates import (
    validate_structure_prompt,
    validate_pre_embedding_prompt,
    validate_post_embedding_prompt,
)

# Full validation
prompt = validate_structure_prompt(
    mesh_path="structure.stl",
    validation_stage="both",
    manufacturing_constraints={
        "min_channel_diameter": 0.5,
        "min_wall_thickness": 0.3,
    },
)

# Pre-embedding only
prompt = validate_pre_embedding_prompt(
    mesh_path="structure.stl",
    network_path="network.json",
)

# Post-embedding only
prompt = validate_post_embedding_prompt(
    embedded_mesh_path="domain_with_void.stl",
    expected_outlets=2,
)
```

### Iteration Templates

```python
from automation.task_templates import (
    iterate_design_prompt,
    fix_validation_issues_prompt,
    optimize_structure_prompt,
)

# Iterative improvement
prompt = iterate_design_prompt(
    current_design_path="design.stl",
    validation_report_path="report.json",
    improvement_goals=[
        "Fix Murray's law violations",
        "Improve flow uniformity",
    ],
    max_iterations=5,
)

# Fix specific issues
prompt = fix_validation_issues_prompt(
    mesh_path="structure.stl",
    validation_report=report_dict,
)

# Optimize for specific target
prompt = optimize_structure_prompt(
    mesh_path="structure.stl",
    optimization_target="flow_uniformity",  # or "coverage", "printability", "all"
)
```

## Command-Line Interface

### Commands

```bash
# Generate command
python -m automation.cli generate \
    --organ liver \
    --segments 500 \
    --plate-size 200,200,200 \
    --seed 42 \
    --output ./output \
    --provider openai \
    --model gpt-4 \
    --verbose

# Validate command
python -m automation.cli validate \
    --input structure.stl \
    --stage both \
    --min-channel 0.5 \
    --min-wall 0.3 \
    --output ./output

# Iterate command
python -m automation.cli iterate \
    --input design.stl \
    --report validation_report.json \
    --max-iterations 5 \
    --output ./output

# Interactive command
python -m automation.cli interactive \
    --task "Help me design a liver network" \
    --provider openai
```

### Environment Variables

Set API keys via environment variables:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Integration with Generation and Validity

The automation module is designed to work with the generation and validity libraries:

```python
from automation import AgentRunner, LLMClient
from automation.task_templates import generate_structure_prompt

# The agent has access to these libraries via its system prompt
client = LLMClient(provider="openai")
runner = AgentRunner(client=client)

# The agent will use generation and validity modules
result = runner.run_task("""
Generate a liver vascular network with the following requirements:
1. 500 arterial segments, 500 venous segments
2. Build plate size: 200x200x200 mm
3. Minimum channel diameter: 0.5 mm

After generation:
1. Run pre-embedding validation
2. Embed into domain
3. Run post-embedding validation
4. Export domain-with-void scaffold and surface mesh

Report any validation issues and suggest fixes.
""")
```

## Examples

See the `examples/` directory for working examples:

- `examples/single_agent_organgenerator_v1.ipynb` - Interactive Jupyter notebook demonstrating the Single Agent Organ Generator V1 workflow with all features including basic generation, interactive sessions, and validation workflows

## Single Agent Organ Generator V1 Workflow

The Single Agent Organ Generator V1 is an interactive workflow that guides users through the complete organ structure generation process.

### Workflow States

The workflow progresses through the following states:

| State | Description |
|-------|-------------|
| `INIT` | Ask for project name and output units |
| `REQUIREMENTS` | Ask for structure description |
| `GENERATING` | Generate structure using the library |
| `VISUALIZING` | Show 3D visualization and generated files |
| `REVIEW` | Ask if user is satisfied |
| `CLARIFYING` | If not satisfied, ask what's wrong |
| `FINALIZING` | Output embedded structure, STL mesh, and code |
| `COMPLETE` | Workflow finished |

### Usage

#### Programmatic Usage

```python
from automation.workflow import SingleAgentOrganGeneratorV1, run_single_agent_workflow
from automation.agent_runner import create_agent

# Option 1: Use convenience function
context = run_single_agent_workflow(
    provider="openai",
    model="gpt-4",
    base_output_dir="./my_projects"
)

# Option 2: Create workflow manually for more control
agent = create_agent(provider="openai", model="gpt-4")
workflow = SingleAgentOrganGeneratorV1(
    agent=agent,
    base_output_dir="./my_projects",
    verbose=True,
)

# Run interactively
context = workflow.run()

# Or step through programmatically
workflow.step("my_project")  # INIT -> REQUIREMENTS
workflow.step("Generate a liver vascular network")  # REQUIREMENTS -> GENERATING
# ... etc
```

#### CLI Usage

```bash
# Run the workflow interactively
python -m automation.cli workflow

# With custom output directory
python -m automation.cli workflow --output ./my_projects

# With specific LLM provider
python -m automation.cli workflow --provider openai --model gpt-4
```

### Generated Artifacts

The workflow generates the following artifacts in the project output directory:

| File | Description |
|------|-------------|
| `design_spec.json` | The design specification used for generation |
| `network.json` | The generated vascular network |
| `structure.stl` | Surface mesh of the vascular structure |
| `embedded_structure.stl` | Domain with void carved out |
| `generation_code.py` | Python code to reproduce the generation |
| `project_summary.json` | Summary of the project with all file paths |

### State Persistence

The workflow supports saving and loading state for resuming interrupted sessions:

```python
# Save state
workflow.save_state("workflow_state.json")

# Load state later
workflow.load_state("workflow_state.json")
workflow.run()  # Resume from saved state
```

## Devin Integration

Devin AI is now supported as an LLM provider. Unlike traditional chat-based LLMs, Devin uses a session-based API where tasks are executed asynchronously.

### Using Devin

```python
from automation import LLMClient, create_agent

# Using LLMClient directly
client = LLMClient(provider="devin")  # Uses DEVIN_API_KEY env var
response = client.chat("Generate a liver vascular network with 500 segments")
print(response.content)

# Using create_agent convenience function
agent = create_agent(provider="devin", output_dir="./output")
result = agent.run_task("Generate a liver network")
```

### CLI Usage

```bash
# Set your API key
export DEVIN_API_KEY="your-api-key"

# Run generation with Devin
python -m automation.cli generate --organ liver --segments 500 --provider devin
```

### How It Works

The Devin provider adapts the session-based Devin API to work within the standard LLMClient interface:

1. **Session Creation**: When you call `chat()`, a new Devin session is created with your prompt
2. **Polling**: The client polls the session status until completion (blocked/stopped)
3. **Response Extraction**: Results are extracted from the session's structured output
4. **Conversation Continuity**: Subsequent `chat()` calls with `continue_conversation=True` send messages to the existing session

Note: Devin sessions may take longer than traditional LLM calls since Devin executes actual code and tasks. The default timeout is 5 minutes.
