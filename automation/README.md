# Automation Module

This module provides the LLM integration layer for AI-driven organ structure generation and validation. It enables natural language interaction with the generation and validation libraries through a unified interface supporting multiple LLM providers.

## Overview

The automation module comprises four integrated components:

| Component | Purpose |
|-----------|---------|
| **LLM Client** | Unified interface for multiple LLM providers (OpenAI, Anthropic, etc.) |
| **Agent Runner** | Task orchestration with iteration control and artifact management |
| **Task Templates** | Pre-built prompts for generation, validation, and iteration workflows |
| **Single Agent Workflow** | Interactive, guided workflow for complete organ structure generation |

## Execution Modes

Understanding the available execution modes is essential for effective use of this module.

### Available Modes

| Mode | Description | Artifacts Created | Recommended Use Case |
|------|-------------|-------------------|---------------------|
| **Prompt-only** (default) | LLM returns text responses without code execution | None | Planning, prototyping, specification generation |
| **Execution-enabled** | LLM-generated code runs in a sandboxed environment | Only in allowed directories | Testing with controlled output |
| **Spec-only** (recommended) | LLM produces structured specification; Python executes generation locally | Full artifact set | Production workflows |

### Important Considerations

**Default Behavior**: The `AgentRunner` has `auto_execute_code=False` by default. In this mode, the LLM describes intended actions but does not create files.

**Sandbox Restrictions**: When execution is enabled, the `CodeExecutor` enforces security restrictions:
- File writes are permitted only in configured `allowed_output_dirs`
- Blocked patterns include `exec()`, `eval()`, `__import__()`, and `subprocess`
- Execution timeout is enforced (default: 30 seconds)

**Artifact Extraction**: File paths mentioned by the LLM (e.g., `*.stl`) are extracted via pattern matching but may not correspond to existing files.

### Enabling Code Execution

To enable code execution with controlled output directories:

```python
from automation.agent_runner import AgentRunner, AgentConfig, CodeExecutor

# Configure the agent with execution enabled
config = AgentConfig(
    output_dir="./output",
    auto_execute_code=True,
)

runner = AgentRunner(client=client, config=config)

# Restrict file writes to the output directory
runner.code_executor.allowed_output_dirs = ["./output"]
```

### Recommended Approach: Specification-Based Generation

For production workflows, we recommend using the LLM to produce a structured specification, then executing generation locally:

```python
from automation import SingleAgentOrganGeneratorV1
from generation import design_from_spec

# Use the workflow to capture requirements
workflow = SingleAgentOrganGeneratorV1(provider="openai")
workflow.run()

# Extract the compiled specification
spec = workflow.context.get_current_object().spec

# Execute generation locally
network = design_from_spec(spec)
```

This approach provides the benefits of LLM-assisted specification while maintaining deterministic, reproducible generation.

## Module Structure

```
automation/
├── __init__.py              # Public API exports
├── llm_client.py            # Multi-provider LLM client
├── agent_runner.py          # Task orchestration and execution
├── workflow.py              # Interactive workflow implementation
├── cli.py                   # Command-line interface
└── task_templates/          # Structured prompt templates
    ├── __init__.py
    ├── generate_structure.py   # Generation task prompts
    ├── validate_structure.py   # Validation task prompts
    └── iterate_design.py       # Iteration task prompts
```

## Quick Start

### Programmatic Usage

```python
from automation import AgentRunner, LLMClient
from automation.task_templates import generate_structure_prompt

# Initialize the LLM client
client = LLMClient(provider="openai", api_key="sk-...")

# Create the agent runner
runner = AgentRunner(client=client, output_dir="./output")

# Execute a generation task
result = runner.run_task(
    task=generate_structure_prompt(
        organ_type="liver",
        constraints={"plate_size": (200, 200, 200)}
    )
)

print(f"Status: {result.status}")
print(f"Artifacts: {result.artifacts}")
```

### Command-Line Usage

```bash
# Generate a liver vascular structure
python -m automation.cli generate --organ liver --segments 500 --output ./output

# Validate an existing structure
python -m automation.cli validate --input structure.stl --stage both

# Iteratively refine a design
python -m automation.cli iterate --input design.stl --max-iterations 5

# Launch an interactive session
python -m automation.cli interactive
```

## LLM Client

The LLM client provides a unified interface for interacting with multiple language model providers.

### Supported Providers

| Provider | Available Models | Environment Variable |
|----------|-----------------|---------------------|
| **OpenAI** | gpt-4, gpt-3.5-turbo, gpt-5 | `OPENAI_API_KEY` |
| **Anthropic** | claude-3-opus, claude-3-sonnet | `ANTHROPIC_API_KEY` |
| **xAI/Grok** | grok-beta, grok-2 | `XAI_API_KEY` |
| **Google/Gemini** | gemini-pro, gemini-1.5-pro | `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| **Mistral** | mistral-large, mistral-medium | `MISTRAL_API_KEY` |
| **Groq** | llama-3.1-70b, mixtral-8x7b | `GROQ_API_KEY` |
| **Devin** | devin (session-based) | `DEVIN_API_KEY` |
| **Local** | Any OpenAI-compatible API | N/A (use `api_base`) |

### Client Configuration

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

### Conversation Interface

```python
# Single message interaction
response = client.chat("Generate a vascular network design spec")
print(response.content)
print(f"Tokens used: {response.usage['total_tokens']}")

# Multi-turn conversation
response1 = client.chat("Create a liver network", continue_conversation=True)
response2 = client.chat("Now add a venous tree", continue_conversation=True)

# Conversation management
client.clear_history()           # Reset conversation state
history = client.get_history()   # Retrieve conversation history
```

## Agent Runner

The Agent Runner orchestrates task execution, managing iterations, artifacts, and conversation state.

### Configuration

```python
from automation.agent_runner import AgentRunner, AgentConfig

config = AgentConfig(
    repo_path="/path/to/Agentic-Organ-Generation",
    output_dir="./output",
    max_iterations=10,
    auto_execute_code=False,  # Disabled by default for safety
    verbose=True,
    save_conversation=True,
)

runner = AgentRunner(client=client, config=config)
```

### Task Execution

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
# Launch an interactive session
runner.run_interactive(initial_task="Help me design a kidney network")

# Alternatively, use the CLI
# python -m automation.cli interactive --task "Design a kidney network"
```

### Convenience Function

The `create_agent` function provides a simplified setup for common use cases:

```python
from automation.agent_runner import create_agent

agent = create_agent(
    provider="openai",
    model="gpt-4",
    output_dir="./my_output",
    verbose=True,
)

result = agent.run_task("Generate a liver network")
```

## Task Templates

Task templates provide structured prompts for common operations, ensuring consistent and effective LLM interactions.

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

# Liver-specific generation with dual vascular trees
prompt = generate_liver_prompt(
    arterial_segments=500,
    venous_segments=500,
    plate_size=(200, 200, 200),
    seed=42,
)

# Custom organ with explicit domain and port configuration
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

The Single Agent Organ Generator V1 provides an interactive, guided workflow for complete organ structure generation. It combines LLM-assisted requirements capture with deterministic generation and validation.

### Workflow States

The workflow progresses through a defined sequence of states:

| State | Description |
|-------|-------------|
| `PROJECT_INIT` | Configure project name and output directory |
| `OBJECT_PLANNING` | Define object count, names, and variant relationships |
| `FRAME_OF_REFERENCE` | Establish coordinate conventions and spatial reference |
| `REQUIREMENTS_CAPTURE` | Adaptive, rule-based requirements gathering |
| `SPEC_COMPILATION` | Convert requirements to executable design specification |
| `GENERATION` | Execute vascular network generation algorithm |
| `ANALYSIS_VALIDATION` | Perform analysis and validation checks |
| `ITERATION` | Review results and refine design parameters |
| `FINALIZATION` | Embed network and export production artifacts |
| `COMPLETE` | Workflow completed successfully |

## Rule Engine

The workflow employs an adaptive rule engine that intelligently determines which questions to ask based on missing information, ambiguous language, and potential feasibility issues.

### Core Concept

Rather than presenting a fixed sequence of questions, the rule engine operates as an adaptive loop:

1. **Parse user intent**: Extract explicit values and identify ambiguities
2. **Evaluate rules**: Check for missing fields, ambiguous language, and conflicts
3. **Plan questions**: Generate the minimal set of questions needed (maximum 4 per turn)
4. **Apply responses**: Update requirements based on user answers
5. **Iterate**: Repeat until the specification is generation-ready

### Rule Families

The rule engine implements three families of rules, each addressing a different category of specification issues.

#### Family A: Completeness Rules

These rules ensure all required fields are populated before proceeding to generation.

| Rule | Purpose |
|------|---------|
| **A1: Required-field gate** | Verify presence of required fields (domain, inlets, min_radius, min_clearance) |
| **A2: Generation-ready gate** | Confirm all fields necessary for generation are populated |
| **A3: Finalization-ready gate** | Verify embedding and export parameters are defined |

#### Family B: Ambiguity Rules

These rules trigger when user language contains ambiguous terms that require clarification.

| Rule | Purpose |
|------|---------|
| **B1: Spatial language** | Detect relative terms ("left/right/top") and require coordinate convention |
| **B2: Vague quantifiers** | Detect qualitative terms ("dense/thin/big") and require numeric values |
| **B3: Implicit I/O** | Detect flow-related terms and require explicit I/O geometry |
| **B4: Symmetry** | Detect symmetry requirements and require axis specification |

#### Family C: Conflict/Feasibility Rules

These rules identify specifications that would fail or produce impractical results.

| Rule | Purpose |
|------|---------|
| **C1: Printability** | Verify min_radius is compatible with voxel_pitch |
| **C2: Clearance** | Verify clearance requirements are feasible given density targets |
| **C3: Complexity budget** | Verify terminal count is within configured limits |
| **C4: I/O feasibility** | Verify inlet/outlet dimensions are compatible with domain size |

### Resolution Strategy

The rule engine employs a three-tier approach to resolve missing or ambiguous fields:

| Tier | Approach | Example |
|------|----------|---------|
| **1. Inference** | Extract explicit values from user text (high confidence only) | "box 2x6x3 cm" → domain size, "diameter 1mm" → min_radius |
| **2. Defaults** | Propose sensible defaults (marked as assumed) | Domain: 0.02×0.06×0.03 m, Min radius: 100 μm, Min clearance: 200 μm |
| **3. Questions** | Ask targeted questions (ranked by rework cost) | High-cost fields first (coordinate frame, I/O placement) |

### Question Prioritization

Questions are prioritized based on the cost of changing the associated field later in the workflow:

| Priority | Fields | Rationale |
|----------|--------|-----------|
| **High** | Coordinate frame, inlet/outlet placement, domain scale, min radius/clearance | Changes require complete regeneration |
| **Medium** | Target terminals/depth, segment length range | Changes require partial regeneration |
| **Low** | Aesthetic parameters (angles, tortuosity), export units | Changes can be applied post-generation |

### Using the Rule Engine

```python
from automation import (
    RuleEngine, IntentParser, QuestionPlanner,
    run_rule_based_capture, ObjectRequirements
)

# Initialize requirements object
requirements = ObjectRequirements()

# Execute adaptive requirements capture
updated_req, collected_answers = run_rule_based_capture(
    requirements=requirements,
    intent="I want a dense liver-like branching network with inlet/outlet on the same side",
    organ_type="liver",
    verbose=True
)

# The function performs the following steps:
# 1. Detects ambiguities in user intent ("dense", "same side")
# 2. Proposes defaults for missing required fields
# 3. Generates only the questions necessary to proceed
# 4. Returns the updated requirements object
```

### Rule Engine Components

#### IntentParser

Extracts explicit values and detects ambiguities from user text.

```python
from automation import IntentParser

parser = IntentParser("box 2x6x3 cm with diameter 1mm and 2 inlets")

# Check for ambiguities
print(parser.has_spatial_ambiguity())    # False (no left/right/top)
print(parser.has_vague_quantifiers())    # False (no dense/thin)
print(parser.has_implicit_io())          # False (explicit "2 inlets")

# Get extracted values
print(parser.extracted_values)
# {'box_size': ('2', '6', '3', 'cm'), 'diameter': ('1', 'mm'), 'count': ('2',)}
```

#### RuleEngine

Evaluates requirements against all three rule families.

```python
from automation import RuleEngine, ObjectRequirements

engine = RuleEngine(organ_type="liver")
requirements = ObjectRequirements()

result = engine.evaluate(requirements, intent="dense liver network")

# Check results
print(result.is_generation_ready)  # False (missing required fields)
print(len(result.missing_fields))  # Number of missing required fields
print(len(result.ambiguity_flags)) # Number of ambiguity issues
print(len(result.conflict_flags))  # Number of conflict issues
print(len(result.proposed_defaults)) # Number of proposed defaults
```

#### QuestionPlanner

Plans minimal questions ranked by rework cost.

```python
from automation import QuestionPlanner

planner = QuestionPlanner()
questions = planner.plan(eval_result, max_questions=4)

for q in questions:
    print(f"{q.field}: {q.question_text} [{q.default_value}] (cost: {q.rework_cost})")

# Format output for agent communication
output = planner.format_turn_output(requirements, eval_result, questions)
print(output)
```

### Agent Output Format

Each turn, the rule engine outputs:

1. **Current requirements snapshot** — domain, inlets, outlets, constraints
2. **Issues to address** — missing fields, ambiguities, conflicts
3. **Proposed defaults** — values that will be used if not overridden
4. **Questions** — only the questions needed (max 4)

Example output:
```
=== Current Requirements Snapshot ===
Domain: box (0.02, 0.06, 0.03)
Inlets: 0
Outlets: 0
Min radius: None
Min clearance: None
Target terminals: None

=== Issues to Address ===
[!] Missing required field: At least one inlet
[!] Missing required field: Minimum radius
[?] Vague quantifier detected: 'dense' - needs numeric mapping

=== Proposed Defaults ===
  constraints.min_radius_m: 0.0001 (100 micron minimum radius)
  constraints.min_clearance_m: 0.0002 (200 micron minimum clearance)

=== Questions ===
1. How many inlets? [1]
2. Target complexity (terminals)? [300]

(Say 'use defaults' to accept all proposed defaults)
```

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

Devin AI is supported as an LLM provider. Unlike traditional chat-based LLMs, Devin operates through a session-based API where tasks are executed asynchronously.

### Configuration

```python
from automation import LLMClient, create_agent

# Direct client usage
client = LLMClient(provider="devin")  # Uses DEVIN_API_KEY environment variable
response = client.chat("Generate a liver vascular network with 500 segments")
print(response.content)

# Using the convenience function
agent = create_agent(provider="devin", output_dir="./output")
result = agent.run_task("Generate a liver network")
```

### Command-Line Usage

```bash
# Configure API key
export DEVIN_API_KEY="your-api-key"

# Execute generation with Devin
python -m automation.cli generate --organ liver --segments 500 --provider devin
```

### Implementation Details

The Devin provider adapts the session-based API to the standard LLMClient interface:

| Step | Description |
|------|-------------|
| **Session Creation** | A new Devin session is created when `chat()` is called |
| **Status Polling** | The client polls session status until completion |
| **Response Extraction** | Results are extracted from the session's structured output |
| **Conversation Continuity** | Subsequent `chat()` calls with `continue_conversation=True` send messages to the existing session |

Note: Devin sessions may require more time than traditional LLM calls due to actual code execution. The default timeout is 5 minutes.
