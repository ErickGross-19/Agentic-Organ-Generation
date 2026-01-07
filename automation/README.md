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
| `PROJECT_INIT` | Ask for project name and output directory |
| `OBJECT_PLANNING` | Plan objects (count, names, variants) |
| `FRAME_OF_REFERENCE` | Establish coordinate conventions |
| `REQUIREMENTS_CAPTURE` | Adaptive rule-based requirements gathering |
| `SPEC_COMPILATION` | Compile requirements to design spec |
| `GENERATION` | Generate structure using the library |
| `ANALYSIS_VALIDATION` | Analyze and validate the structure |
| `ITERATION` | Review and iterate on design |
| `FINALIZATION` | Embed and export final artifacts |
| `COMPLETE` | Workflow finished |

## Rule Engine (Adaptive Requirements Capture)

The workflow uses a **rule engine** instead of fixed question groups to adaptively determine what questions to ask based on what's missing, what's ambiguous, and what would cause expensive rework if wrong.

### Core Concept

Instead of asking all questions in sequence (Groups A-G), the rule engine runs an adaptive loop:

1. **Parse user intent** → extract explicit values and detect ambiguities
2. **Run validators** → check for missing fields, ambiguities, and conflicts
3. **Generate minimal questions** → only ask what's needed (max 4 per turn)
4. **Apply answers** → update requirements
5. **Repeat** until generation-ready

### Three Rule Families

#### Family A — Completeness Rules

Ensure required fields exist before proceeding.

| Rule | Description |
|------|-------------|
| A1: Required-field gate | Check for missing required fields (domain, inlets, min_radius, min_clearance) |
| A2: Generation-ready gate | Verify all fields needed for generation are populated |
| A3: Finalization-ready gate | Verify embedding/export fields are ready |

#### Family B — Ambiguity Rules

Trigger only when user language is ambiguous.

| Rule | Description |
|------|-------------|
| B1: Spatial-language | Detect "left/right/top/bottom" → require coordinate convention |
| B2: Vague quantifiers | Detect "dense/thin/big/small" → require numeric mapping |
| B3: Implicit I/O | Detect "perfusion/flow" → require I/O geometry |
| B4: Symmetry | Detect "symmetric" → require axis specification |

#### Family C — Conflict/Feasibility Rules

Prevent specs that cannot work or will cause issues.

| Rule | Description |
|------|-------------|
| C1: Printability | Check min_radius vs voxel_pitch compatibility |
| C2: Clearance | Check clearance vs density feasibility |
| C3: Complexity budget | Check terminals vs configured limits |
| C4: I/O feasibility | Check inlet radius vs domain size |

### Attempt Strategy

Every missing/ambiguous field is handled by a 3-tier approach:

1. **Attempt 1: Infer from user text** (high confidence only)
   - Extract explicit values like "box 2x6x3 cm", "diameter 1mm", "2 inlets"
   
2. **Attempt 2: Propose concrete default** (mark as assumed)
   - Domain → default 0.02×0.06×0.03 m box
   - Min radius → 100 microns
   - Min clearance → 200 microns
   
3. **Attempt 3: Ask targeted question** (only if needed)
   - Questions ranked by rework cost (high → medium → low)

### Rework Cost Ranking

Questions are prioritized by how expensive it would be to change later:

| Priority | Fields |
|----------|--------|
| **High** | Coordinate frame, inlet/outlet placement, domain scale, min radius/clearance |
| **Medium** | Target terminals/depth, segment length range |
| **Low** | Aesthetic defaults (angles, tortuosity), export unit preference |

### Using the Rule Engine

```python
from automation import (
    RuleEngine, IntentParser, QuestionPlanner,
    run_rule_based_capture, ObjectRequirements
)

# Create requirements object
requirements = ObjectRequirements()

# Run adaptive capture
updated_req, collected_answers = run_rule_based_capture(
    requirements=requirements,
    intent="I want a dense liver-like branching network with inlet/outlet on the same side",
    organ_type="liver",
    verbose=True
)

# The function will:
# 1. Detect ambiguities ("dense", "same side")
# 2. Propose defaults for missing fields
# 3. Ask only the questions needed
# 4. Return updated requirements
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
