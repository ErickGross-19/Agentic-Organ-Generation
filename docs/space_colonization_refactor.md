# Space Colonization Runtime + Progress Refactor

This document describes the refactoring of the space colonization growth backend to eliminate nested loops and per-call tqdm progress bars while preserving multi-inlet interleaving behavior.

## Problem: Nested Loops and Progress Bar Explosion

The original implementation had a nested loop structure that created hundreds or thousands of progress bars:

In `_generate_multi_inlet_blended()`:
```python
for iteration in range(config.max_iterations):  # e.g., 500 iterations
    for i in range(n_inlets):  # e.g., 5 inlets
        result = space_colonization_step(...)  # Creates tqdm bar with max_steps=100!
```

If `max_iterations=500`, `n_inlets=5`, and `max_steps=100`, this created up to 2500 separate progress bars, each showing "Space colonization: 100/100".

Similarly, `_generate_multi_inlet_partitioned()` had:
```python
for i, inlet in enumerate(inlets):  # per inlet
    for iteration in range(iterations_per_inlet):  # iterations per inlet
        result = space_colonization_step(...)  # Creates tqdm bar
```

## Solution: Single-Step Architecture

The refactor introduces a true single-step iteration function and moves iteration control to a single outer loop.

### New Components

1. **SpaceColonizationState** - Persistent state dataclass containing:
   - Network reference and parameters
   - Per-inlet active tip sets
   - Attractor points (remaining)
   - Cached KD-trees with rebuild counters
   - RNG state and counters

2. **space_colonization_one_step()** - True single-step function that:
   - Does exactly ONE iteration of growth
   - NO tqdm bars, NO printing
   - Returns `SingleStepResult` with metrics

3. **Single Outer Loop** - Backend now uses:
   ```python
   for global_iter in range(max_iterations):
       inlet_idx = global_iter % n_inlets  # Round-robin
       result = space_colonization_one_step(state_by_inlet[inlet_idx])
   ```

### Key Features

**Single Optional Progress Bar**: One tqdm bar at the outer level, controlled by config:
- `config.progress = True/False`
- Environment variable: `SPACE_COLONIZATION_PROGRESS=0` to disable

**KD-tree Caching**: Reduces O(N log N) rebuilds:
- `kdtree_rebuild_tip_every`: Rebuild tip KD-tree every N steps (default: 1)
- `kdtree_rebuild_all_nodes_every`: Rebuild all-nodes KD-tree every N steps (default: 10)
- `kdtree_rebuild_all_nodes_min_new_nodes`: Rebuild if this many nodes added (default: 5)

**Stall Detection**: Early stopping when no progress:
- `stall_steps_per_inlet`: Mark inlet as stalled after N steps with no growth (default: 10)
- All inlets stalled triggers early stop

**Interleaving Strategy**:
- `interleaving_strategy: "round_robin"` (default) - Cycle through inlets
- `interleaving_strategy: "weighted"` - Prioritize inlets with more active tips

## Configuration

New backend parameters in `SpaceColonizationConfig`:

```python
use_single_step: bool = True  # Use new single-step implementation
progress: bool = False  # Show progress bar (single bar at outer level)
kdtree_rebuild_tip_every: int = 1
kdtree_rebuild_all_nodes_every: int = 10
kdtree_rebuild_all_nodes_min_new_nodes: int = 5
stall_steps_per_inlet: int = 10
merge_check_every_steps: int = 50
interleaving_strategy: str = "round_robin"
```

Example in DesignSpec JSON:
```json
{
  "backend_params": {
    "progress": false,
    "kdtree_rebuild_tip_every": 1,
    "kdtree_rebuild_all_nodes_every": 10,
    "kdtree_rebuild_all_nodes_min_new_nodes": 5,
    "stall_steps_per_inlet": 10,
    "merge_check_every_steps": 50,
    "interleaving_strategy": "round_robin"
  }
}
```

## Backward Compatibility

The old multi-step functions (`space_colonization_step`, `space_colonization_step_v2`) remain available but are now considered legacy. The new `run_space_colonization_multi_step()` wrapper provides backward compatibility by internally using `space_colonization_one_step()`.

## Performance Improvements

1. **Console Output**: At most one progress bar instead of hundreds/thousands
2. **KD-tree Rebuilds**: Reduced frequency via caching policy
3. **Early Stopping**: Stall detection prevents "dead grinding tail"
4. **Memory**: State objects allow efficient incremental updates

## Files Modified

- `generation/ops/space_colonization.py`: Added `SpaceColonizationState`, `SingleStepResult`, `create_space_colonization_state()`, `space_colonization_one_step()`, `run_space_colonization_multi_step()`
- `generation/backends/space_colonization_backend.py`: Refactored `_generate_multi_inlet_blended()` and `_generate_multi_inlet_partitioned()` to use single outer loop
- `examples/designspec/malaria_venule_space_colonization.json`: Updated with new parameters
