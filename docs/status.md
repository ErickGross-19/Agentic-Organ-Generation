# Backend Status

This document describes the current status of generation backends in the AOG system.

## Active Backends

### scaffold_topdown (Preferred)

The `scaffold_topdown` backend is the recommended choice for generating bifurcating tree structures. It provides:

- Recursive top-down tree generation with configurable branching
- Online collision avoidance during growth
- Post-pass collision resolution
- Support for multiple inlets (forest mode)
- Configurable cone angles, jitter, and curvature

Use this backend for any bifurcating tree generation needs.

### space_colonization

The `space_colonization` backend implements attractor-based organic growth, suitable for:

- Dense vascular networks
- Organic-looking tree structures
- Multi-inlet forest generation with merge support

### programmatic

The `programmatic` backend supports DSL-based generation for:

- Explicit path definitions
- Waypoint-based routing
- Custom network topologies

## Deprecated Backends

### kary_tree (Deprecated)

**Status:** DEPRECATED - Will be removed in a future release

**Replacement:** Use `scaffold_topdown` instead

The `kary_tree` backend is deprecated because `scaffold_topdown` provides better collision avoidance and more flexible tree generation. Existing specs using `kary_tree` will continue to work but will emit deprecation warnings.

To migrate, change:
```json
{
  "policies": {
    "growth": {
      "backend": "kary_tree"
    }
  }
}
```

To:
```json
{
  "policies": {
    "growth": {
      "backend": "scaffold_topdown"
    }
  }
}
```

## Blocked/Unfinished Backends

### cco_hybrid (Blocked)

**Status:** NOT FINISHED - Blocked from use

The CCO (Constrained Constructive Optimization) hybrid backend is not yet complete. Attempting to use this backend will result in an error:

```
CCO backend is not finished; do not use.
```

### NLP Optimization (Blocked)

**Status:** NOT FINISHED - Blocked from use

NLP (Non-Linear Programming) optimization for bifurcation point selection is not yet complete. Attempting to enable `use_nlp_optimization` in backend_params will result in an error:

```
NLP optimization is not finished; do not use.
```

## Summary Table

| Backend | Status | Notes |
|---------|--------|-------|
| scaffold_topdown | Active | Preferred for bifurcating trees |
| space_colonization | Active | Recommended for organic growth |
| programmatic | Active | For DSL-based generation |
| kary_tree | Deprecated | Use scaffold_topdown instead |
| cco_hybrid | Blocked | Not finished |
| NLP optimization | Blocked | Not finished |
