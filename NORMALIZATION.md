# DesignSpec Unit Normalization

This document explains how unit conversion works in the DesignSpec pipeline, which keys/paths are considered unitful, how defaults are applied, and how to interpret the audit report.

## Overview

All geometric values in DesignSpec are normalized to **meters** internally. Input units are specified via `meta.input_units` and converted on load. This ensures consistent behavior regardless of the input unit system.

## Supported Input Units

| Unit | Scale Factor | Example |
|------|-------------|---------|
| `m` (meters) | 1.0 | 0.005m stays 0.005m |
| `cm` (centimeters) | 0.01 | 0.5cm becomes 0.005m |
| `mm` (millimeters) | 0.001 | 5mm becomes 0.005m |
| `um` (micrometers) | 0.000001 | 5000um becomes 0.005m |

## Dimensional Scaling

Different types of values scale differently based on their dimensionality:

| Type | Scale | Example |
|------|-------|---------|
| Length (radius, pitch, clearance) | scale | 5mm → 0.005m |
| Area | scale² | 25mm² → 2.5e-5m² |
| Volume (min_component_volume) | scale³ | 1mm³ → 1e-9m³ |

## Unitful Fields

### Policy Length Fields

The following fields in policies are treated as length values and scaled by `scale`:

**resolution**: `min_channel_diameter`, `min_pitch`, `max_pitch`

**pathfinding**: `clearance`, `local_radius`, `pitch_coarse`, `pitch_fine`, `corridor_radius_buffer`

**ports**: `ridge_width`, `ridge_clearance`, `port_margin`

**channels**: `length`, `start_offset`, `stop_before_boundary`, `radius_end`, `hook_depth`, `ring_sigma`

**embedding**: `voxel_pitch`, `shell_thickness`, `carve_depth`

**validity**: `min_diameter_threshold`

**open_port**: `probe_length`, `local_region_size`, `validation_pitch`

**repair**: `voxel_pitch`

**growth**: `min_segment_length`, `max_segment_length`, `step_size`, `min_radius`

**collision**: `collision_clearance`

**unified_collision**: `min_clearance`, `min_radius`

**composition**: `repair_voxel_pitch`

**mesh_synthesis**: `voxel_repair_pitch`, `radius_clamp_min`, `radius_clamp_max`

**mesh_merge**: `voxel_pitch`

**network_cleanup**: `snap_tol`, `min_segment_length`, `merge_tol`

**radius**: `min_radius`, `max_radius`

### Policy Volume Fields

The following fields are treated as volume values and scaled by `scale³`:

**repair**: `min_component_volume`

**composition**: `min_component_volume`

**mesh_merge**: `min_component_volume`

### Nested Policy Fields

Nested policies within `composition` are recursively normalized:

- `composition.synthesis_policy` → normalized as `mesh_synthesis`
- `composition.merge_policy` → normalized as `mesh_merge`
- `composition.repair_policy` → normalized as `repair`

### Domain Length Fields

All domain geometric fields are normalized: `center`, `radii`, `radius`, `height`, `min_corner`, `max_corner`, `size`, `start`, `end`, `top_radius`, `bottom_radius`, `translation`, `x_min`, `x_max`, `y_min`, `y_max`, `z_min`, `z_max`, `length`, `radius_top`, `radius_bottom`, `semi_axis_a`, `semi_axis_b`, `semi_axis_c`

### Port Length Fields

Port fields `position` and `radius` are normalized.

### Backend Params

Length fields in `growth.backend_params` and `component.build.backend_params` are normalized: `step_size`, `min_segment_length`, `max_segment_length`, `influence_radius`, `kill_radius`, `perception_radius`, `clearance`, `min_radius`, `max_radius`, `wall_margin_m`, `terminal_radius`, `collision_clearance`, `min_terminal_separation`

## Default Policy Filling

When a DesignSpec is loaded, missing policies are automatically filled with defaults. This ensures no required runtime policy is `None` or missing required keys.

### Rules

1. If a policy object is `null`, it is replaced with the default policy object for that policy type
2. If a policy object exists but is missing keys, missing keys are filled from defaults (explicitly set values are not overwritten)
3. If `use_resolution_policy=true` and pitch is `null`, that is OK; otherwise pitch must be non-null and > 0

### Disabling Default Filling

To disable default filling, pass `fill_defaults=False` to `DesignSpec.from_dict()` or `DesignSpec.from_json()`:

```python
spec = DesignSpec.from_dict(spec_dict, fill_defaults=False)
```

## Preflight Validation

Preflight validation runs automatically when loading a DesignSpec and catches issues before pipeline execution.

### Error Codes

| Code | Description |
|------|-------------|
| `MISSING_REQUIRED_POLICY` | A policy required for a stage is null |
| `PITCH_NOT_NORMALIZED` | A voxel_pitch value appears to be in mm but was not normalized |
| `PITCH_TOO_LARGE` | Voxel pitch is too large relative to domain scale |
| `NO_DOMAINS` | No domains defined in spec |
| `INVALID_DOMAIN_REF` | Component references a domain that doesn't exist |
| `DOMAIN_NOT_NORMALIZED` | Domain dimensions appear unnormalized |
| `PORT_RADIUS_NOT_NORMALIZED` | Port radius appears unnormalized |
| `VOLUME_NOT_NORMALIZED` | Volume field appears unnormalized |

### Strict Mode

To make preflight errors raise exceptions instead of warnings:

```python
spec = DesignSpec.from_dict(spec_dict, preflight_strict=True)
```

### Disabling Preflight

To disable preflight validation:

```python
spec = DesignSpec.from_dict(spec_dict, run_preflight=False)
```

## Unit Audit Report

In debug mode, a detailed audit report is generated showing all unit conversions performed.

### Enabling Debug Mode

Set the environment variable `AOG_NORMALIZATION_DEBUG=1`:

```bash
export AOG_NORMALIZATION_DEBUG=1
python my_script.py
```

### Accessing the Audit Report

```python
spec = DesignSpec.from_dict(spec_dict)
if spec.unit_audit_report:
    print(f"Converted {len(spec.unit_audit_report.entries)} fields")
    for entry in spec.unit_audit_report.entries:
        print(f"  {entry.path}: {entry.original_value} -> {entry.normalized_value} ({entry.scale_type})")
```

### Saving the Audit Report

```python
if spec.unit_audit_report:
    spec.unit_audit_report.save("output/unit_audit_report.json")
```

### Report Format

The audit report JSON contains:

```json
{
  "input_units": "mm",
  "scale_factor": 0.001,
  "total_conversions": 42,
  "entries": [
    {
      "path": "domains.cylinder_domain.radius",
      "field_name": "radius",
      "original_value": 5.0,
      "normalized_value": 0.005,
      "scale_type": "length",
      "scale_factor": 0.001
    }
  ]
}
```

## Example: Malaria Venule Spec

A malaria venule spec with `input_units: "mm"` and the following values:

```json
{
  "meta": {"input_units": "mm"},
  "domains": {
    "cylinder_domain": {"radius": 5.0, "height": 2.0}
  },
  "policies": {
    "composition": {
      "merge_policy": {"voxel_pitch": 0.05, "min_component_volume": 0.001}
    }
  }
}
```

After normalization:

- `domains.cylinder_domain.radius`: 5.0mm → 0.005m
- `domains.cylinder_domain.height`: 2.0mm → 0.002m
- `policies.composition.merge_policy.voxel_pitch`: 0.05mm → 5e-5m
- `policies.composition.merge_policy.min_component_volume`: 0.001mm³ → 1e-12m³

## Troubleshooting

### "merge_policy.voxel_pitch appears to be in mm but was not normalized"

This error indicates that a voxel_pitch value is too large (> 10mm). Check that:
1. `meta.input_units` is set correctly
2. The spec is being loaded via `DesignSpec.from_dict()` or `DesignSpec.from_json()`

### "domain radius is X m but voxel_pitch is Y m (ratio too large)"

This warning indicates that the voxel pitch is more than 1/10 of the domain size, which may cause degenerate meshes. Consider reducing the voxel pitch.

### "policy <name> is null but required for stage <stage>"

This error indicates a required policy is missing. Either:
1. Add the policy to your spec
2. Enable default filling with `fill_defaults=True` (default)

### Degenerate union meshes (8 faces / 6 vertices)

This typically indicates unnormalized voxel pitches. A voxel pitch of 0.05 (interpreted as 50mm instead of 0.05mm) would create a single voxel covering the entire domain. Ensure:
1. `meta.input_units` is set to `"mm"` if values are in millimeters
2. Nested policies like `composition.merge_policy.voxel_pitch` are being normalized

### Empty domain_with_void.stl

This can occur when:
1. Voxel pitch is too large relative to domain
2. The void mesh doesn't intersect the domain
3. Preflight validation is disabled and errors are not caught

Enable preflight validation and check for warnings about pitch ratios.
