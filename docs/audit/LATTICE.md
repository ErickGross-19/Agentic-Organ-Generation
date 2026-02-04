# LATTICE Scaffold Audit

**Category**: Lattice Structures
**Total Scaffolds**: 6
**Audited**: 6/6 ✅ COMPLETE

## Checklist

- [x] 1. basic ✅
- [x] 2. gyroid ✅
- [x] 3. honeycomb ✅
- [x] 4. octet_truss ✅
- [x] 5. schwarz_p ✅
- [x] 6. voronoi ✅

## Category Statistics

| Scaffold | Backend | Frontend | ParamMeta | Status |
|----------|---------|----------|-----------|--------|
| basic | 20 | 20 | 20 | ✅ Audited (3 stats-only) |
| gyroid | 22 | 22 | 22 | ✅ Audited (7 stats-only) |
| honeycomb | 22 | 22 | 22 | ✅ Audited (5 stats-only) |
| octet_truss | 22 | 22 | 22 | ✅ Audited (6 stats-only) |
| schwarz_p | 23 | 23 | 23 | ✅ Audited (8 stats-only) |
| voronoi | 20 | 20 | 20 | ✅ Audited (3 stats-only) |
| **TOTAL** | **129** | **129** | **129** | 6/6 ✅ |

| Metric | Value | Notes |
|--------|-------|-------|
| Backend Params | 129 | basic: 20, gyroid: 22, honeycomb: 22, octet_truss: 22, schwarz_p: 23, voronoi: 20 |
| Frontend Unique Props | 129 | basic: 20, gyroid: 22, honeycomb: 22, octet_truss: 22, schwarz_p: 23, voronoi: 20 |
| Frontend Legacy Aliases | 25 | basic: 5, gyroid: 5, honeycomb: 6, octet_truss: 3, schwarz_p: 4, voronoi: 2 |
| ParamMeta UI Controls | 129 | basic: 18 (20 effective), gyroid: 22, honeycomb: 22, octet_truss: 22, schwarz_p: 23, voronoi: 20 |
| Dead Code Params | 0 | All params active |
| Stats-Only (FEA) | 32 | basic: 3, gyroid: 7, honeycomb: 5, octet_truss: 6, schwarz_p: 8, voronoi: 3 |
| Bio Verified | 129/129 | All defaults verified |

### Frontend Legacy Alias Mapping

| Scaffold | Alias | Maps To | Backend Status |
|----------|-------|---------|----------------|
| basic | `bounding_box_mm` | `bounding_box_x/y/z_mm` | ✅ (backend legacy) |
| basic | `unit_cell` | `lattice_type` | ✅ (backend legacy) |
| basic | `cell_size_mm` | `unit_cell_size_mm` | ✅ (backend legacy) |
| basic | `bounding_box` (object) | `bounding_box_x/y/z_mm` | ✅ (frontend convenience) |
| gyroid | `bounding_box_mm` | `bounding_box_x/y/z_mm` | ✅ (backend legacy tuple) |
| gyroid | `cell_size_mm` | `unit_cell_size_mm` | ✅ (backend legacy) |
| gyroid | `wall_thickness_mm` | `wall_thickness_um` | ✅ (backend legacy, mm→um) |
| gyroid | `bounding_box_mm` (object) | `bounding_box_x/y/z_mm` | ✅ (frontend convenience) |
| honeycomb | `bounding_box_mm` | `bounding_box_x/y/z_mm` | ✅ (backend legacy tuple) |
| honeycomb | `cell_size_mm` | `cell_inner_length_mm` | ✅ (backend legacy) |
| honeycomb | `height_mm` | `cell_height_mm` | ✅ (backend legacy) |
| honeycomb | `bounding_box_mm` (object) | `bounding_box_x/y/z_mm` | ✅ (frontend convenience) |
| octet_truss | `bounding_box_mm` | `bounding_box_x/y/z_mm` | ✅ (backend legacy tuple) |
| octet_truss | `cell_size_mm` | `unit_cell_edge_mm` | ✅ (backend legacy) |
| octet_truss | `bounding_box_mm` (object) | `bounding_box_x/y/z_mm` | ✅ (frontend convenience) |
| schwarz_p | `bounding_box_mm` | `bounding_box_x/y/z_mm` | ✅ (backend legacy tuple) |
| schwarz_p | `cell_size_mm` | `unit_cell_size_mm` | ✅ (backend legacy) |
| schwarz_p | `wall_thickness_mm` | `wall_thickness_um` | ✅ (backend legacy, mm→μm) |
| schwarz_p | `bounding_box_mm` (object) | `bounding_box_x/y/z_mm` | ✅ (frontend convenience) |
| voronoi | `bounding_box_mm` | `bounding_box_x/y/z_mm` | ✅ (backend legacy tuple) |
| voronoi | `cell_count` | `seed_count` | ✅ (backend legacy) |
| voronoi | `bounding_box_mm` (object) | `bounding_box_x/y/z_mm` | ✅ (frontend convenience) |

---

## Detailed Audits

### 1. basic

**Status**: ✅ AUDITED - FULLY SYNCED

**File Locations**:
- Backend: `backend/app/geometry/lattice/basic.py`
- Frontend Interface: `frontend/lib/types/scaffolds/lattice.ts` (lines 11-49)
- ParamMeta: `frontend/lib/parameterMeta/lattice.ts` (lines 9-71)

**Parameter Count**:
- Backend params: 20 primary + 3 legacy
- Frontend props: 20 unique + 2 legacy + 1 object format
- ParamMeta controls: 18 entries (20 effective via bounding_box composite)

#### Parameter Tracing (Line-by-Line)

| # | Parameter | Backend Line | Type | Default | Active/Stats | Usage Location |
|---|-----------|--------------|------|---------|--------------|----------------|
| 1 | `bounding_box_x_mm` | 449 | float | 10.0 | ✅ ACTIVE | L725-730: bx = params.bounding_box_x_mm |
| 2 | `bounding_box_y_mm` | 450 | float | 10.0 | ✅ ACTIVE | L725-730: by = params.bounding_box_y_mm |
| 3 | `bounding_box_z_mm` | 451 | float | 10.0 | ✅ ACTIVE | L725-730: bz = params.bounding_box_z_mm |
| 4 | `unit_cell_size_mm` | 452 | float | 2.0 | ✅ ACTIVE | L735: cell = ...unit_cell_size_mm |
| 5 | `lattice_type` | 455 | Literal['cubic','bcc','fcc'] | 'cubic' | ✅ ACTIVE | L765-771: dispatches to strut function |
| 6 | `strut_diameter_mm` | 458 | float | 0.5 | ✅ ACTIVE | L736: base_radius = strut_diameter_mm / 2 |
| 7 | `strut_taper` | 459 | float | 0.0 | ✅ ACTIVE | L739, L814: passed to make_strut() |
| 8 | `strut_profile` | 460 | str | 'circular' | ✅ ACTIVE | L740-745, L815: profile selection |
| 9 | `porosity` | 463 | float | 0.7 | ⚡ STATS | L893: target_porosity (actual calculated L879) |
| 10 | `pore_size_um` | 464 | float | 500.0 | ⚡ STATS | L904: reference (actual calculated L882) |
| 11 | `enable_gradient_density` | 467 | bool | False | ✅ ACTIVE | L748, L796: controls gradient calculation |
| 12 | `gradient_axis` | 468 | str | 'z' | ✅ ACTIVE | L749, L797: passed to _calculate_gradient_diameter |
| 13 | `gradient_start_density` | 469 | float | 0.5 | ✅ ACTIVE | L750, L800: gradient interpolation |
| 14 | `gradient_end_density` | 470 | float | 1.0 | ✅ ACTIVE | L751, L801: gradient interpolation |
| 15 | `enable_node_filleting` | 473 | bool | False | ✅ ACTIVE | L756, L828, L857: fillet sphere creation |
| 16 | `fillet_radius_factor` | 474 | float | 0.3 | ✅ ACTIVE | L757, L860: fillet_radius calculation |
| 17 | `enable_node_spheres` | 475 | bool | False | ✅ ACTIVE | L754, L828, L850: node sphere creation |
| 18 | `node_sphere_factor` | 476 | float | 1.1 | ✅ ACTIVE | L755, L851: sphere_radius calculation |
| 19 | `resolution` | 479 | int | 8 | ✅ ACTIVE | L812, L852, L861: mesh resolution |
| 20 | `seed` | 480 | int\|None | None | ⚡ STATS | L918: stats only (lattice is deterministic) |

#### Legacy Aliases

| Alias | Maps To | Backend Line | Notes |
|-------|---------|--------------|-------|
| `bounding_box_mm` | tuple format | 483 | Legacy tuple (x,y,z) |
| `unit_cell` | `lattice_type` | 484 | Literal['cubic','bcc'] |
| `cell_size_mm` | `unit_cell_size_mm` | 485 | Old naming |

**Dead Code Count**: 0 (all 17 active params used in generate_lattice())

#### Stats-Only Parameters

| Parameter | Purpose | Notes |
|-----------|---------|-------|
| `porosity` | Target porosity | Actual calculated from volume (L879) |
| `pore_size_um` | Target pore size | Actual estimated from geometry (L882) |
| `seed` | Random seed | Unused - basic lattice is fully deterministic |

#### Biological/Engineering Accuracy

| Parameter | Default | Range | Literature Source | Verification |
|-----------|---------|-------|-------------------|--------------|
| `unit_cell_size_mm` | 2.0 | 1.5-3.0 mm | Bone scaffold literature | ✅ Cell infiltration optimal |
| `strut_diameter_mm` | 0.5 | 0.3-0.8 mm | Trabecular bone | ✅ Middle of range |
| `porosity` | 0.7 | 60-80% | Trabecular bone density | ✅ 70% target |
| `pore_size_um` | 500.0 | 300-800 μm | Bone ingrowth literature | ✅ Middle of range |
| `fillet_radius_factor` | 0.3 | 0.1-0.5 | Stress concentration reduction | ✅ Reasonable |
| `node_sphere_factor` | 1.1 | 1.0-2.0 | Node reinforcement | ✅ Slightly larger |

**Implementation Notes**:
- Supports 3 lattice types: cubic (12 struts), BCC (20 struts), FCC (24 struts)
- Strut tapering mimics natural bone trabeculae (thicker at nodes)
- 4 strut profiles: circular, square, hexagonal, elliptical
- Density gradient along any axis for functionally graded scaffolds
- Node spheres and fillets reduce stress concentrations at junctions
- Uses `batch_union()` for efficient boolean operations

---

### 2. gyroid

**Status**: ✅ AUDITED - FULLY SYNCED

**File Locations**:
- Backend: `backend/app/geometry/lattice/gyroid.py`
- Frontend Interface: `frontend/lib/types/scaffolds/lattice.ts` (lines 55-93)
- ParamMeta: `frontend/lib/parameterMeta/lattice.ts` (lines 73-102)

**Parameter Count**:
- Backend params: 22 primary + 3 legacy
- Frontend props: 22 unique + 2 legacy + 1 object format
- ParamMeta controls: 22 entries

#### Parameter Tracing (Line-by-Line)

| # | Parameter | Backend Line | Type | Default | Active/Stats | Usage Location |
|---|-----------|--------------|------|---------|--------------|----------------|
| 1 | `bounding_box_x_mm` | 77 | float | 10.0 | ✅ ACTIVE | L383: bx = params.bounding_box_x_mm |
| 2 | `bounding_box_y_mm` | 78 | float | 10.0 | ✅ ACTIVE | L384: by = params.bounding_box_y_mm |
| 3 | `bounding_box_z_mm` | 79 | float | 10.0 | ✅ ACTIVE | L385: bz = params.bounding_box_z_mm |
| 4 | `unit_cell_size_mm` | 80 | float | 1.5 | ✅ ACTIVE | L388: L = ...unit_cell_size_mm |
| 5 | `pore_size_um` | 83 | float | 700.0 | ⚡ STATS | L465: copied to stats |
| 6 | `porosity` | 84 | float | 0.56 | ⚡ STATS | L461: target_porosity (actual estimated L453-454) |
| 7 | `wall_thickness_um` | 85 | float | 250.0 | ⚡ STATS | L466: copied to stats |
| 8 | `isovalue` | 86 | float | 0.0 | ✅ ACTIVE | L417, L425: isosurface extraction level |
| 9 | `elastic_modulus_target_gpa` | 89 | float | 15.0 | ⚡ STATS | L467: FEA reference |
| 10 | `stress_distribution_uniformity` | 90 | float | 0.8 | ⚡ STATS | L473: FEA reference |
| 11 | `yield_strength_target_mpa` | 91 | float | 100.0 | ⚡ STATS | L474: FEA reference |
| 12 | `surface_area_ratio` | 94 | float | 2.5 | ⚡ STATS | L468: reference value |
| 13 | `enable_surface_texture` | 95 | bool | False | ✅ ACTIVE | L437: controls texture application |
| 14 | `texture_amplitude_um` | 96 | float | 10.0 | ✅ ACTIVE | L439: _apply_surface_texture() |
| 15 | `enable_gradient` | 99 | bool | False | ✅ ACTIVE | L410: controls gradient field computation |
| 16 | `gradient_axis` | 100 | str | 'z' | ✅ ACTIVE | L414: passed to _compute_gradient_field |
| 17 | `gradient_start_porosity` | 101 | float | 0.5 | ✅ ACTIVE | L415: gradient interpolation |
| 18 | `gradient_end_porosity` | 102 | float | 0.7 | ✅ ACTIVE | L416: gradient interpolation |
| 19 | `samples_per_cell` | 105 | int | 20 | ✅ ACTIVE | L392: grid resolution calculation |
| 20 | `mesh_density` | 106 | float | 1.0 | ✅ ACTIVE | L392: scales effective_samples |
| 21 | `seed` | 107 | int\|None | None | ✅ ACTIVE | L441: texture randomization |
| 22 | `resolution` | 108 | int | 15 | ✅ ACTIVE | L392: scales mesh quality |

#### Legacy Aliases

| Alias | Maps To | Backend Line | Notes |
|-------|---------|--------------|-------|
| `bounding_box_mm` | tuple format | 111 | Legacy (x,y,z) tuple |
| `cell_size_mm` | `unit_cell_size_mm` | 112 | Old naming |
| `wall_thickness_mm` | `wall_thickness_um` | 113 | mm → um conversion |

**Dead Code Count**: 0 (all 15 active params used in generate_gyroid())

#### Stats-Only Parameters

| Parameter | Purpose | Notes |
|-----------|---------|-------|
| `pore_size_um` | Target pore size | Reference for design intent |
| `porosity` | Target porosity | Actual estimated from isovalue (L453-454) |
| `wall_thickness_um` | Target wall thickness | Reference for design intent |
| `elastic_modulus_target_gpa` | FEA optimization target | Not used in geometry generation |
| `stress_distribution_uniformity` | FEA optimization target | Not used in geometry generation |
| `yield_strength_target_mpa` | FEA optimization target | Not used in geometry generation |
| `surface_area_ratio` | Reference value | Actual calculated from mesh |

#### Biological/Engineering Accuracy

| Parameter | Default | Range | Literature Source | Verification |
|-----------|---------|-------|-------------------|--------------|
| `unit_cell_size_mm` | 1.5 | 1.0-3.0 mm | Taniguchi et al., 2016 | ✅ Optimal for bone |
| `pore_size_um` | 700.0 | 500-800 μm | Osteogenic differentiation | ✅ |
| `porosity` | 0.56 | 0.50-0.62 | Gibson & Ashby, 1997 | ✅ Trabecular range |
| `wall_thickness_um` | 250.0 | 200-300 μm | Structural integrity | ✅ |
| `isovalue` | 0.0 | -1 to 1 | TPMS mathematics | ✅ Neutral porosity |
| `elastic_modulus_target_gpa` | 15.0 | 10-30 GPa | Cortical bone | ✅ |
| `stress_distribution_uniformity` | 0.8 | 0-1 | Design target | ✅ High uniformity |
| `surface_area_ratio` | 2.5 | 1-5 mm²/mm³ | Cell attachment | ✅ |
| `texture_amplitude_um` | 10.0 | 5-20 μm | Osteoblast preference | ✅ |

**Implementation Notes**:
- TPMS implicit function: sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = isovalue
- Uses marching cubes (scikit-image) for isosurface extraction
- Porosity gradient achieved by spatially varying isovalue
- Surface texture via noise-based vertex displacement along normals
- High surface area to volume ratio ideal for cell attachment
- Fully interconnected pore network for nutrient transport

---

### 3. honeycomb

**Status**: ✅ AUDITED - FULLY SYNCED

**File Locations**:
- Backend: `backend/app/geometry/lattice/honeycomb.py`
- Frontend Interface: `frontend/lib/types/scaffolds/lattice.ts` (lines 214-252)
- ParamMeta: `frontend/lib/parameterMeta/lattice.ts` (lines 198-227)

**Parameter Count**:
- Backend params: 22 primary + 3 legacy
- Frontend props: 22 unique + 2 legacy + 1 object format
- ParamMeta controls: 22 entries

#### Parameter Tracing (Line-by-Line)

| # | Parameter | Backend Line | Type | Default | Active/Stats | Usage Location |
|---|-----------|--------------|------|---------|--------------|----------------|
| 1 | `bounding_box_x_mm` | 80 | float | 10.0 | ✅ ACTIVE | L600: bx |
| 2 | `bounding_box_y_mm` | 81 | float | 10.0 | ✅ ACTIVE | L601: by |
| 3 | `bounding_box_z_mm` | 82 | float | 5.0 | ✅ ACTIVE | L602: bz, default height |
| 4 | `cell_height_mm` | 83 | float\|None | None | ✅ ACTIVE | L607-608: extrusion height |
| 5 | `wall_thickness_mm` | 86 | float | 1.0 | ✅ ACTIVE | L631: wall creation |
| 6 | `cell_inner_length_mm` | 87 | float | 3.0 | ✅ ACTIVE | L613: cell_size |
| 7 | `cell_angle_deg` | 88 | float | 120.0 | ⚡ STATS | L701: reference only (always 120°) |
| 8 | `cell_orientation` | 89 | str | 'flat_top' | ✅ ACTIVE | L616-622: orientation parsing |
| 9 | `porosity` | 92 | float | 0.85 | ⚡ STATS | L694: target (actual calc L666) |
| 10 | `num_cells_x` | 93 | int\|None | None | ✅ ACTIVE | L634, L679: cell count override |
| 11 | `num_cells_y` | 94 | int\|None | None | ✅ ACTIVE | L635, L680: cell count override |
| 12 | `enable_gradient_wall_thickness` | 97 | bool | False | ✅ ACTIVE | L636: gradient control |
| 13 | `wall_thickness_start_mm` | 98 | float | 0.8 | ✅ ACTIVE | L637: gradient start |
| 14 | `wall_thickness_end_mm` | 99 | float | 1.5 | ✅ ACTIVE | L638: gradient end |
| 15 | `gradient_axis` | 100 | str | 'x' | ✅ ACTIVE | L639: gradient direction |
| 16 | `enable_wall_texture` | 103 | bool | False | ⚡ STATS | L717-718: manufacturing reference |
| 17 | `texture_depth_um` | 104 | float | 20.0 | ⚡ STATS | L719: reference only |
| 18 | `enable_wall_perforations` | 105 | bool | False | ✅ ACTIVE | L640, L653-655: perforation creation |
| 19 | `perforation_diameter_um` | 106 | float | 200.0 | ✅ ACTIVE | L625: converted to mm |
| 20 | `perforation_spacing_mm` | 107 | float | 0.5 | ✅ ACTIVE | L642: spacing parameter |
| 21 | `resolution` | 110 | int | 1 | ✅ ACTIVE | L643: mesh resolution |
| 22 | `seed` | 111 | int\|None | None | ⚡ STATS | L723: reference (deterministic) |

#### Legacy Aliases

| Alias | Maps To | Backend Line | Notes |
|-------|---------|--------------|-------|
| `bounding_box_mm` | tuple format | 114 | Legacy (x,y,z) tuple |
| `cell_size_mm` | `cell_inner_length_mm` | 115 | Old naming |
| `height_mm` | `cell_height_mm` | 116 | Old naming |

**Dead Code Count**: 0 (all 17 active params used in generate_honeycomb())

#### Stats-Only Parameters

| Parameter | Purpose | Notes |
|-----------|---------|-------|
| `cell_angle_deg` | Reference value | Always 120° for regular hexagon |
| `porosity` | Target porosity | Actual calculated from volume (L666) |
| `enable_wall_texture` | Manufacturing reference | Not geometrically modeled (L717) |
| `texture_depth_um` | Manufacturing reference | For post-processing specification |
| `seed` | Random seed | Unused - honeycomb is deterministic |

#### Biological/Engineering Accuracy

| Parameter | Default | Range | Literature Source | Verification |
|-----------|---------|-------|-------------------|--------------|
| `wall_thickness_mm` | 1.0 | 0.5-2.0 mm | Structural integrity | ✅ |
| `cell_inner_length_mm` | 3.0 | 2-5 mm | Cell infiltration | ✅ |
| `cell_angle_deg` | 120.0 | 120° | Regular hexagon | ✅ |
| `porosity` | 0.85 | 0.80-0.90 | High permeability | ✅ |
| `perforation_diameter_um` | 200.0 | 50-500 μm | Cell migration | ✅ |
| `texture_depth_um` | 20.0 | 5-50 μm | Cell attachment | ✅ |

**Implementation Notes**:
- Supports flat-top and pointy-top hexagon orientations
- Wall segments created as rectangular prisms, deduplicated by position
- Gradient wall thickness along any axis via stacked/interpolated segments
- Wall perforations via cylinder subtraction for nutrient transport
- Uses `batch_union()` for efficient wall combination
- Clips to bounding box for clean edges

---

### 4. octet_truss

**Status**: ✅ AUDITED - FULLY SYNCED

**File Locations**:
- Backend: `backend/app/geometry/lattice/octet_truss.py`
- Frontend Interface: `frontend/lib/types/scaffolds/lattice.ts` (lines 138-176)
- ParamMeta: `frontend/lib/parameterMeta/lattice.ts` (lines 137-167)

**Parameter Count**:
- Backend params: 22 primary + 2 legacy
- Frontend props: 22 unique + 1 legacy + 1 object format
- ParamMeta controls: 22 entries

#### Parameter Tracing (Line-by-Line)

| # | Parameter | Backend Line | Type | Default | Active/Stats | Usage Location |
|---|-----------|--------------|------|---------|--------------|----------------|
| 1 | `bounding_box_x_mm` | 84 | float | 10.0 | ✅ ACTIVE | L476: bx |
| 2 | `bounding_box_y_mm` | 85 | float | 10.0 | ✅ ACTIVE | L477: by |
| 3 | `bounding_box_z_mm` | 86 | float | 10.0 | ✅ ACTIVE | L478: bz |
| 4 | `unit_cell_edge_mm` | 87 | float | 7.5 | ✅ ACTIVE | L482: cell size |
| 5 | `strut_diameter_mm` | 90 | float | 1.2 | ✅ ACTIVE | L483: base_radius |
| 6 | `strut_taper` | 91 | float | 0.0 | ✅ ACTIVE | L512, L542: taper control |
| 7 | `strut_surface_roughness` | 92 | float | 0.0 | ✅ ACTIVE | L513, L544: roughness |
| 8 | `relative_density` | 95 | float | 0.48 | ⚡ STATS | L638: target (actual calc L625) |
| 9 | `pore_size_um` | 96 | float | 500.0 | ⚡ STATS | L647: reference |
| 10 | `d_over_pore_ratio` | 97 | float | 2.4 | ⚡ STATS | L649: reference |
| 11 | `normalized_modulus` | 100 | float | 0.25 | ⚡ STATS | L650: FEA reference |
| 12 | `yield_strength_target_mpa` | 101 | float | 50.0 | ⚡ STATS | L651: FEA reference |
| 13 | `elastic_modulus_target_gpa` | 102 | float | 10.0 | ⚡ STATS | L652: FEA reference |
| 14 | `node_fillet_radius_mm` | 105 | float | 0.0 | ✅ ACTIVE | L585, L588: fillet creation |
| 15 | `enable_node_spheres` | 106 | bool | False | ✅ ACTIVE | L558: sphere creation |
| 16 | `node_sphere_factor` | 107 | float | 1.2 | ✅ ACTIVE | L559: sphere radius |
| 17 | `enable_gradient` | 110 | bool | False | ✅ ACTIVE | L514: gradient control |
| 18 | `gradient_axis` | 111 | str | 'z' | ✅ ACTIVE | L529: gradient direction |
| 19 | `gradient_start_density` | 112 | float | 0.3 | ✅ ACTIVE | L530: gradient start |
| 20 | `gradient_end_density` | 113 | float | 0.6 | ✅ ACTIVE | L531: gradient end |
| 21 | `resolution` | 116 | int | 8 | ✅ ACTIVE | L543, L579, L607: mesh quality |
| 22 | `seed` | 117 | int\|None | None | ✅ ACTIVE | L486: rng initialization |

#### Legacy Aliases

| Alias | Maps To | Backend Line | Notes |
|-------|---------|--------------|-------|
| `bounding_box_mm` | tuple format | 120 | Legacy (x,y,z) tuple |
| `cell_size_mm` | `unit_cell_edge_mm` | 121 | Old naming |

**Dead Code Count**: 0 (all 16 active params used in generate_octet_truss())

#### Stats-Only Parameters

| Parameter | Purpose | Notes |
|-----------|---------|-------|
| `relative_density` | Target density | Actual calculated from volume (L625) |
| `pore_size_um` | Target pore size | Reference for design intent |
| `d_over_pore_ratio` | Design ratio | Strut diameter to pore size |
| `normalized_modulus` | FEA target | E/E_s ratio for optimization |
| `yield_strength_target_mpa` | FEA target | Not used in geometry |
| `elastic_modulus_target_gpa` | FEA target | Not used in geometry |

#### Biological/Engineering Accuracy

| Parameter | Default | Range | Literature Source | Verification |
|-----------|---------|-------|-------------------|--------------|
| `unit_cell_edge_mm` | 7.5 | 5-10 mm | Pore interconnectivity | ✅ |
| `strut_diameter_mm` | 1.2 | 0.8-1.5 mm | Taniguchi et al., 2016 | ✅ |
| `relative_density` | 0.48 | 0.3-0.6 | Bone-mimicking | ✅ Mid-range |
| `pore_size_um` | 500.0 | 200-800 μm | Bone ingrowth | ✅ |
| `normalized_modulus` | 0.25 | 0.1-0.5 | E/E_s ratio | ✅ |
| `node_sphere_factor` | 1.2 | 1.0-2.0 | Reinforcement | ✅ |

**Implementation Notes**:
- FCC-based structure: 8 corners + 6 face centers per unit cell
- 12 edge struts + 24 face diagonal struts + 12 internal octahedron struts
- Strut tapering mimics natural bone (thicker at nodes)
- Surface roughness via random variation in strut radius
- Node fillets approximated using small spheres for smooth stress distribution
- Density gradient scales strut radius by sqrt(density_factor/0.5)
- Uses `batch_union()` for efficient combination of all parts

---

### 5. schwarz_p

**Status**: ✅ AUDITED - FULLY SYNCED

**File Locations**:
- Backend: `backend/app/geometry/lattice/schwarz_p.py`
- Frontend Interface: `frontend/lib/types/scaffolds/lattice.ts` (lines 95-136)
- ParamMeta: `frontend/lib/parameterMeta/lattice.ts` (lines 104-135)

**Parameter Count**:
- Backend params: 23 primary + 3 legacy
- Frontend props: 23 unique + 2 deprecated + 1 object format
- ParamMeta controls: 23 entries

#### Parameter Tracing (Line-by-Line)

| # | Parameter | Backend Line | Type | Default | Active/Stats | Usage Location |
|---|-----------|--------------|------|---------|--------------|----------------|
| 1 | `bounding_box_x_mm` | 82 | float | 10.0 | ✅ ACTIVE | L313: bx |
| 2 | `bounding_box_y_mm` | 83 | float | 10.0 | ✅ ACTIVE | L314: by |
| 3 | `bounding_box_z_mm` | 84 | float | 10.0 | ✅ ACTIVE | L315: bz |
| 4 | `unit_cell_size_mm` | 85 | float | 2.0 | ✅ ACTIVE | L318: period L |
| 5 | `porosity` | 88 | float | 0.75 | ⚡ STATS | L437: target (actual estimated L419-420) |
| 6 | `small_pore_size_um` | 89 | float | 200.0 | ⚡ STATS | L441: bi-modal pore design reference |
| 7 | `large_pore_size_um` | 90 | float | 400.0 | ⚡ STATS | L442: vascularization reference |
| 8 | `wall_thickness_um` | 91 | float | 200.0 | ⚡ STATS | L443: structural reference |
| 9 | `isovalue` | 92 | float | 0.0 | ✅ ACTIVE | L410, L414: surface level |
| 10 | `k_parameter` | 95 | float | 1.0 | ✅ ACTIVE | L344: wave number modifier |
| 11 | `s_parameter` | 96 | float | 1.0 | ✅ ACTIVE | L344: surface stretching |
| 12 | `fluid_permeability_target` | 99 | float | 1e-9 | ⚡ STATS | L446: design reference |
| 13 | `diffusion_coefficient_ratio` | 100 | float | 0.3 | ✅ ACTIVE | L350-352: isovalue offset |
| 14 | `mechanical_stability` | 103 | float | 0.7 | ⚡ STATS | L447: design reference |
| 15 | `elastic_modulus_target_gpa` | 104 | float | 5.0 | ⚡ STATS | L356, L456-457: Gibson-Ashby |
| 16 | `enable_gradient` | 107 | bool | False | ✅ ACTIVE | L359: gradient branch |
| 17 | `gradient_axis` | 108 | str | 'z' | ✅ ACTIVE | L373-385: axis selection |
| 18 | `gradient_start_porosity` | 109 | float | 0.6 | ✅ ACTIVE | L362, L365: start isovalue |
| 19 | `gradient_end_porosity` | 110 | float | 0.85 | ✅ ACTIVE | L363, L366: end isovalue |
| 20 | `samples_per_cell` | 113 | int | 20 | ✅ ACTIVE | L326: grid resolution |
| 21 | `mesh_density` | 114 | float | 1.0 | ✅ ACTIVE | L322: resolution scale |
| 22 | `seed` | 115 | int\|None | None | ⚡ STATS | L462: reserved for future |
| 23 | `resolution` | 116 | int | 15 | ✅ ACTIVE | L327: mesh quality |

#### Legacy Aliases

| Alias | Maps To | Backend Line | Notes |
|-------|---------|--------------|-------|
| `bounding_box_mm` | tuple format | 119 | Legacy (x,y,z) tuple |
| `cell_size_mm` | `unit_cell_size_mm` | 120 | Old naming |
| `wall_thickness_mm` | `wall_thickness_um` | 121 | mm → μm conversion |

**Dead Code Count**: 0 (all 15 active params used in generate_schwarz_p())

#### Stats-Only Parameters

| Parameter | Purpose | Notes |
|-----------|---------|-------|
| `porosity` | Target porosity | Actual estimated from isovalue (L419-420) |
| `small_pore_size_um` | Bi-modal pore design | 100-200 μm for cell attachment |
| `large_pore_size_um` | Vascularization reference | 300-500 μm for vessel formation |
| `wall_thickness_um` | Structural reference | Design intent specification |
| `fluid_permeability_target` | Design target | Darcy permeability specification |
| `mechanical_stability` | Design target | Stability factor (0-1) |
| `elastic_modulus_target_gpa` | Gibson-Ashby input | Computes suggested porosity |
| `seed` | Reserved | For future stochastic features |

#### Biological/Engineering Accuracy

| Parameter | Default | Range | Literature Source | Verification |
|-----------|---------|-------|-------------------|--------------|
| `unit_cell_size_mm` | 2.0 | 1.0-3.0 mm | Taniguchi et al., 2016 | ✅ Cell infiltration |
| `porosity` | 0.75 | 70-80% | Karageorgiou & Kaplan, 2005 | ✅ Nutrient transport |
| `small_pore_size_um` | 200.0 | 100-200 μm | Bi-modal pore distribution | ✅ Cell attachment |
| `large_pore_size_um` | 400.0 | 300-500 μm | Vascularization literature | ✅ Vessel formation |
| `wall_thickness_um` | 200.0 | 150-250 μm | Structural integrity | ✅ Middle of range |
| `isovalue` | 0.0 | -3 to +3 | TPMS mathematics | ✅ Neutral porosity |
| `k_parameter` | 1.0 | 0.5-2.0 | Wave number | ✅ Standard |
| `s_parameter` | 1.0 | 0.5-2.0 | Surface stretching | ✅ Standard |
| `fluid_permeability_target` | 1e-9 | 10⁻¹⁰-10⁻⁸ m² | Scaffold permeability | ✅ Within range |
| `diffusion_coefficient_ratio` | 0.3 | 0.1-0.5 | Effective/bulk ratio | ✅ Within range |
| `elastic_modulus_target_gpa` | 5.0 | 1-10 GPa | Cartilage-like scaffolds | ✅ Appropriate |

**Implementation Notes**:
- TPMS implicit function: cos(2πx/L) + cos(2πy/L) + cos(2πz/L) = isovalue
- Uses marching cubes (scikit-image) for isosurface extraction
- Gradient porosity via spatially-varying isovalue field
- Diffusion coefficient ratio modulates isovalue offset for pore connectivity
- Gibson-Ashby relationship: E = E_s × (1-φ)² for porosity suggestion
- Simpler topology than gyroid with higher orthogonal connectivity
- Self-supporting for 3D printing

---

### 6. voronoi

**Status**: ✅ AUDITED - FULLY SYNCED

**File Locations**:
- Backend: `backend/app/geometry/lattice/voronoi.py`
- Frontend Interface: `frontend/lib/types/scaffolds/lattice.ts` (lines 178-212)
- ParamMeta: `frontend/lib/parameterMeta/lattice.ts` (lines 169-196)

**Parameter Count**:
- Backend params: 20 primary + 2 legacy
- Frontend props: 20 unique + 1 deprecated + 1 object format
- ParamMeta controls: 20 entries

#### Parameter Tracing (Line-by-Line)

| # | Parameter | Backend Line | Type | Default | Active/Stats | Usage Location |
|---|-----------|--------------|------|---------|--------------|----------------|
| 1 | `bounding_box_x_mm` | 83 | float | 10.0 | ✅ ACTIVE | L439: bx |
| 2 | `bounding_box_y_mm` | 84 | float | 10.0 | ✅ ACTIVE | L440: by |
| 3 | `bounding_box_z_mm` | 85 | float | 10.0 | ✅ ACTIVE | L441: bz |
| 4 | `target_pore_radius_mm` | 88 | float | 0.6 | ✅ ACTIVE | L456: auto seed_count calc |
| 5 | `target_porosity` | 89 | float | 0.70 | ⚡ STATS | L545: target (actual calc L533) |
| 6 | `seed_count` | 90 | int\|None | None | ✅ ACTIVE | L449-450: Voronoi cells |
| 7 | `strut_diameter_mm` | 93 | float | 0.3 | ✅ ACTIVE | L443: strut radius |
| 8 | `strut_taper` | 94 | float | 0.0 | ✅ ACTIVE | L507: make_strut() |
| 9 | `enable_strut_roughness` | 95 | bool | False | ✅ ACTIVE | L508: roughness toggle |
| 10 | `strut_roughness_amplitude` | 96 | float | 0.02 | ✅ ACTIVE | L509: roughness magnitude |
| 11 | `random_coefficient` | 99 | float | 0.5 | ⚡ STATS | L551: reference value |
| 12 | `irregularity` | 100 | float | 0.5 | ⚡ STATS | L552: reference value |
| 13 | `seed` | 101 | int\|None | None | ✅ ACTIVE | L446: RNG init |
| 14 | `enable_gradient` | 104 | bool | False | ✅ ACTIVE | L490: gradient control |
| 15 | `gradient_direction` | 105 | str | 'z' | ✅ ACTIVE | L493: gradient axis |
| 16 | `density_gradient_start` | 106 | float | 0.5 | ✅ ACTIVE | L494: gradient start |
| 17 | `density_gradient_end` | 107 | float | 0.9 | ✅ ACTIVE | L495: gradient end |
| 18 | `resolution` | 110 | int | 8 | ✅ ACTIVE | L506: cylinder segments |
| 19 | `margin_factor` | 111 | float | 0.2 | ✅ ACTIVE | L463: seed margin |
| 20 | `min_strut_length_mm` | 112 | float | 0.1 | ✅ ACTIVE | L475: strut filter |

#### Legacy Aliases

| Alias | Maps To | Backend Line | Notes |
|-------|---------|--------------|-------|
| `bounding_box_mm` | tuple format | 115 | Legacy (x,y,z) tuple |
| `cell_count` | `seed_count` | 116 | Old naming |

**Dead Code Count**: 0 (all 17 active params used in generate_voronoi())

#### Stats-Only Parameters

| Parameter | Purpose | Notes |
|-----------|---------|-------|
| `target_porosity` | Target porosity | Actual calculated from volume (L533) |
| `random_coefficient` | Design reference | Level of randomness (0-1) |
| `irregularity` | Design reference | Shape irregularity (0-1) |

#### Biological/Engineering Accuracy

| Parameter | Default | Range | Literature Source | Verification |
|-----------|---------|-------|-------------------|--------------|
| `target_pore_radius_mm` | 0.6 | 0.3-0.9 mm | Bone scaffold literature | ✅ Middle of range |
| `target_porosity` | 0.70 | 60-80% | Trabecular bone | ✅ Within range |
| `strut_diameter_mm` | 0.3 | 0.2-0.5 mm | Strength/porosity balance | ✅ Within range |
| `strut_roughness_amplitude` | 0.02 | 5-50 μm | Cell attachment | ✅ 20 μm |
| `random_coefficient` | 0.5 | 0-1 | Balance | ✅ Middle |
| `irregularity` | 0.5 | 0-1 | Balance | ✅ Middle |
| `density_gradient_start` | 0.5 | 0.3-0.8 | Gradient range | ✅ |
| `density_gradient_end` | 0.9 | 0.5-0.95 | Gradient range | ✅ |
| `min_strut_length_mm` | 0.1 | 0.05-0.3 mm | Filter threshold | ✅ |

**Implementation Notes**:
- Uses scipy.spatial.Voronoi for 3D tessellation
- Mirror points added for clean boundary handling
- Randomized but reproducible via seed parameter
- Mimics natural trabecular bone architecture
- Strut tapering via segmented frustum construction
- Surface roughness via random radius variation
- Density gradient scales strut radius by position
- Uses `batch_union()` for efficient boolean operations
- Clips final result to bounding box

---
