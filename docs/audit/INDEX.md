# MorphoStruct Scaffold Audit Tracker

**Started**: 2026-01-30
**Total Scaffolds**: 39 + 27 primitives

## Progress Summary

| Status | Count |
|--------|-------|
| Audited | 39/39 |
| Pending | 0/39 |

## Category Progress

| Category | Audited | Total | File | Status |
|----------|---------|-------|------|--------|
| [TUBULAR](./TUBULAR.md) | 7/7 | 7 | `TUBULAR.md` | ✅ COMPLETE |
| [ORGAN](./ORGAN.md) | 6/6 | 6 | `ORGAN.md` | ✅ COMPLETE |
| [SKELETAL](./SKELETAL.md) | 7/7 | 7 | `SKELETAL.md` | ✅ COMPLETE |
| [SOFT_TISSUE](./SOFT_TISSUE.md) | 4/4 | 4 | `SOFT_TISSUE.md` | ✅ COMPLETE |
| [MICROFLUIDIC](./MICROFLUIDIC.md) | 3/3 | 3 | `MICROFLUIDIC.md` | ✅ COMPLETE |
| [DENTAL](./DENTAL.md) | 3/3 | 3 | `DENTAL.md` | ✅ COMPLETE |
| [LATTICE](./LATTICE.md) | 6/6 | 6 | `LATTICE.md` | ✅ COMPLETE |
| [LEGACY](./LEGACY.md) | 3/3 | 3 | `LEGACY.md` | ✅ COMPLETE |
| [PRIMITIVES](./PRIMITIVES.md) | 27/27 | 27 | `PRIMITIVES.md` | ✅ NEW |

## Aggregate Statistics (All 8 Categories Complete)

| Metric | Total | Per-Category Breakdown |
|--------|-------|------------------------|
| Backend Params | 1299 | TUBULAR: 266, ORGAN: 216, SKELETAL: 306, SOFT_TISSUE: 139, MICROFLUIDIC: 101, DENTAL: 117, LEGACY: 25, LATTICE: 129 |
| Frontend Unique Props | 1300 | Backend + 1 frontend-only (porous_disc.porosity_target) |
| Frontend Legacy Aliases | 70 | TUBULAR: 11, ORGAN: 11, SKELETAL: 7, SOFT_TISSUE: 0, MICROFLUIDIC: 13, DENTAL: 0, LEGACY: 3, LATTICE: 25 |
| Frontend Total Props | 1370 | 1300 unique + 70 legacy aliases |
| ParamMeta UI Controls | 1299 | All backend params have UI controls |
| Dead Code Params | 0 | All fixed |
| Stats-Only (FEA) | 85 | TUBULAR: 3, ORGAN: 3, SKELETAL: 14, SOFT_TISSUE: 8, MICROFLUIDIC: 13, DENTAL: 12, LEGACY: 0, LATTICE: 32 |
| Alive (Used) Params | 1214 | 1299 - 85 FEA/reference specs |
| Bio Verified | 1299 | All defaults verified |
| Bio Unverified | 0 | |
| New Primitives | 27 | 4 basic + 8 geometric + 7 architectural + 8 organic |

### Frontend Legacy Alias Mapping
| Scaffold | Alias | Maps To | Backend Status |
|----------|-------|---------|----------------|
| bladder | `diameter_mm` | `dome_diameter_mm` | ✅ |
| bladder | `wall_thickness_mm` | `wall_thickness_empty_mm` | ✅ |
| nerve_conduit | `channel_count` | `num_channels` | ✅ |
| spinal_cord | `cord_diameter_mm` | `total_diameter_mm` | ✅ |
| spinal_cord | `gray_matter_ratio` | `gray_matter_volume_ratio` | ✅ |
| spinal_cord | `channel_count` | `num_guidance_channels` | ✅ |
| trachea | `length_mm` | `total_length_mm` | ✅ (FIXED) |
| trachea | `ring_count` | `num_cartilage_rings` | ✅ |
| trachea | `posterior_gap_angle_deg` | `posterior_opening_degrees` | ✅ |
| trachea | `ring_spacing_mm` | `interring_spacing_mm` | ✅ |
| trachea | `ligament_thickness_mm` | `annular_ligament_thickness_mm` | ✅ (FIXED) |
| cardiac_patch | `patch_size` | `patch_length/width/thickness` | ✅ |
| cardiac_patch | `alignment_angle` | `epicardial_helix_angle` | ✅ |
| kidney_tubule | `tubule_diameter` | `tubule_outer_diameter` | ✅ |
| kidney_tubule | `length` | `tubule_length` | ✅ |
| kidney_tubule | `wall_thickness` | calculated from outer-inner | ✅ |
| liver_sinusoid | `length` | `sinusoid_length` | ✅ |
| liver_sinusoid | `fenestration_size` | `fenestration_diameter` | ✅ |
| lung_alveoli | `alveoli_diameter` | `alveolar_diameter` | ✅ |
| lung_alveoli | `branch_generations` | `scaffold_generations` | ✅ |
| pancreatic_islet | `cluster_count` | `islet_count` | ✅ |
| pancreatic_islet | `spacing` | `islet_spacing` | ✅ |
| gradient_scaffold | `start_porosity` | `starting_porosity` | ✅ |
| gradient_scaffold | `end_porosity` | `ending_porosity` | ✅ |
| gradient_scaffold | `dimensions_mm` | `bounding_box_mm` | ✅ |
| organ_on_chip | `channel_width_mm` | `main_channel_width_um` | ✅ |
| organ_on_chip | `channel_depth_mm` | `main_channel_height_um` | ✅ |
| organ_on_chip | `chamber_size_mm` | `chamber_width/length/height` | ✅ |
| organ_on_chip | `chamber_count` | `num_chambers` | ✅ |
| organ_on_chip | `inlet_count` | `num_inlets` | ✅ |
| organ_on_chip | `chip_size_mm` | `chip_length/width/thickness` | ✅ |
| perfusable_network | `inlet_diameter_mm` | `large_vessel_diameter_mm` | ✅ |
| perfusable_network | `branch_generations` | `num_branching_generations` | ✅ |
| perfusable_network | `murray_ratio` | `murrays_law_ratio` | ✅ |
| perfusable_network | `branching_angle_deg` | `bifurcation_angle_deg` | ✅ |
| articular_cartilage | `zone_ratios` | individual zone ratio params | ✅ (backend array) |
| articular_cartilage | `pore_gradient` | individual pore size params | ✅ (backend array) |
| haversian_bone | `canal_diameter_um` | `haversian_canal_diameter_um` | ✅ |
| intervertebral_disc | `np_diameter` | computed from `nucleus_percentage` | ✅ (stats-only) |
| intervertebral_disc | `af_ring_count` | `num_lamellae` | ✅ |
| intervertebral_disc | `annulus_percentage` | reference value | ✅ (stats-only) |
| intervertebral_disc | `stiffness_gradient` | reference value | ✅ (stats-only) |
| vascular_network | `outer_radius_mm` | `outer_radius` | ✅ FIXED (@deprecated alias) |
| vascular_network | `height_mm` | `height` | ✅ FIXED (@deprecated alias) |
| vascular_network | `inlet_radius_mm` | `inlet_radius` | ✅ FIXED (@deprecated alias) |
| basic (lattice) | `bounding_box_mm` | `bounding_box_x/y/z_mm` | ✅ (backend legacy tuple) |
| basic (lattice) | `unit_cell` | `lattice_type` | ✅ (backend legacy) |
| basic (lattice) | `cell_size_mm` | `unit_cell_size_mm` | ✅ (backend legacy) |
| basic (lattice) | `bounding_box` (object) | `bounding_box_x/y/z_mm` | ✅ (frontend convenience) |
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

## Quick Reference

### Completed Scaffolds
1. ✅ bladder (TUBULAR) - 43 params
2. ✅ blood_vessel (TUBULAR) - 43 params
3. ✅ nerve_conduit (TUBULAR) - 44 params
4. ✅ simple_conduit (TUBULAR) - 7 params
5. ✅ spinal_cord (TUBULAR) - 45 params
6. ✅ trachea (TUBULAR) - 61 params
7. ✅ skeletal_muscle (SOFT_TISSUE) - 34 params (3 dead code FIXED)
8. ✅ multilayer_skin (SOFT_TISSUE) - 45 params (8 dead code FIXED, 3 new params added)
9. ✅ cornea (SOFT_TISSUE) - 27 params (2 dead code FIXED, 3 documented as reference)
10. ✅ adipose (SOFT_TISSUE) - 33 params (2 dead code FIXED, 1 documented as FEA reference)
11. ✅ articular_cartilage (SKELETAL) - 41 params (0 dead code, exemplary implementation)
12. ✅ haversian_bone (SKELETAL) - 46 params (2 dead code FIXED, 3 stats-only documented)
13. ✅ intervertebral_disc (SKELETAL) - 51 params (0 dead code, 6 stats-only documented)
14. ✅ meniscus (SKELETAL) - 48 params (4 dead code FIXED, 5 stats-only documented)
15. ✅ osteochondral (SKELETAL) - 37 params (0 dead code, exemplary implementation)
16. ✅ tendon_ligament (SKELETAL) - 51 params (1 dead code FIXED)
17. ✅ trabecular_bone (SKELETAL) - 32 params (0 dead code, exemplary implementation)
18. ✅ cardiac_patch (ORGAN) - 34 params (1 dead code FIXED: crimp_wavelength)
19. ✅ hepatic_lobule (ORGAN) - 35 params (0 dead code, exemplary implementation)
20. ✅ kidney_tubule (ORGAN) - 35 params (0 dead code, exemplary implementation)
21. ✅ liver_sinusoid (ORGAN) - 39 params (0 dead code, exemplary implementation)
22. ✅ lung_alveoli (ORGAN) - 36 params (0 dead code, 1 reference-only documented)
23. ✅ pancreatic_islet (ORGAN) - 37 params (0 dead code, 2 FEA reference documented)
24. ✅ gradient_scaffold (MICROFLUIDIC) - 32 params (0 dead code, 4 FEA/stats-only documented)
25. ✅ organ_on_chip (MICROFLUIDIC) - 38 params (0 dead code, 6 stats-only documented)
26. ✅ perfusable_network (MICROFLUIDIC) - 31 params (0 dead code, 3 stats-only documented)
27. ✅ porous_disc (LEGACY) - 6 params (0 dead code, 1 frontend-only informational)
28. ✅ primitive (LEGACY) - 4 params (0 dead code, modifications UI FIXED)
29. ✅ vascular_network (LEGACY) - 15 params (0 dead code, Murray's law verified)
30. ✅ dentin_pulp (DENTAL) - 37 params (0 dead code, 10 stats-only FEA/reference)
31. ✅ ear_auricle (DENTAL) - 38 params (0 dead code, 1 stats-only FEA reference)
32. ✅ nasal_septum (DENTAL) - 42 params (0 dead code, 1 stats-only target reference)
33. ✅ basic (LATTICE) - 20 params (0 dead code, 3 stats-only: porosity, pore_size_um, seed)
34. ✅ gyroid (LATTICE) - 22 params (0 dead code, 7 stats-only: porosity/pore targets, FEA references)
35. ✅ honeycomb (LATTICE) - 22 params (0 dead code, 5 stats-only: cell_angle, porosity, texture params, seed)
36. ✅ octet_truss (LATTICE) - 22 params (0 dead code, 6 stats-only: relative_density, pore_size, d_over_pore, FEA targets)
37. ✅ schwarz_p (LATTICE) - 23 params (0 dead code, 8 stats-only: porosity, pore sizes, wall thickness, permeability, stability, modulus, seed)
38. ✅ voronoi (LATTICE) - 20 params (0 dead code, 3 stats-only: target_porosity, random_coefficient, irregularity)
39. ✅ vascular_perfusion_dish (TUBULAR) - 23 params (0 dead code, collision-aware vascular network)

### All 39 Scaffolds Complete! ✅
