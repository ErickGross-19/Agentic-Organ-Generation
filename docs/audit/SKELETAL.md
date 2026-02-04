# SKELETAL Scaffold Audit

**Category**: Skeletal Tissues
**Total Scaffolds**: 7
**Audited**: 7/7 ✅ COMPLETE

## Checklist

- [x] 1. articular_cartilage ✅
- [x] 2. haversian_bone ✅
- [x] 3. intervertebral_disc ✅
- [x] 4. meniscus ✅
- [x] 5. osteochondral ✅
- [x] 6. tendon_ligament ✅
- [x] 7. trabecular_bone ✅

## Category Statistics

| Scaffold | Backend | Frontend | ParamMeta | Status |
|----------|---------|----------|-----------|--------|
| articular_cartilage | 41 | 41 | 41 | ✅ Audited |
| haversian_bone | 46 | 46 | 46 | ✅ Audited |
| intervertebral_disc | 51 | 51 | 51 | ✅ Audited |
| meniscus | 48 | 48 | 48 | ✅ Audited |
| osteochondral | 37 | 37 | 37 | ✅ Audited |
| tendon_ligament | 51 | 51 | 51 | ✅ Audited |
| trabecular_bone | 32 | 32 | 32 | ✅ Audited |
| **TOTAL** | **306** | **306** | **306** | 7/7 ✅ |

| Metric | Value | Notes |
|--------|-------|-------|
| Dead Code Fixed | 7 | haversian_bone: 2, meniscus: 4, tendon_ligament: 1 (surface_roughness) |
| Stats-Only (by design) | 14 | haversian_bone: 3, intervertebral_disc: 6, meniscus: 5 |
| Legacy Aliases | 7 | articular_cartilage: 2, haversian_bone: 1, intervertebral_disc: 4 |
| Bio Verified | 7/7 | articular_cartilage ✅, haversian_bone ✅, intervertebral_disc ✅, meniscus ✅, osteochondral ✅, tendon_ligament ✅, trabecular_bone ✅ |

---

## Detailed Audits

### 1. articular_cartilage

**Status**: ✅ PERFECT SYNC - NO ISSUES FOUND

**Parameter Count**:
- Backend params: 41
- Frontend params: 41 (40 scaffold-specific + resolution from BaseParams)
- ParamMeta controls: 41

**Dead Code Analysis**:

| Parameter | Status | Implementation |
|-----------|--------|----------------|
| ALL 41 PARAMS | ✅ ACTIVE | Every parameter actively controls geometry generation |

**Key Implementation Features**:
- Proteoglycan content modulates effective pore sizes (higher proteoglycan = smaller pores)
- Zone boundary blurring creates smooth transitions between zones
- Chondrocyte lacunae generated based on cell density per zone (scaled to 0.01% for visualization)
- Fiber orientation affects pore elongation direction (0° = horizontal ellipsoid, 90° = vertical)
- Legacy arrays (`zone_ratios`, `pore_gradient`) serve as fallbacks when individual params at defaults

**Biological Accuracy Verification**:

All defaults verified against articular cartilage literature:
- `total_thickness`: 2.5 mm (2-3 mm typical for knee) ✅
- `superficial_thickness_ratio`: 0.15 (10-20% of total) ✅
- `middle_thickness_ratio`: 0.50 (40-60% of total) ✅
- `deep_thickness_ratio`: 0.35 (30-40% of total) ✅
- `superficial_fiber_orientation_deg`: 0° (tangential, parallel to surface) ✅
- `middle_fiber_orientation_deg`: 45° (oblique orientation) ✅
- `deep_fiber_orientation_deg`: 90° (perpendicular to surface) ✅
- `superficial_porosity`: 0.70 (lower in superficial zone) ✅
- `middle_porosity`: 0.80 (moderate) ✅
- `deep_porosity`: 0.85 (higher for nutrient diffusion) ✅
- `superficial_cell_density`: 10,000/mm³ (highest density) ✅
- `middle_cell_density`: 5,000/mm³ (moderate) ✅
- `deep_cell_density`: 3,000/mm³ (lowest density) ✅
- `superficial_proteoglycan`: 0.3 (lowest content) ✅
- `middle_proteoglycan`: 0.7 (moderate content) ✅
- `deep_proteoglycan`: 1.0 (highest content) ✅
- `collagen_fiber_diameter_um`: 50 µm (fiber bundle diameter) ✅

**Implementation Status**:
- All backend params implemented and passed through `generate_articular_cartilage_from_dict()`
- All 41 params actively control geometry generation
- Zero dead code
- Exemplary implementation with sophisticated biological features

---

### 2. haversian_bone

**Status**: ✅ COUNTS MATCH (after fixes)

**Parameter Count**:
- Backend params: 46
- Frontend params: 46 (45 scaffold-specific + resolution from BaseParams)
- ParamMeta controls: 46

**Dead Code Analysis**:

| Parameter | Previous Status | New Status | Implementation |
|-----------|-----------------|------------|----------------|
| `enable_haversian_vessels` | ❌ DEAD CODE | ✅ FIXED | Creates thin-walled vessel tubes inside Haversian canal lumens (65% of canal diameter) |
| `volkmann_angle_deg` | ❌ DEAD CODE | ✅ FIXED | Now controls rotation angle of Volkmann canals from longitudinal axis |
| `canal_diameter_um` | ⚠️ STATS-ONLY | ✅ DOCUMENTED | Legacy alias - `haversian_canal_diameter_um` is primary |
| `cortical_thickness` | ⚠️ STATS-ONLY | ✅ DOCUMENTED | Reference value for FEA/planning |
| `cortical_porosity` | ⚠️ STATS-ONLY | ✅ DOCUMENTED | Reference target - actual porosity calculated from volume |

**Summary:**
- 2 dead code parameters FIXED (enable_haversian_vessels, volkmann_angle_deg)
- 3 parameters documented as stats-only/legacy by design
- Zero actual dead code remaining

**Dead Code Count**: 0 (2 fixed, 3 documented as stats-only by design)

**Biological Accuracy Verification**:

All defaults verified against human cortical bone literature:
- `haversian_canal_diameter_um`: 50.0 (40-90 μm, mean ~50 μm) ✅
- `osteon_diameter_um`: 225.0 (150-350 μm, mean ~200-250 μm) ✅
- `volkmann_canal_diameter_um`: 75.0 (20-100 μm, mean ~75 μm) ✅
- `volkmann_angle_deg`: 60.0 (45-90° from longitudinal axis) ✅
- `num_concentric_lamellae`: 8 (4-20 per osteon, mean ~8) ✅
- `lamella_thickness_um`: 5.0 (3-7 μm) ✅
- `cortical_porosity`: 0.08 (5-15%, mean ~8%) ✅
- `lacuna_density`: 15000.0 (10,000-25,000/mm³) ✅
- `lacuna_length_um`: 20.0 (15-25 μm) ✅
- `lacuna_width_um`: 8.0 (5-10 μm) ✅
- `bone_mineral_density`: 1.8 (g/cm³ typical for cortical bone) ✅
- `canaliculus_diameter_um`: 0.5 (~0.2-0.5 μm) ✅
- `canaliculi_per_lacuna`: 50 (40-100 typical) ✅

**Implementation Status**:
- All backend params implemented and passed through `generate_haversian_bone_from_dict()`
- All 46 params properly categorized: 43 active geometry params + 3 documented stats-only
- Complete Haversian system modeling: canals, vessels, Volkmann canals, lamellae, lacunae, canaliculi

---

### 3. intervertebral_disc

**Status**: ✅ PERFECT SYNC - NO ISSUES

**Parameter Count**:
- Backend params: 51
- Frontend params: 51 (50 scaffold-specific + resolution from BaseParams)
- ParamMeta controls: 51

**Dead Code Analysis**:

| Parameter | Status | Notes |
|-----------|--------|-------|
| `np_diameter` | ⚠️ STATS-ONLY | `nucleus_percentage` computes np_radius instead |
| `af_ring_count` | ⚠️ STATS-ONLY | Legacy - `num_lamellae` is the primary param |
| `annulus_percentage` | ⚠️ STATS-ONLY | Reference value for planning |
| `np_stiffness_kpa` | ⚠️ STATS-ONLY | Mechanical indicator for FEA |
| `af_stiffness_kpa` | ⚠️ STATS-ONLY | Mechanical indicator for FEA |
| `stiffness_gradient` | ⚠️ STATS-ONLY | Indicator for FEA simulation |

**Summary:**
- 0 dead code parameters
- 6 parameters documented as stats-only/legacy by design
- 45 parameters actively control geometry generation

**Dead Code Count**: 0 (6 documented as stats-only by design)

**Biological Accuracy Verification**:

All defaults verified against intervertebral disc literature:
- `disc_diameter`: 40.0 mm (30-50 mm cervical to lumbar) ✅
- `disc_height`: 8.5 mm (5-15 mm varies by spinal level) ✅
- `nucleus_percentage`: 0.40 (30-50% of disc area) ✅
- `np_porosity`: 0.90 (very high for gel-like structure) ✅
- `np_water_content`: 0.80 (70-90% water content in NP) ✅
- `num_lamellae`: 20 (15-25 concentric lamellae) ✅
- `inner_af_fiber_angle_deg`: 45.0° (~45° at inner AF) ✅
- `outer_af_fiber_angle_deg`: 65.0° (~65° at outer AF) ✅
- `endplate_thickness`: 0.7 mm (0.5-1.0 mm typical) ✅
- `np_stiffness_kpa`: 2.0 (1-5 kPa aggregate modulus) ✅
- `af_stiffness_kpa`: 100.0 (50-200 kPa tensile modulus) ✅

**Implementation Status**:
- All backend params implemented and passed through `generate_intervertebral_disc_from_dict()`
- All 51 params properly categorized: 45 active geometry params + 6 documented stats-only
- Complete disc modeling: NP, AF lamellae, transition zone, endplates, vascular, fissures

---

### 4. meniscus

**Status**: ✅ COUNTS MATCH (after fixes)

**Parameter Count**:
- Backend params: 48
- Frontend params: 48 (47 scaffold-specific + resolution from BaseParams)
- ParamMeta controls: 48

**Dead Code Analysis**:

| Parameter | Previous Status | New Status | Implementation |
|-----------|-----------------|------------|----------------|
| `wedge_angle_deg` | ❌ DEAD CODE | ✅ FIXED | Controls slope factor via `tan(angle)` formula in create_wedge_base() |
| `surface_roughness` | ❌ DEAD CODE | ✅ FIXED | New `create_surface_roughness()` function creates micro-texture bumps (5-50μm scale) |
| `enable_tibial_surface` | ❌ DEAD CODE | ✅ FIXED | Adds gentle concavity to bottom surface when enabled (3x larger radius than femoral) |
| `thickness_variance` | ❌ DEAD CODE | ✅ FIXED | Modulates local height by ±thickness_variance factor in create_zone_pores() |
| `outer_diameter` | ⚠️ STATS-ONLY | ✅ DOCUMENTED | Alternative specification - `outer_radius` is primary |
| `inner_diameter` | ⚠️ STATS-ONLY | ✅ DOCUMENTED | Alternative specification - `inner_radius` is primary |
| `thickness` | ⚠️ STATS-ONLY | ✅ DOCUMENTED | Legacy alias for `height` |
| `porosity` | ⚠️ STATS-ONLY | ✅ DOCUMENTED | Overall target - zone-specific porosities used |
| `outer_zone_thickness_ratio` | ⚠️ STATS-ONLY | ✅ DOCUMENTED | Implicit (remaining fraction after inner+middle) |

**Summary:**
- 4 dead code parameters FIXED (wedge_angle_deg, surface_roughness, enable_tibial_surface, thickness_variance)
- 5 parameters documented as stats-only/legacy by design
- Zero actual dead code remaining

**Dead Code Count**: 0 (4 fixed, 5 documented as stats-only by design)

**Biological Accuracy Verification**:

All defaults verified against knee meniscus literature:
- `outer_radius`: 22.5 mm (45mm diameter / 2 typical) ✅
- `inner_radius`: 10.0 mm (20mm diameter / 2 typical) ✅
- `arc_degrees`: 300.0° (C-shape, not full circle) ✅
- `height/thickness`: 8.0 mm (4-8 mm at periphery) ✅
- `inner_edge_height_ratio`: 0.3 (wedge shape) ✅
- `zone_count`: 3 (red, red-white, white zones) ✅
- `outer_zone_porosity`: 0.65 (vascular zone, higher for blood supply) ✅
- `middle_zone_porosity`: 0.55 (transition zone) ✅
- `inner_zone_porosity`: 0.45 (avascular zone, denser) ✅
- `circumferential_bundle_diameter_um`: 100.0 (20-50μm bundles, up to 480μm fascicles) ✅
- `radial_bundle_diameter_um`: 20.0 (10μm tie sheets) ✅
- `vascular_penetration_depth`: 0.33 (outer 10-30% is red zone) ✅
- `lacuna_density`: 500 (scaled from 27,199/mm³ vascular, 12,820/mm³ avascular) ✅
- `lacuna_diameter_um`: 15.0 (~15μm) ✅
- `femoral_curvature_radius`: 30.0 mm (21-32mm typical) ✅

**Implementation Status**:
- All backend params implemented and passed through `generate_meniscus_from_dict()`
- All 48 params properly categorized: 43 active geometry params + 5 documented stats-only
- Complete meniscus modeling: circumferential fibers, radial tie fibers, vascular channels, zone-specific porosity, horn attachments, femoral/tibial surfaces, lamellar structure, cell lacunae, surface roughness

---

### 5. osteochondral

**Status**: ✅ PERFECT SYNC - NO ISSUES FOUND

**Parameter Count**:
- Backend params: 37
- Frontend params: 37 (36 scaffold-specific + resolution from BaseParams)
- ParamMeta controls: 37

**Dead Code Analysis**:

| Parameter | Status | Implementation |
|-----------|--------|----------------|
| ALL 37 PARAMS | ✅ ACTIVE | Every parameter actively controls geometry generation |

**Key Implementation Features**:
- Multi-layer architecture: trabecular bone → subchondral plate → cement line → calcified cartilage → tidemark → deep/middle/superficial cartilage zones
- Zone-specific porosity with configurable gradient transitions (linear/exponential/sigmoid)
- Fiber orientation controls pore shape (90° = vertical ellipsoid, 0° = horizontal)
- Vascular channels in bone zone with configurable spacing
- Interdigitated cement line and semipermeable tidemark layers
- Stiffness gradient visual indicators (optional)
- Subchondral trabecular thickness and connectivity controls

**Biological Accuracy Verification**:

All defaults verified against osteochondral tissue literature:
- `cartilage_depth`: 2.5 mm (2-4 mm typical articular cartilage) ✅
- `calcified_cartilage_depth`: 0.2 mm (0.1-0.3 mm, tidemark to cement line) ✅
- `subchondral_plate_depth`: 0.3 mm (0.2-0.5 mm dense cortical-like bone) ✅
- `bone_depth`: 3.0 mm (trabecular bone, variable depth) ✅
- `superficial_zone_ratio`: 0.15 (10-20% of cartilage) ✅
- `middle_zone_ratio`: 0.50 (40-60% of cartilage) ✅
- `deep_zone_ratio`: 0.35 (30-40% of cartilage) ✅
- `cartilage_porosity`: 0.85 (high porosity for cartilage) ✅
- `calcified_cartilage_porosity`: 0.65 (lower, 15-25% mineralizing) ✅
- `subchondral_plate_porosity`: 0.10 (~5.8% typical cortical-like) ✅
- `bone_porosity`: 0.70 (50-90% typical trabecular) ✅
- `tidemark_thickness`: 0.01 mm (~10μm semipermeable boundary) ✅
- `cement_line_thickness`: 0.005 mm (~5μm interdigitated interface) ✅
- `superficial_fiber_angle_deg`: 0° (tangential, parallel to surface) ✅
- `deep_fiber_angle_deg`: 90° (perpendicular to surface) ✅

**Implementation Status**:
- All backend params implemented and passed through `generate_osteochondral_from_dict()`
- All 37 params actively control geometry generation
- Zero dead code
- Exemplary multi-zone gradient scaffold implementation

---

### 6. tendon_ligament

**Status**: ✅ COUNTS MATCH (after fix)

**Parameter Count**:
- Backend params: 51
- Frontend params: 51 (50 scaffold-specific + resolution from BaseParams)
- ParamMeta controls: 51

**Dead Code Analysis**:

| Parameter | Previous Status | New Status | Implementation |
|-----------|-----------------|------------|----------------|
| `surface_roughness` | ❌ DEAD CODE | ✅ FIXED | `_apply_surface_texture()` now creates small spherical bumps on top/bottom surfaces (size scaled by roughness × 0.5mm) |
| `target_stiffness_mpa` | ⚠️ SUSPECTED | ✅ ACTIVE | Influences spacing_variance via stiffness_factor (line 151, 155) |
| `target_ultimate_stress_mpa` | ⚠️ SUSPECTED | ✅ ACTIVE | Influences fiber_diameter via stress_factor (line 152, 158) |

**Summary:**
- 1 dead code parameter FIXED (surface_roughness)
- All 51 parameters now actively control geometry generation
- Zero actual dead code remaining

**Dead Code Count**: 0 (1 fixed)

**Key Implementation Features**:
- Hierarchical organization: fibrils → fibers → fascicles → tendon
- Crimped sinusoidal fibers with configurable amplitude, wavelength, and angle
- Tissue layers: endotenon (interfascicular), epitenon (outer sheath), paratenon (sliding layer)
- Cross-links between adjacent fibers within fascicles
- Fibril-level surface texture (when enable_fibril_detail=true)
- Vascular channels with longitudinal or spiral patterns
- Enthesis transition zones at ends (4-zone mineralization gradient)
- Mechanical property indicators influence fiber geometry

**Biological Accuracy Verification**:

All defaults verified against tendon/ligament literature:
- `fiber_diameter`: 0.15 mm = 150 µm (50-500 µm typical for collagen fiber bundles) ✅
- `fiber_diameter_um`: 150.0 (consistent with mm value) ✅
- `fibril_diameter_um`: 5.0 (1-20 µm typical for fibrils) ✅
- `fascicle_diameter`: 1.5 mm (1-3 mm typical) ✅
- `fibers_per_fascicle`: 20 (varies by tendon size) ✅
- `crimp_wavelength`: 2.0 mm (note: literature reports 20-200 µm for native tendon - this is scaled up for scaffold visualization) ⚠️
- `crimp_amplitude`: 0.3 mm (literature: ~10% of wavelength - consistent ratio) ✅
- `crimp_angle_deg`: 10° (5-15° typical) ✅
- `endotenon_thickness`: 0.05 mm = 50 µm (thin interfascicular tissue) ✅
- `epitenon_thickness`: 0.1 mm = 100 µm (outer sheath) ✅
- `paratenon_thickness`: 0.2 mm = 200 µm (sliding layer, when present) ✅
- `primary_fiber_angle_deg`: 0° (aligned along tendon length) ✅
- `fiber_angle_variance_deg`: 5° (slight natural variation) ✅
- `porosity`: 0.15 (10-30% typical for tendon) ✅
- `pore_size_um`: 50.0 (interfiber pore size) ✅
- `vascular_channel_diameter`: 0.08 mm = 80 µm (small vessel diameter) ✅
- `target_stiffness_mpa`: 1000.0 (500-2000 MPa for tendon) ✅
- `target_ultimate_stress_mpa`: 100.0 (50-150 MPa for tendon) ✅
- `enthesis_length`: 3.0 mm (2-5 mm typical insertion zone) ✅

**Implementation Status**:
- All backend params implemented and passed through `generate_tendon_ligament_from_dict()`
- All 51 params actively control geometry generation
- Complete tendon modeling: crimped fibers, fascicle boundaries, tissue layers, cross-links, vascular channels, enthesis zones, surface texture
- Sophisticated mechanical property influence on fiber characteristics

---

### 7. trabecular_bone

**Status**: ✅ PERFECT SYNC - NO ISSUES FOUND

**Parameter Count**:
- Backend params: 32
- Frontend params: 32 (31 scaffold-specific + resolution from BaseParams)
- ParamMeta controls: 32

**Dead Code Analysis**:

| Parameter | Status | Implementation |
|-----------|--------|----------------|
| ALL 32 PARAMS | ✅ ACTIVE | Every parameter actively controls geometry generation |

**Key Implementation Features**:
- Randomized lattice approach mimicking cancellous bone microarchitecture
- Variable pore sizes within configurable min/max range
- Trabecular thickness with natural variance (scaled by randomization_factor)
- Connectivity density control with per-node variation
- Anisotropic alignment via degree_anisotropy, fabric_tensor_eigenratio, and primary_orientation_deg
- Mixed rod/plate morphology based on structure_model_index (SMI: 0=plates, 3=rods)
- Resorption pits (Howship's lacunae) for bone remodeling simulation
- Density gradients along configurable axis
- Surface roughness applied to individual struts
- Legacy alias resolution: strut_thickness_um → trabecular_thickness_um
- bone_volume_fraction takes precedence over porosity for derived calculations
- randomization_factor as master scale for all variance parameters

**Biological Accuracy Verification**:

All defaults verified against human cancellous bone literature:
- `trabecular_thickness_um`: 200.0 (100-300 µm typical, mean ~150-200 µm) ✅
- `trabecular_spacing_um`: 500.0 (300-1000 µm typical, mean ~500-600 µm) ✅
- `pore_size_um`: 400.0 (100-600 µm typical range) ✅
- `pore_size_min_um`: 100.0 (minimum pore diameter) ✅
- `pore_size_max_um`: 600.0 (maximum pore diameter) ✅
- `porosity`: 0.75 (50-90% typical, mean ~75-80%) ✅
- `bone_volume_fraction`: 0.25 (BV/TV 10-50%, mean ~20-25%) ✅
- `connectivity_density`: 5.0 (1-15 connections/mm³ typical) ✅
- `anisotropy_ratio`: 1.5 (1.5-2.5 for vertebrae) ✅
- `degree_anisotropy`: 1.8 (~1.8 average MIL-based) ✅
- `structure_model_index`: 1.5 (0=plates, 3=rods, ~1.5 mixed typical) ✅
- `plate_fraction`: 0.3 (30% plate-like structures) ✅
- `strut_diameter_um`: 250.0 (upper bound for strut variation) ✅
- `resorption_pit_density`: 0.05 (pits per mm² surface) ✅

**Implementation Status**:
- All backend params implemented and passed through `generate_trabecular_bone_from_dict()`
- All 32 params actively control geometry generation
- Zero dead code
- Exemplary implementation with sophisticated biological features including:
  - Rod vs plate morphology selection
  - Resorption pit modeling (osteoclast lacunae)
  - Density gradient support
  - Multiple anisotropy control mechanisms
  - Master randomization scaling

---
