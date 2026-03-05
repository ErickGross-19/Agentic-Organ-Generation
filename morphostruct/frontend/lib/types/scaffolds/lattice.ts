/**
 * Lattice Parameters (Basic and Advanced)
 */

import { BaseParams, UnitCell } from './base';

// ============================================================================
// Lattice Parameters
// ============================================================================

export interface LatticeParams extends BaseParams {
  // === Basic Geometry ===
  bounding_box?: { x: number; y: number; z: number };
  bounding_box_x_mm?: number;
  bounding_box_y_mm?: number;
  bounding_box_z_mm?: number;
  unit_cell_size_mm?: number;
  /** @deprecated Use unit_cell_size_mm instead */
  cell_size_mm?: number;

  // === Lattice Type ===
  lattice_type?: 'cubic' | 'bcc' | 'fcc';
  /** @deprecated Use lattice_type instead */
  unit_cell?: UnitCell;

  // === Strut Properties ===
  strut_diameter_mm: number;
  strut_taper?: number;
  strut_profile?: 'circular' | 'square' | 'hexagonal' | 'elliptical';

  // === Porosity Control ===
  porosity?: number;
  pore_size_um?: number;

  // === Gradient Features ===
  enable_gradient_density?: boolean;
  gradient_axis?: 'x' | 'y' | 'z';
  gradient_start_density?: number;
  gradient_end_density?: number;

  // === Node Features ===
  enable_node_filleting?: boolean;
  fillet_radius_factor?: number;
  enable_node_spheres?: boolean;
  node_sphere_factor?: number;

  // === Generation Settings ===
  seed?: number;
}

// ============================================================================
// Advanced Lattice Parameters
// ============================================================================

export interface GyroidParams extends BaseParams {
  // === Basic Geometry ===
  bounding_box_mm?: { x: number; y: number; z: number };
  bounding_box_x_mm?: number;
  bounding_box_y_mm?: number;
  bounding_box_z_mm?: number;
  unit_cell_size_mm?: number;
  /** @deprecated Use unit_cell_size_mm instead */
  cell_size_mm?: number;

  // === Porosity Control ===
  pore_size_um?: number;
  porosity?: number;
  wall_thickness_um?: number;
  /** @deprecated Use wall_thickness_um instead */
  wall_thickness_mm?: number;
  isovalue: number;

  // === Mechanical Properties (Target) ===
  elastic_modulus_target_gpa?: number;
  stress_distribution_uniformity?: number;
  yield_strength_target_mpa?: number;

  // === Surface Properties ===
  surface_area_ratio?: number;
  enable_surface_texture?: boolean;
  texture_amplitude_um?: number;

  // === Gradient Features ===
  enable_gradient?: boolean;
  gradient_axis?: 'x' | 'y' | 'z';
  gradient_start_porosity?: number;
  gradient_end_porosity?: number;

  // === Quality & Generation ===
  samples_per_cell: number;
  mesh_density?: number;
  seed?: number;
}

export interface SchwarzPParams extends BaseParams {
  // === Basic Geometry ===
  bounding_box_mm?: { x: number; y: number; z: number };
  bounding_box_x_mm?: number;
  bounding_box_y_mm?: number;
  bounding_box_z_mm?: number;
  unit_cell_size_mm?: number;
  /** @deprecated Use unit_cell_size_mm instead */
  cell_size_mm?: number;

  // === Porosity Control ===
  porosity?: number;
  small_pore_size_um?: number;
  large_pore_size_um?: number;
  wall_thickness_um?: number;
  /** @deprecated Use wall_thickness_um instead */
  wall_thickness_mm?: number;
  isovalue: number;

  // === TPMS Shape Parameters ===
  k_parameter?: number;
  s_parameter?: number;

  // === Transport Properties ===
  fluid_permeability_target?: number;
  diffusion_coefficient_ratio?: number;

  // === Mechanical Properties ===
  mechanical_stability?: number;
  elastic_modulus_target_gpa?: number;

  // === Gradient Features ===
  enable_gradient?: boolean;
  gradient_axis?: 'x' | 'y' | 'z';
  gradient_start_porosity?: number;
  gradient_end_porosity?: number;

  // === Quality & Generation ===
  samples_per_cell: number;
  mesh_density?: number;
  seed?: number;
}

export interface OctetTrussParams extends BaseParams {
  // === Basic Geometry ===
  bounding_box_mm?: { x: number; y: number; z: number };
  bounding_box_x_mm?: number;
  bounding_box_y_mm?: number;
  bounding_box_z_mm?: number;
  unit_cell_edge_mm?: number;
  /** @deprecated Use unit_cell_edge_mm instead */
  cell_size_mm?: number;

  // === Strut Properties ===
  strut_diameter_mm: number;
  strut_taper?: number;
  strut_surface_roughness?: number;

  // === Porosity & Pore Control ===
  relative_density?: number;
  pore_size_um?: number;
  d_over_pore_ratio?: number;

  // === Mechanical Properties (Target) ===
  normalized_modulus?: number;
  yield_strength_target_mpa?: number;
  elastic_modulus_target_gpa?: number;

  // === Node Features ===
  node_fillet_radius_mm?: number;
  enable_node_spheres?: boolean;
  node_sphere_factor?: number;

  // === Gradient Features ===
  enable_gradient?: boolean;
  gradient_axis?: 'x' | 'y' | 'z';
  gradient_start_density?: number;
  gradient_end_density?: number;

  // === Generation Settings ===
  seed?: number;
}

export interface VoronoiParams extends BaseParams {
  // === Basic Geometry ===
  bounding_box_mm?: { x: number; y: number; z: number };
  bounding_box_x_mm?: number;
  bounding_box_y_mm?: number;
  bounding_box_z_mm?: number;

  // === Pore Structure ===
  target_pore_radius_mm?: number;
  target_porosity?: number;
  seed_count?: number;
  /** @deprecated Use seed_count instead */
  cell_count?: number;

  // === Strut Properties ===
  strut_diameter_mm: number;
  strut_taper?: number;
  enable_strut_roughness?: boolean;
  strut_roughness_amplitude?: number;

  // === Randomization Control ===
  random_coefficient?: number;
  irregularity?: number;
  seed?: number;

  // === Gradient Features ===
  enable_gradient?: boolean;
  gradient_direction?: 'x' | 'y' | 'z';
  density_gradient_start?: number;
  density_gradient_end?: number;

  // === Quality & Generation ===
  margin_factor: number;
  min_strut_length_mm?: number;
}

export interface HoneycombParams extends BaseParams {
  // === Basic Geometry ===
  bounding_box_mm?: { x: number; y: number; z: number };
  bounding_box_x_mm?: number;
  bounding_box_y_mm?: number;
  bounding_box_z_mm?: number;
  cell_height_mm?: number;
  /** @deprecated Use cell_height_mm instead */
  height_mm?: number;

  // === Cell Geometry ===
  wall_thickness_mm: number;
  cell_inner_length_mm?: number;
  /** @deprecated Use cell_inner_length_mm instead */
  cell_size_mm?: number;
  cell_angle_deg?: number;
  cell_orientation?: 'flat_top' | 'pointy_top';

  // === Porosity ===
  porosity?: number;
  num_cells_x?: number;
  num_cells_y?: number;

  // === Wall Features ===
  enable_gradient_wall_thickness?: boolean;
  wall_thickness_start_mm?: number;
  wall_thickness_end_mm?: number;
  gradient_axis?: 'x' | 'y' | 'z';

  // === Surface Properties ===
  enable_wall_texture?: boolean;
  texture_depth_um?: number;
  enable_wall_perforations?: boolean;
  perforation_diameter_um?: number;
  perforation_spacing_mm?: number;

  // === Generation Settings ===
  seed?: number;
}

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULT_LATTICE: LatticeParams = {
  resolution: 8,
  // === Basic Geometry ===
  bounding_box: { x: 10.0, y: 10.0, z: 10.0 },
  bounding_box_x_mm: 10.0,
  bounding_box_y_mm: 10.0,
  bounding_box_z_mm: 10.0,
  unit_cell_size_mm: 2.0,
  cell_size_mm: 2.0,

  // === Lattice Type ===
  lattice_type: 'cubic',
  unit_cell: UnitCell.CUBIC,

  // === Strut Properties ===
  strut_diameter_mm: 0.5,
  strut_taper: 0.0,
  strut_profile: 'circular',

  // === Porosity Control ===
  porosity: 0.7,
  pore_size_um: 500.0,

  // === Gradient Features ===
  enable_gradient_density: false,
  gradient_axis: 'z',
  gradient_start_density: 0.5,
  gradient_end_density: 1.0,

  // === Node Features ===
  enable_node_filleting: false,
  fillet_radius_factor: 0.3,
  enable_node_spheres: false,
  node_sphere_factor: 1.1,

  // === Generation Settings ===
  seed: undefined,
};

export const DEFAULT_GYROID: GyroidParams = {
  resolution: 15,
  // === Basic Geometry ===
  bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
  bounding_box_x_mm: 10.0,
  bounding_box_y_mm: 10.0,
  bounding_box_z_mm: 10.0,
  unit_cell_size_mm: 1.5,
  cell_size_mm: 1.5,

  // === Porosity Control ===
  pore_size_um: 700.0,
  porosity: 0.56,
  wall_thickness_um: 250.0,
  wall_thickness_mm: 0.25,
  isovalue: 0.0,

  // === Mechanical Properties (Target) ===
  elastic_modulus_target_gpa: 15.0,
  stress_distribution_uniformity: 0.8,
  yield_strength_target_mpa: 100.0,

  // === Surface Properties ===
  surface_area_ratio: 2.5,
  enable_surface_texture: false,
  texture_amplitude_um: 10.0,

  // === Gradient Features ===
  enable_gradient: false,
  gradient_axis: 'z',
  gradient_start_porosity: 0.5,
  gradient_end_porosity: 0.7,

  // === Quality & Generation ===
  samples_per_cell: 20,
  mesh_density: 1.0,
  seed: undefined,
};

export const DEFAULT_SCHWARZ_P: SchwarzPParams = {
  resolution: 15,
  // === Basic Geometry ===
  bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
  bounding_box_x_mm: 10.0,
  bounding_box_y_mm: 10.0,
  bounding_box_z_mm: 10.0,
  unit_cell_size_mm: 2.0,
  cell_size_mm: 2.0,

  // === Porosity Control ===
  porosity: 0.75,
  small_pore_size_um: 200.0,
  large_pore_size_um: 400.0,
  wall_thickness_um: 200.0,
  wall_thickness_mm: 0.2,
  isovalue: 0.0,

  // === TPMS Shape Parameters ===
  k_parameter: 1.0,
  s_parameter: 1.0,

  // === Transport Properties ===
  fluid_permeability_target: 1e-9,
  diffusion_coefficient_ratio: 0.3,

  // === Mechanical Properties ===
  mechanical_stability: 0.7,
  elastic_modulus_target_gpa: 5.0,

  // === Gradient Features ===
  enable_gradient: false,
  gradient_axis: 'z',
  gradient_start_porosity: 0.6,
  gradient_end_porosity: 0.85,

  // === Quality & Generation ===
  samples_per_cell: 20,
  mesh_density: 1.0,
  seed: undefined,
};

export const DEFAULT_OCTET_TRUSS: OctetTrussParams = {
  resolution: 8,
  // === Basic Geometry ===
  bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
  bounding_box_x_mm: 10.0,
  bounding_box_y_mm: 10.0,
  bounding_box_z_mm: 10.0,
  unit_cell_edge_mm: 7.5,
  cell_size_mm: 7.5,

  // === Strut Properties ===
  strut_diameter_mm: 1.2,
  strut_taper: 0.0,
  strut_surface_roughness: 0.0,

  // === Porosity & Pore Control ===
  relative_density: 0.48,
  pore_size_um: 500.0,
  d_over_pore_ratio: 2.4,

  // === Mechanical Properties (Target) ===
  normalized_modulus: 0.25,
  yield_strength_target_mpa: 50.0,
  elastic_modulus_target_gpa: 10.0,

  // === Node Features ===
  node_fillet_radius_mm: 0.0,
  enable_node_spheres: false,
  node_sphere_factor: 1.2,

  // === Gradient Features ===
  enable_gradient: false,
  gradient_axis: 'z',
  gradient_start_density: 0.3,
  gradient_end_density: 0.6,

  // === Generation Settings ===
  seed: undefined,
};

export const DEFAULT_VORONOI: VoronoiParams = {
  resolution: 8,
  // === Basic Geometry ===
  bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
  bounding_box_x_mm: 10.0,
  bounding_box_y_mm: 10.0,
  bounding_box_z_mm: 10.0,

  // === Pore Structure ===
  target_pore_radius_mm: 0.6,
  target_porosity: 0.70,
  seed_count: undefined,
  cell_count: 30,

  // === Strut Properties ===
  strut_diameter_mm: 0.3,
  strut_taper: 0.0,
  enable_strut_roughness: false,
  strut_roughness_amplitude: 0.02,

  // === Randomization Control ===
  random_coefficient: 0.5,
  irregularity: 0.5,
  seed: undefined,

  // === Gradient Features ===
  enable_gradient: false,
  gradient_direction: 'z',
  density_gradient_start: 0.5,
  density_gradient_end: 0.9,

  // === Quality & Generation ===
  margin_factor: 0.2,
  min_strut_length_mm: 0.1,
};

export const DEFAULT_HONEYCOMB: HoneycombParams = {
  resolution: 1,
  // === Basic Geometry ===
  bounding_box_mm: { x: 10.0, y: 10.0, z: 5.0 },
  bounding_box_x_mm: 10.0,
  bounding_box_y_mm: 10.0,
  bounding_box_z_mm: 5.0,
  cell_height_mm: undefined,
  height_mm: undefined,

  // === Cell Geometry ===
  wall_thickness_mm: 1.0,
  cell_inner_length_mm: 3.0,
  cell_size_mm: 3.0,
  cell_angle_deg: 120.0,
  cell_orientation: 'flat_top',

  // === Porosity ===
  porosity: 0.85,
  num_cells_x: undefined,
  num_cells_y: undefined,

  // === Wall Features ===
  enable_gradient_wall_thickness: false,
  wall_thickness_start_mm: 0.8,
  wall_thickness_end_mm: 1.5,
  gradient_axis: 'x',

  // === Surface Properties ===
  enable_wall_texture: false,
  texture_depth_um: 20.0,
  enable_wall_perforations: false,
  perforation_diameter_um: 200.0,
  perforation_spacing_mm: 0.5,

  // === Generation Settings ===
  seed: undefined,
};
