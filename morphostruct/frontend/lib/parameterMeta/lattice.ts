/**
 * Parameter metadata for lattice scaffold types
 * Includes: lattice (basic), gyroid, schwarz_p, octet_truss, voronoi, honeycomb
 */

import { ScaffoldType } from '../types/scaffolds';
import { ParamMeta, boundingBoxMeta } from './types';

export const LATTICE_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  bounding_box: boundingBoxMeta,
  unit_cell_size_mm: { type: 'number', label: 'Unit Cell Size', min: 1.5, max: 3.0, step: 0.1, unit: 'mm', description: '1.5-3.0mm for cell infiltration' },

  // === Lattice Type ===
  lattice_type: {
    type: 'enum',
    label: 'Lattice Type',
    options: [
      { value: 'cubic', label: 'Cubic' },
      { value: 'bcc', label: 'BCC (Body-Centered)' },
      { value: 'fcc', label: 'FCC (Face-Centered)' },
    ],
    description: 'Unit cell type for scaffold structure',
  },

  // === Strut Properties ===
  strut_diameter_mm: { type: 'number', label: 'Strut Diameter', min: 0.3, max: 0.8, step: 0.05, unit: 'mm', description: '0.3-0.8mm for bone scaffolds' },
  strut_taper: { type: 'number', label: 'Strut Taper', min: 0, max: 0.3, step: 0.05, description: 'Taper ratio from center to joints', advanced: true },
  strut_profile: {
    type: 'enum',
    label: 'Strut Profile',
    options: [
      { value: 'circular', label: 'Circular' },
      { value: 'square', label: 'Square' },
      { value: 'hexagonal', label: 'Hexagonal' },
      { value: 'elliptical', label: 'Elliptical' },
    ],
    description: 'Cross-section profile',
    advanced: true,
  },

  // === Porosity Control ===
  porosity: { type: 'number', label: 'Porosity', min: 0.6, max: 0.8, step: 0.05, description: 'Target void fraction (60-80% for trabecular-like)' },
  pore_size_um: { type: 'number', label: 'Pore Size', min: 300, max: 800, step: 50, unit: 'um', description: '300-800 um for bone ingrowth' },

  // === Gradient Features ===
  enable_gradient_density: { type: 'boolean', label: 'Enable Gradient Density', description: 'Enable density gradient across scaffold', advanced: true },
  gradient_axis: {
    type: 'enum',
    label: 'Gradient Axis',
    options: [
      { value: 'x', label: 'X' },
      { value: 'y', label: 'Y' },
      { value: 'z', label: 'Z' },
    ],
    description: 'Axis for density gradient',
    advanced: true,
  },
  gradient_start_density: { type: 'number', label: 'Gradient Start Density', min: 0.1, max: 1.0, step: 0.1, description: 'Strut density at gradient start', advanced: true },
  gradient_end_density: { type: 'number', label: 'Gradient End Density', min: 0.1, max: 1.0, step: 0.1, description: 'Strut density at gradient end', advanced: true },

  // === Node Features ===
  enable_node_filleting: { type: 'boolean', label: 'Enable Node Filleting', description: 'Add fillets at strut joints', advanced: true },
  fillet_radius_factor: { type: 'number', label: 'Fillet Radius Factor', min: 0.1, max: 0.5, step: 0.05, description: 'Fillet radius as fraction of strut radius', advanced: true },
  enable_node_spheres: { type: 'boolean', label: 'Enable Node Spheres', description: 'Add spherical nodes at joints', advanced: true },
  node_sphere_factor: { type: 'number', label: 'Node Sphere Factor', min: 1.0, max: 2.0, step: 0.1, description: 'Node sphere size as multiple of strut radius', advanced: true },

  // === Quality & Generation ===
  resolution: { type: 'number', label: 'Resolution', min: 6, max: 16, step: 1, description: 'Cylinder segments (6-16)', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for stochastic features', advanced: true },
};

export const GYROID_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  bounding_box_x_mm: { type: 'number', label: 'Bounding Box X', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold X dimension' },
  bounding_box_y_mm: { type: 'number', label: 'Bounding Box Y', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold Y dimension' },
  bounding_box_z_mm: { type: 'number', label: 'Bounding Box Z', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold Z dimension' },
  unit_cell_size_mm: { type: 'number', label: 'Unit Cell Size', min: 0.5, max: 5, step: 0.1, unit: 'mm', description: '1.0-3.0mm optimal for bone scaffolds' },
  // === Porosity Control ===
  pore_size_um: { type: 'number', label: 'Pore Size', min: 200, max: 1200, step: 50, unit: 'um', description: '500-800 um optimal for bone ingrowth' },
  porosity: { type: 'number', label: 'Porosity', min: 0.3, max: 0.8, step: 0.02, description: '0.5-0.62 for trabecular bone-like' },
  wall_thickness_um: { type: 'number', label: 'Wall Thickness', min: 100, max: 500, step: 25, unit: 'um', description: '200-300 um for structural integrity' },
  isovalue: { type: 'number', label: 'Isovalue', min: -1, max: 1, step: 0.05, description: 'Primary porosity control (-1 to 1)' },
  // === Mechanical Properties ===
  elastic_modulus_target_gpa: { type: 'number', label: 'Target Elastic Modulus', min: 0.5, max: 30, step: 0.5, unit: 'GPa', description: '10-20 GPa for cortical bone', advanced: true },
  stress_distribution_uniformity: { type: 'number', label: 'Stress Uniformity', min: 0, max: 1, step: 0.1, description: 'Target stress uniformity (0-1)', advanced: true },
  yield_strength_target_mpa: { type: 'number', label: 'Target Yield Strength', min: 10, max: 200, step: 10, unit: 'MPa', description: 'Target yield strength', advanced: true },
  // === Surface Properties ===
  surface_area_ratio: { type: 'number', label: 'Surface Area Ratio', min: 1, max: 5, step: 0.1, description: 'mm2/mm3, higher for cell attachment', advanced: true },
  enable_surface_texture: { type: 'boolean', label: 'Enable Surface Texture', description: 'Add micro-texture for cell attachment', advanced: true },
  texture_amplitude_um: { type: 'number', label: 'Texture Amplitude', min: 1, max: 50, step: 1, unit: 'um', description: '5-20 um for osteoblasts', advanced: true },
  // === Gradient Features ===
  enable_gradient: { type: 'boolean', label: 'Enable Gradient', description: 'Enable porosity/density gradient', advanced: true },
  gradient_axis: { type: 'enum', label: 'Gradient Axis', options: [{ value: 'x', label: 'X' }, { value: 'y', label: 'Y' }, { value: 'z', label: 'Z' }], description: 'Axis for gradient', advanced: true },
  gradient_start_porosity: { type: 'number', label: 'Gradient Start Porosity', min: 0.3, max: 0.9, step: 0.05, description: 'Porosity at gradient start', advanced: true },
  gradient_end_porosity: { type: 'number', label: 'Gradient End Porosity', min: 0.3, max: 0.9, step: 0.05, description: 'Porosity at gradient end', advanced: true },
  // === Quality & Generation ===
  samples_per_cell: { type: 'number', label: 'Samples per Cell', min: 10, max: 40, step: 5, description: 'Grid resolution per unit cell (15-30)', advanced: true },
  mesh_density: { type: 'number', label: 'Mesh Density', min: 0.5, max: 2, step: 0.1, description: 'Output mesh density factor', advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 30, step: 2, description: 'Overall resolution multiplier', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducibility', advanced: true },
};

export const SCHWARZ_P_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  bounding_box_x_mm: { type: 'number', label: 'Bounding Box X', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold X dimension' },
  bounding_box_y_mm: { type: 'number', label: 'Bounding Box Y', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold Y dimension' },
  bounding_box_z_mm: { type: 'number', label: 'Bounding Box Z', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold Z dimension' },
  unit_cell_size_mm: { type: 'number', label: 'Unit Cell Size', min: 0.5, max: 5, step: 0.1, unit: 'mm', description: '1.0-3.0mm typical' },
  // === Porosity Control ===
  porosity: { type: 'number', label: 'Porosity', min: 0.5, max: 0.9, step: 0.02, description: '0.7-0.8 for high permeability' },
  small_pore_size_um: { type: 'number', label: 'Small Pore Size', min: 50, max: 300, step: 25, unit: 'um', description: '100-200 um for cell attachment' },
  large_pore_size_um: { type: 'number', label: 'Large Pore Size', min: 200, max: 700, step: 50, unit: 'um', description: '300-500 um for vascularization' },
  wall_thickness_um: { type: 'number', label: 'Wall Thickness', min: 100, max: 400, step: 25, unit: 'um', description: '150-250 um for structural integrity' },
  isovalue: { type: 'number', label: 'Isovalue', min: -3, max: 3, step: 0.25, description: 'Primary porosity control (-3 to 3)' },
  // === TPMS Shape Parameters ===
  k_parameter: { type: 'number', label: 'K Parameter', min: 0.5, max: 2, step: 0.1, description: 'Wave number modifier', advanced: true },
  s_parameter: { type: 'number', label: 'S Parameter', min: 0.5, max: 2, step: 0.1, description: 'Surface stretching factor', advanced: true },
  // === Transport Properties ===
  fluid_permeability_target: { type: 'number', label: 'Fluid Permeability Target', min: 1e-10, max: 1e-8, step: 1e-10, description: 'Target permeability (m2)', advanced: true },
  diffusion_coefficient_ratio: { type: 'number', label: 'Diffusion Ratio', min: 0.1, max: 0.8, step: 0.05, description: 'Effective/bulk diffusion ratio', advanced: true },
  // === Mechanical Properties ===
  mechanical_stability: { type: 'number', label: 'Mechanical Stability', min: 0.3, max: 1, step: 0.1, description: 'Stability factor (0-1)', advanced: true },
  elastic_modulus_target_gpa: { type: 'number', label: 'Target Elastic Modulus', min: 0.5, max: 20, step: 0.5, unit: 'GPa', description: 'Target elastic modulus', advanced: true },
  // === Gradient Features ===
  enable_gradient: { type: 'boolean', label: 'Enable Gradient', description: 'Enable porosity/density gradient', advanced: true },
  gradient_axis: { type: 'enum', label: 'Gradient Axis', options: [{ value: 'x', label: 'X' }, { value: 'y', label: 'Y' }, { value: 'z', label: 'Z' }], description: 'Axis for gradient', advanced: true },
  gradient_start_porosity: { type: 'number', label: 'Gradient Start Porosity', min: 0.4, max: 0.9, step: 0.05, description: 'Porosity at gradient start', advanced: true },
  gradient_end_porosity: { type: 'number', label: 'Gradient End Porosity', min: 0.5, max: 0.95, step: 0.05, description: 'Porosity at gradient end', advanced: true },
  // === Quality & Generation ===
  samples_per_cell: { type: 'number', label: 'Samples per Cell', min: 10, max: 40, step: 5, description: 'Grid resolution per unit cell (15-30)', advanced: true },
  mesh_density: { type: 'number', label: 'Mesh Density', min: 0.5, max: 2, step: 0.1, description: 'Output mesh density factor', advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 30, step: 2, description: 'Overall resolution multiplier', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducibility', advanced: true },
};

export const OCTET_TRUSS_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  bounding_box_x_mm: { type: 'number', label: 'Bounding Box X', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold X dimension' },
  bounding_box_y_mm: { type: 'number', label: 'Bounding Box Y', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold Y dimension' },
  bounding_box_z_mm: { type: 'number', label: 'Bounding Box Z', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold Z dimension' },
  unit_cell_edge_mm: { type: 'number', label: 'Unit Cell Edge', min: 3, max: 15, step: 0.5, unit: 'mm', description: '5-10mm for sufficient pore interconnectivity' },
  // === Strut Properties ===
  strut_diameter_mm: { type: 'number', label: 'Strut Diameter', min: 0.4, max: 2.5, step: 0.1, unit: 'mm', description: '0.8-1.5mm for bone scaffolds' },
  strut_taper: { type: 'number', label: 'Strut Taper', min: 0, max: 0.4, step: 0.05, description: 'Taper ratio from center to joints', advanced: true },
  strut_surface_roughness: { type: 'number', label: 'Surface Roughness', min: 0, max: 1, step: 0.1, description: 'Surface roughness factor (0-1)', advanced: true },
  // === Porosity & Pore Control ===
  relative_density: { type: 'number', label: 'Relative Density', min: 0.2, max: 0.7, step: 0.02, description: '0.3-0.6 for bone-mimicking properties' },
  pore_size_um: { type: 'number', label: 'Pore Size', min: 100, max: 1000, step: 50, unit: 'um', description: '200-800 um for bone ingrowth' },
  d_over_pore_ratio: { type: 'number', label: 'D/Pore Ratio', min: 1, max: 5, step: 0.2, description: 'Strut diameter to pore size ratio', advanced: true },
  // === Mechanical Properties ===
  normalized_modulus: { type: 'number', label: 'Normalized Modulus', min: 0.05, max: 0.6, step: 0.05, description: 'E/Es ratio (0.1-0.5 typical)', advanced: true },
  yield_strength_target_mpa: { type: 'number', label: 'Target Yield Strength', min: 10, max: 150, step: 10, unit: 'MPa', description: 'Target yield strength', advanced: true },
  elastic_modulus_target_gpa: { type: 'number', label: 'Target Elastic Modulus', min: 1, max: 30, step: 1, unit: 'GPa', description: 'Target elastic modulus', advanced: true },
  // === Node Features ===
  node_fillet_radius_mm: { type: 'number', label: 'Node Fillet Radius', min: 0, max: 1, step: 0.1, unit: 'mm', description: 'Fillet radius at strut joints', advanced: true },
  enable_node_spheres: { type: 'boolean', label: 'Enable Node Spheres', description: 'Add spherical nodes at joints', advanced: true },
  node_sphere_factor: { type: 'number', label: 'Node Sphere Factor', min: 1, max: 2, step: 0.1, description: 'Node sphere size relative to strut', advanced: true },
  // === Gradient Features ===
  enable_gradient: { type: 'boolean', label: 'Enable Gradient', description: 'Enable density gradient', advanced: true },
  gradient_axis: { type: 'enum', label: 'Gradient Axis', options: [{ value: 'x', label: 'X' }, { value: 'y', label: 'Y' }, { value: 'z', label: 'Z' }], description: 'Axis for gradient', advanced: true },
  gradient_start_density: { type: 'number', label: 'Gradient Start Density', min: 0.2, max: 0.6, step: 0.05, description: 'Relative density at start', advanced: true },
  gradient_end_density: { type: 'number', label: 'Gradient End Density', min: 0.3, max: 0.7, step: 0.05, description: 'Relative density at end', advanced: true },
  // === Quality & Generation ===
  resolution: { type: 'number', label: 'Resolution', min: 4, max: 20, step: 2, description: 'Cylinder segments (6-16)', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for stochastic features', advanced: true },
};

export const VORONOI_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  bounding_box_x_mm: { type: 'number', label: 'Bounding Box X', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold X dimension' },
  bounding_box_y_mm: { type: 'number', label: 'Bounding Box Y', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold Y dimension' },
  bounding_box_z_mm: { type: 'number', label: 'Bounding Box Z', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold Z dimension' },
  // === Pore Structure ===
  target_pore_radius_mm: { type: 'number', label: 'Target Pore Radius', min: 0.2, max: 1.2, step: 0.05, unit: 'mm', description: '0.3-0.9mm for bone scaffolds' },
  target_porosity: { type: 'number', label: 'Target Porosity', min: 0.5, max: 0.9, step: 0.02, description: '0.6-0.8 for trabecular-like' },
  seed_count: { type: 'number', label: 'Seed Count', min: 10, max: 200, step: 10, description: 'Number of Voronoi seed points' },
  // === Strut Properties ===
  strut_diameter_mm: { type: 'number', label: 'Strut Diameter', min: 0.1, max: 0.8, step: 0.05, unit: 'mm', description: '0.2-0.5mm for bone scaffolds' },
  strut_taper: { type: 'number', label: 'Strut Taper', min: 0, max: 0.3, step: 0.05, description: 'Taper ratio from center to joints', advanced: true },
  enable_strut_roughness: { type: 'boolean', label: 'Enable Strut Roughness', description: 'Add surface roughness', advanced: true },
  strut_roughness_amplitude: { type: 'number', label: 'Roughness Amplitude', min: 0.005, max: 0.05, step: 0.005, unit: 'mm', description: 'Surface roughness amplitude', advanced: true },
  // === Randomization Control ===
  random_coefficient: { type: 'number', label: 'Random Coefficient', min: 0, max: 1, step: 0.1, description: 'Overall randomness (0=regular, 1=random)', advanced: true },
  irregularity: { type: 'number', label: 'Irregularity', min: 0, max: 1, step: 0.1, description: 'Shape irregularity factor', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducibility' },
  // === Gradient Features ===
  enable_gradient: { type: 'boolean', label: 'Enable Gradient', description: 'Enable density gradient', advanced: true },
  gradient_direction: { type: 'enum', label: 'Gradient Direction', options: [{ value: 'x', label: 'X' }, { value: 'y', label: 'Y' }, { value: 'z', label: 'Z' }], description: 'Direction for gradient', advanced: true },
  density_gradient_start: { type: 'number', label: 'Density Gradient Start', min: 0.3, max: 0.8, step: 0.05, description: 'Density at gradient start', advanced: true },
  density_gradient_end: { type: 'number', label: 'Density Gradient End', min: 0.5, max: 0.95, step: 0.05, description: 'Density at gradient end', advanced: true },
  // === Quality & Generation ===
  resolution: { type: 'number', label: 'Resolution', min: 4, max: 16, step: 2, description: 'Cylinder segments (6-12)', advanced: true },
  margin_factor: { type: 'number', label: 'Margin Factor', min: 0.1, max: 0.5, step: 0.05, description: 'Margin around bbox for seed points', advanced: true },
  min_strut_length_mm: { type: 'number', label: 'Min Strut Length', min: 0.05, max: 0.3, step: 0.05, unit: 'mm', description: 'Minimum strut length filter', advanced: true },
};

export const HONEYCOMB_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  bounding_box_x_mm: { type: 'number', label: 'Bounding Box X', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold X dimension' },
  bounding_box_y_mm: { type: 'number', label: 'Bounding Box Y', min: 5, max: 50, step: 1, unit: 'mm', description: 'Scaffold Y dimension' },
  bounding_box_z_mm: { type: 'number', label: 'Bounding Box Z', min: 2, max: 30, step: 1, unit: 'mm', description: 'Scaffold Z dimension' },
  cell_height_mm: { type: 'number', label: 'Cell Height', min: 1, max: 30, step: 0.5, unit: 'mm', description: 'Extrusion height (uses bbox z if empty)', advanced: true },
  // === Cell Geometry ===
  wall_thickness_mm: { type: 'number', label: 'Wall Thickness', min: 0.3, max: 3, step: 0.1, unit: 'mm', description: '0.5-2.0mm for structural integrity' },
  cell_inner_length_mm: { type: 'number', label: 'Cell Inner Length', min: 1, max: 8, step: 0.5, unit: 'mm', description: '2-5mm for cell infiltration' },
  cell_angle_deg: { type: 'number', label: 'Cell Angle', min: 100, max: 140, step: 5, unit: 'deg', description: '120 for regular hexagon', advanced: true },
  cell_orientation: { type: 'enum', label: 'Cell Orientation', options: [{ value: 'flat_top', label: 'Flat Top' }, { value: 'pointy_top', label: 'Pointy Top' }], description: 'Hexagon orientation', advanced: true },
  // === Porosity ===
  porosity: { type: 'number', label: 'Porosity', min: 0.7, max: 0.95, step: 0.02, description: '0.8-0.9 for high permeability' },
  num_cells_x: { type: 'number', label: 'Cells X', min: 1, max: 20, step: 1, description: 'Number of cells in X (auto if empty)', advanced: true },
  num_cells_y: { type: 'number', label: 'Cells Y', min: 1, max: 20, step: 1, description: 'Number of cells in Y (auto if empty)', advanced: true },
  // === Wall Features ===
  enable_gradient_wall_thickness: { type: 'boolean', label: 'Gradient Wall Thickness', description: 'Enable gradient wall thickness', advanced: true },
  wall_thickness_start_mm: { type: 'number', label: 'Wall Start Thickness', min: 0.3, max: 2, step: 0.1, unit: 'mm', description: 'Wall thickness at gradient start', advanced: true },
  wall_thickness_end_mm: { type: 'number', label: 'Wall End Thickness', min: 0.5, max: 2.5, step: 0.1, unit: 'mm', description: 'Wall thickness at gradient end', advanced: true },
  gradient_axis: { type: 'enum', label: 'Gradient Axis', options: [{ value: 'x', label: 'X' }, { value: 'y', label: 'Y' }, { value: 'z', label: 'Z' }], description: 'Axis for wall thickness gradient', advanced: true },
  // === Surface Properties ===
  enable_wall_texture: { type: 'boolean', label: 'Enable Wall Texture', description: 'Add surface texture to walls', advanced: true },
  texture_depth_um: { type: 'number', label: 'Texture Depth', min: 5, max: 50, step: 5, unit: 'um', description: 'Texture depth for cell attachment', advanced: true },
  enable_wall_perforations: { type: 'boolean', label: 'Enable Perforations', description: 'Add perforations to walls', advanced: true },
  perforation_diameter_um: { type: 'number', label: 'Perforation Diameter', min: 50, max: 500, step: 25, unit: 'um', description: 'Diameter of wall perforations', advanced: true },
  perforation_spacing_mm: { type: 'number', label: 'Perforation Spacing', min: 0.2, max: 1.5, step: 0.1, unit: 'mm', description: 'Spacing between perforations', advanced: true },
  // === Quality & Generation ===
  resolution: { type: 'number', label: 'Resolution', min: 1, max: 8, step: 1, description: 'Segments per hexagon side', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for stochastic features', advanced: true },
};

// Combined export for lattice scaffold types
export const LATTICE_PARAMETER_META: Partial<Record<ScaffoldType, Record<string, ParamMeta>>> = {
  [ScaffoldType.LATTICE]: LATTICE_META,
  [ScaffoldType.GYROID]: GYROID_META,
  [ScaffoldType.SCHWARZ_P]: SCHWARZ_P_META,
  [ScaffoldType.OCTET_TRUSS]: OCTET_TRUSS_META,
  [ScaffoldType.VORONOI]: VORONOI_META,
  [ScaffoldType.HONEYCOMB]: HONEYCOMB_META,
};
