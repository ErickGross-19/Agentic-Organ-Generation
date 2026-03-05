/**
 * Parameter metadata for microfluidic scaffold types
 * Includes: organ_on_chip, gradient_scaffold, perfusable_network
 */

import { ScaffoldType } from '../types/scaffolds';
import { ParamMeta, boundingBoxMeta } from './types';

export const ORGAN_ON_CHIP_META: Record<string, ParamMeta> = {
  // Main Channel Geometry
  main_channel_width_um: { type: 'number', label: 'Main Channel Width', min: 50, max: 500, step: 10, unit: 'um', description: '50-500 um typical' },
  main_channel_height_um: { type: 'number', label: 'Main Channel Height', min: 25, max: 200, step: 5, unit: 'um', description: '25-200 um typical', advanced: true },
  main_channel_length_mm: { type: 'number', label: 'Main Channel Length', min: 5, max: 30, step: 1, unit: 'mm', description: '5-30 mm typical', advanced: true },

  // Microchannel Network
  enable_microchannels: { type: 'boolean', label: 'Enable Microchannels', description: 'Enable microchannel network for cell guidance' },
  microchannel_width_um: { type: 'number', label: 'Microchannel Width', min: 50, max: 200, step: 10, unit: 'um', description: '50-200 um for cell guidance', advanced: true },
  microchannel_height_um: { type: 'number', label: 'Microchannel Height', min: 20, max: 100, step: 5, unit: 'um', description: '20-100 um typical', advanced: true },
  microchannel_count: { type: 'number', label: 'Microchannel Count', min: 1, max: 20, step: 1, description: 'Per chamber', advanced: true },
  microchannel_spacing_um: { type: 'number', label: 'Microchannel Spacing', min: 100, max: 500, step: 25, unit: 'um', description: 'Center-to-center', advanced: true },

  // Chamber Configuration
  chamber_width_mm: { type: 'number', label: 'Chamber Width', min: 1, max: 20, step: 0.5, unit: 'mm', description: '1-20 mm typical' },
  chamber_height_um: { type: 'number', label: 'Chamber Height', min: 25, max: 150, step: 5, unit: 'um', description: '25-150 um for thin tissue', advanced: true },
  chamber_length_mm: { type: 'number', label: 'Chamber Length', min: 2, max: 15, step: 0.5, unit: 'mm', description: '2-15 mm typical' },
  chamber_volume_nl: { type: 'number', label: 'Chamber Volume', min: 50, max: 2000, step: 50, unit: 'nL', description: 'Informational ~500 nL typical', advanced: true },
  num_chambers: { type: 'number', label: 'Number of Chambers', min: 1, max: 96, step: 1, description: '1-96 for high-throughput' },
  chamber_spacing_mm: { type: 'number', label: 'Chamber Spacing', min: 1, max: 10, step: 0.5, unit: 'mm', description: 'Between chambers', advanced: true },

  // Membrane Configuration
  enable_membrane: { type: 'boolean', label: 'Enable Membrane', description: 'Porous membrane between chambers for co-culture', advanced: true },
  membrane_thickness_um: { type: 'number', label: 'Membrane Thickness', min: 5, max: 50, step: 1, unit: 'um', description: '5-50 um PET/PDMS membranes', advanced: true },
  membrane_pore_size_um: { type: 'number', label: 'Membrane Pore Size', min: 0.1, max: 8, step: 0.1, unit: 'um', description: '0.1-8.0 um for cell separation', advanced: true },
  membrane_pore_density_per_cm2: { type: 'number', label: 'Membrane Pore Density', min: 1e4, max: 1e6, step: 1e4, unit: '/cm^2', description: 'Pores per cm^2', advanced: true },
  membrane_porosity: { type: 'number', label: 'Membrane Porosity', min: 0.01, max: 0.5, step: 0.01, description: 'Calculated void fraction', advanced: true },

  // Inlet/Outlet Ports
  num_inlets: { type: 'number', label: 'Number of Inlets', min: 1, max: 8, step: 1, description: '1-8 typical' },
  num_outlets: { type: 'number', label: 'Number of Outlets', min: 1, max: 8, step: 1, description: '1-8 typical' },
  port_diameter_mm: { type: 'number', label: 'Port Diameter', min: 0.5, max: 3, step: 0.1, unit: 'mm', description: '1/16 inch Mini Luer standard', advanced: true },
  port_depth_mm: { type: 'number', label: 'Port Depth', min: 1, max: 5, step: 0.5, unit: 'mm', description: 'Chip thickness at ports', advanced: true },

  // Multi-layer Architecture
  num_layers: { type: 'number', label: 'Number of Layers', min: 1, max: 4, step: 1, description: '1-4 for stacked organ-on-chip', advanced: true },
  layer_height_mm: { type: 'number', label: 'Layer Height', min: 0.5, max: 3, step: 0.1, unit: 'mm', description: 'Per PDMS layer', advanced: true },
  enable_interlayer_vias: { type: 'boolean', label: 'Enable Interlayer Vias', description: 'Vertical connections between layers', advanced: true },
  via_diameter_mm: { type: 'number', label: 'Via Diameter', min: 0.1, max: 1, step: 0.05, unit: 'mm', advanced: true },

  // Flow Parameters (design)
  design_flow_rate_ul_min: { type: 'number', label: 'Design Flow Rate', min: 1, max: 100, step: 1, unit: 'uL/min', description: '1-100 uL/min typical', advanced: true },
  design_shear_stress_pa: { type: 'number', label: 'Design Shear Stress', min: 0.1, max: 10, step: 0.1, unit: 'Pa', description: '0.1-10 Pa physiological', advanced: true },

  // Cell Seeding
  cell_seeding_density_per_cm2: { type: 'number', label: 'Cell Seeding Density', min: 1e4, max: 1e6, step: 1e4, unit: '/cm^2', description: '1e4-1e6 cells/cm^2', advanced: true },
  enable_cell_traps: { type: 'boolean', label: 'Enable Cell Traps', description: 'For single-cell capture', advanced: true },
  trap_diameter_um: { type: 'number', label: 'Trap Diameter', min: 10, max: 50, step: 5, unit: 'um', description: 'For single-cell capture', advanced: true },
  trap_count: { type: 'number', label: 'Trap Count', min: 10, max: 500, step: 10, advanced: true },

  // Chip Dimensions
  chip_length_mm: { type: 'number', label: 'Chip Length', min: 10, max: 75, step: 5, unit: 'mm', description: 'Standard 25x75 microscope slide' },
  chip_width_mm: { type: 'number', label: 'Chip Width', min: 5, max: 25, step: 1, unit: 'mm' },
  chip_thickness_mm: { type: 'number', label: 'Chip Thickness', min: 1, max: 10, step: 0.5, unit: 'mm', description: 'Total chip height' },

  // Quality/Resolution
  resolution: { type: 'number', label: 'Resolution', min: 4, max: 32, step: 1, description: 'Mesh resolution', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 99999, step: 1, advanced: true },
};

export const GRADIENT_SCAFFOLD_META: Record<string, ParamMeta> = {
  // Gradient Direction and Type
  gradient_direction: {
    type: 'enum',
    label: 'Gradient Direction',
    options: [
      { value: 'linear', label: 'Linear' },
      { value: 'radial', label: 'Radial' },
      { value: 'axial', label: 'Axial' },
    ],
  },
  gradient_axis: {
    type: 'enum',
    label: 'Gradient Axis',
    options: [
      { value: 'x', label: 'X' },
      { value: 'y', label: 'Y' },
      { value: 'z', label: 'Z' },
    ],
    description: 'For linear gradient',
  },
  gradient_type: {
    type: 'enum',
    label: 'Gradient Type',
    options: [
      { value: 'linear', label: 'Linear' },
      { value: 'exponential', label: 'Exponential' },
      { value: 'sigmoid', label: 'Sigmoid' },
    ],
  },

  // Porosity Gradient
  starting_porosity: { type: 'number', label: 'Starting Porosity', min: 0.3, max: 0.95, step: 0.05, description: '0.3-0.95 typical' },
  ending_porosity: { type: 'number', label: 'Ending Porosity', min: 0.3, max: 0.95, step: 0.05, description: '0.3-0.95 typical' },
  porosity_exponent: { type: 'number', label: 'Porosity Exponent', min: 0.5, max: 3, step: 0.1, description: 'For exponential gradients', advanced: true },

  // Pore Size Gradient
  min_pore_size_um: { type: 'number', label: 'Min Pore Size', min: 50, max: 150, step: 10, unit: 'um', description: '100+ μm for vascularization' },
  max_pore_size_um: { type: 'number', label: 'Max Pore Size', min: 200, max: 500, step: 25, unit: 'um', description: '200-500 um for cartilage-like' },
  pore_shape: {
    type: 'enum',
    label: 'Pore Shape',
    options: [
      { value: 'spherical', label: 'Spherical' },
      { value: 'ellipsoidal', label: 'Ellipsoidal' },
      { value: 'cylindrical', label: 'Cylindrical' },
    ],
    advanced: true,
  },
  pore_aspect_ratio: { type: 'number', label: 'Pore Aspect Ratio', min: 0.5, max: 3, step: 0.1, description: 'For ellipsoidal pores', advanced: true },

  // Zone Architecture
  num_zones: { type: 'number', label: 'Number of Zones', min: 2, max: 5, step: 1, description: '2-5 discrete zones' },
  transition_zone_width_mm: { type: 'number', label: 'Transition Zone Width', min: 0.5, max: 3, step: 0.25, unit: 'mm', description: '0.5-3 mm transition regions', advanced: true },
  enable_discrete_zones: { type: 'boolean', label: 'Enable Discrete Zones', description: 'False = continuous gradient', advanced: true },
  zone_boundaries: { type: 'array', label: 'Zone Boundaries', itemCount: 2, itemLabels: ['Boundary 1', 'Boundary 2'], min: 0.0, max: 1.0, step: 0.05, description: 'Custom zone boundaries (normalized 0-1)', advanced: true },

  // Stiffness Gradient (design parameters)
  enable_stiffness_gradient: { type: 'boolean', label: 'Enable Stiffness Gradient', description: 'Enable stiffness variation', advanced: true },
  stiffness_gradient_min_pa: { type: 'number', label: 'Stiffness Min', min: 10000, max: 100000, step: 1000, unit: 'Pa', description: '20-50 kPa cartilage ECM', advanced: true },
  stiffness_gradient_max_pa: { type: 'number', label: 'Stiffness Max', min: 100000000, max: 500000000, step: 5000000, unit: 'Pa', description: '100-500 MPa trabecular bone', advanced: true },
  stiffness_correlation: {
    type: 'enum',
    label: 'Stiffness Correlation',
    options: [
      { value: 'direct', label: 'Direct' },
      { value: 'inverse', label: 'Inverse' },
    ],
    description: 'To porosity',
    advanced: true,
  },

  // Scaffold Geometry
  scaffold_thickness_mm: { type: 'number', label: 'Scaffold Thickness', min: 1, max: 20, step: 0.5, unit: 'mm', description: 'Total thickness along gradient' },
  scaffold_shape: {
    type: 'enum',
    label: 'Scaffold Shape',
    options: [
      { value: 'rectangular', label: 'Rectangular' },
      { value: 'cylindrical', label: 'Cylindrical' },
      { value: 'custom', label: 'Custom' },
    ],
  },
  bounding_box_mm: boundingBoxMeta,
  scaffold_diameter_mm: { type: 'number', label: 'Scaffold Diameter', min: 3, max: 30, step: 1, unit: 'mm', description: 'For cylindrical shape', advanced: true },

  // Grid and Pore Distribution
  grid_spacing_mm: { type: 'number', label: 'Grid Spacing', min: 0.5, max: 5, step: 0.25, unit: 'mm', description: 'Base spacing between pores' },
  pore_base_size_mm: { type: 'number', label: 'Pore Base Size', min: 0.1, max: 2, step: 0.1, unit: 'mm', description: 'Reference pore size', advanced: true },
  enable_stochastic_variation: { type: 'boolean', label: 'Enable Stochastic Variation', description: 'Random position/size variation', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.5, step: 0.05, description: 'Random position jitter (0-1)', advanced: true },
  size_variance: { type: 'number', label: 'Size Variance', min: 0, max: 0.5, step: 0.05, description: 'Random size variation (0-1)', advanced: true },

  // Interconnectivity
  enable_interconnections: { type: 'boolean', label: 'Enable Interconnections', description: 'Add pore connections', advanced: true },
  interconnection_diameter_ratio: { type: 'number', label: 'Interconnection Diameter Ratio', min: 0.1, max: 0.5, step: 0.05, description: 'Relative to pore size', advanced: true },
  min_interconnections_per_pore: { type: 'number', label: 'Min Interconnections/Pore', min: 1, max: 6, step: 1, advanced: true },

  // Quality/Resolution
  resolution: { type: 'number', label: 'Resolution', min: 6, max: 32, step: 2, description: 'Mesh resolution', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 99999, step: 1, advanced: true },
};

export const PERFUSABLE_NETWORK_META: Record<string, ParamMeta> = {
  // Multi-scale Vessel Hierarchy
  large_vessel_diameter_mm: { type: 'number', label: 'Large Vessel Diameter', min: 1, max: 10, step: 0.5, unit: 'mm', description: '1-10 mm for main feeding vessels' },
  arteriole_diameter_um: { type: 'number', label: 'Arteriole Diameter', min: 50, max: 300, step: 25, unit: 'um', description: '50-300 um arterioles', advanced: true },
  capillary_diameter_um: { type: 'number', label: 'Capillary Diameter', min: 5, max: 10, step: 1, unit: 'um', description: '5-10 um capillaries', advanced: true },
  venule_diameter_um: { type: 'number', label: 'Venule Diameter', min: 30, max: 200, step: 10, unit: 'um', description: '30-200 um venules', advanced: true },

  // Branching Parameters
  num_branching_generations: { type: 'number', label: 'Branching Generations', min: 3, max: 12, step: 1, description: '3-12 typical' },
  murrays_law_ratio: { type: 'number', label: "Murray's Law Ratio", min: 1.0, max: 1.5, step: 0.02, description: 'Cube root of 2 (~1.26) for equal splitting' },
  bifurcation_angle_deg: { type: 'number', label: 'Bifurcation Angle', min: 60, max: 75, step: 1, unit: 'deg', description: '60-75 deg physiological' },
  bifurcation_asymmetry: { type: 'number', label: 'Bifurcation Asymmetry', min: 0, max: 1, step: 0.1, description: '0 = symmetric, 1 = max asymmetry', advanced: true },
  enable_trifurcation: { type: 'boolean', label: 'Enable Trifurcation', description: 'Allow 3-way branching', advanced: true },

  // Capillary Network
  capillary_density_per_mm2: { type: 'number', label: 'Capillary Density', min: 500, max: 3000, step: 100, unit: '/mm^2', description: '500-3000 per mm^2 tissue-dependent', advanced: true },
  max_cell_distance_um: { type: 'number', label: 'Max Cell Distance', min: 100, max: 300, step: 25, unit: 'um', description: 'Maximum distance from any point to vessel', advanced: true },
  enable_capillary_bed: { type: 'boolean', label: 'Enable Capillary Bed', description: 'Add capillary network', advanced: true },
  capillary_tortuosity: { type: 'number', label: 'Capillary Tortuosity', min: 0, max: 1, step: 0.1, description: '0-1 waviness', advanced: true },

  // Flow Parameters
  design_flow_rate_ml_min: { type: 'number', label: 'Design Flow Rate', min: 0.1, max: 10, step: 0.1, unit: 'mL/min', description: '0.1-10 mL/min typical', advanced: true },
  target_wall_shear_stress_pa: { type: 'number', label: 'Target Wall Shear Stress', min: 0.5, max: 5, step: 0.25, unit: 'Pa', description: '0.5-5 Pa physiological', advanced: true },

  // Vessel Wall
  vessel_wall_thickness_ratio: { type: 'number', label: 'Wall Thickness Ratio', min: 0.05, max: 0.3, step: 0.01, description: 'Wall thickness as ratio of radius', advanced: true },
  enable_hollow_vessels: { type: 'boolean', label: 'Enable Hollow Vessels', description: 'Create hollow vessel lumens', advanced: true },
  min_wall_thickness_um: { type: 'number', label: 'Min Wall Thickness', min: 20, max: 50, step: 5, unit: 'um', description: '40+ μm for bioprinting feasibility', advanced: true },

  // Network Topology
  network_topology: {
    type: 'enum',
    label: 'Network Topology',
    options: [
      { value: 'hierarchical', label: 'Hierarchical' },
      { value: 'anastomosing', label: 'Anastomosing' },
      { value: 'parallel', label: 'Parallel' },
      { value: 'loop', label: 'Loop' },
    ],
    advanced: true,
  },
  enable_arterio_venous_separation: { type: 'boolean', label: 'Enable A-V Separation', description: 'Separate arterial and venous trees', advanced: true },
  anastomosis_density: { type: 'number', label: 'Anastomosis Density', min: 0, max: 1, step: 0.1, description: '0-1 for cross-connections', advanced: true },

  // Scaffold Geometry
  bounding_box_mm: boundingBoxMeta,
  network_type: {
    type: 'enum',
    label: 'Network Type',
    options: [
      { value: 'arterial', label: 'Arterial' },
      { value: 'venous', label: 'Venous' },
      { value: 'capillary', label: 'Capillary' },
      { value: 'mixed', label: 'Mixed' },
    ],
  },
  inlet_position: {
    type: 'enum',
    label: 'Inlet Position',
    options: [
      { value: 'top', label: 'Top' },
      { value: 'bottom', label: 'Bottom' },
      { value: 'side', label: 'Side' },
    ],
  },
  outlet_position: {
    type: 'enum',
    label: 'Outlet Position',
    options: [
      { value: 'top', label: 'Top' },
      { value: 'bottom', label: 'Bottom' },
      { value: 'side', label: 'Side' },
      { value: 'distributed', label: 'Distributed' },
    ],
    advanced: true,
  },

  // Organic Variation
  enable_organic_variation: { type: 'boolean', label: 'Enable Organic Variation', description: 'Natural-looking randomness', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.3, step: 0.05, description: 'Random position jitter', advanced: true },
  angle_noise: { type: 'number', label: 'Angle Noise', min: 0, max: 0.3, step: 0.05, description: 'Random angle variation', advanced: true },
  diameter_variance: { type: 'number', label: 'Diameter Variance', min: 0, max: 0.3, step: 0.05, description: 'Random diameter variation', advanced: true },

  // Quality/Resolution
  resolution: { type: 'number', label: 'Resolution', min: 6, max: 32, step: 2, description: 'Mesh resolution', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 99999, step: 1, advanced: true },
};

// Combined export for microfluidic scaffold types
export const MICROFLUIDIC_PARAMETER_META: Partial<Record<ScaffoldType, Record<string, ParamMeta>>> = {
  [ScaffoldType.ORGAN_ON_CHIP]: ORGAN_ON_CHIP_META,
  [ScaffoldType.GRADIENT_SCAFFOLD]: GRADIENT_SCAFFOLD_META,
  [ScaffoldType.PERFUSABLE_NETWORK]: PERFUSABLE_NETWORK_META,
};
