/**
 * Microfluidic Parameters
 */

import { BaseParams, GradientDirection, GradientFunctionType, NetworkType } from './base';

// ============================================================================
// Microfluidic Parameters
// ============================================================================

export interface OrganOnChipParams extends BaseParams {
  // === Main Channel Geometry ===
  /** Main channel width in micrometers (50-500 um typical) */
  main_channel_width_um?: number;
  /** Main channel height in micrometers (25-200 um typical) */
  main_channel_height_um?: number;
  /** Main channel length in millimeters (5-30 mm typical) */
  main_channel_length_mm?: number;
  /** @deprecated Legacy param - use main_channel_width_um */
  channel_width_mm?: number;
  /** @deprecated Legacy param - use main_channel_height_um */
  channel_depth_mm?: number;
  /** @deprecated Legacy param - use chamber_width_mm etc */
  chamber_size_mm?: { x: number; y: number; z: number };

  // === Microchannel Network ===
  /** Microchannel width in micrometers (50-200 um) */
  microchannel_width_um?: number;
  /** Microchannel height in micrometers (20-100 um typical) */
  microchannel_height_um?: number;
  /** Enable microchannel network */
  enable_microchannels?: boolean;
  /** Number of microchannels per chamber */
  microchannel_count?: number;
  /** Center-to-center microchannel spacing in micrometers */
  microchannel_spacing_um?: number;

  // === Chamber Configuration ===
  /** Chamber width in millimeters (1-20 mm typical) */
  chamber_width_mm?: number;
  /** Chamber height in micrometers (25-150 um) */
  chamber_height_um?: number;
  /** Chamber length in millimeters (2-15 mm typical) */
  chamber_length_mm?: number;
  /** Chamber volume in nanoliters (~500 nL typical) */
  chamber_volume_nl?: number;
  /** Number of chambers (1-96 for high-throughput) */
  num_chambers?: number;
  /** @deprecated Legacy alias for num_chambers */
  chamber_count?: number;
  /** Chamber spacing in millimeters */
  chamber_spacing_mm?: number;

  // === Membrane Configuration ===
  /** Enable membrane between chambers */
  enable_membrane?: boolean;
  /** Membrane thickness in micrometers (5-50 um) */
  membrane_thickness_um?: number;
  /** Membrane pore size in micrometers (0.1-8.0 um) */
  membrane_pore_size_um?: number;
  /** Membrane pore density per cm^2 */
  membrane_pore_density_per_cm2?: number;
  /** Membrane porosity (0-1) */
  membrane_porosity?: number;

  // === Inlet/Outlet Ports ===
  /** Number of inlets (1-8 typical) */
  num_inlets?: number;
  /** @deprecated Legacy alias for num_inlets */
  inlet_count?: number;
  /** Number of outlets (1-8 typical) */
  num_outlets?: number;
  /** Port diameter in millimeters (standard Luer fitting) */
  port_diameter_mm?: number;
  /** Port depth in millimeters */
  port_depth_mm?: number;

  // === Multi-layer Architecture ===
  /** Number of layers (1-4 for stacked) */
  num_layers?: number;
  /** Layer height in millimeters */
  layer_height_mm?: number;
  /** Enable interlayer vias */
  enable_interlayer_vias?: boolean;
  /** Via diameter in millimeters */
  via_diameter_mm?: number;

  // === Flow Parameters ===
  /** Design flow rate in uL/min (1-100 uL/min typical) */
  design_flow_rate_ul_min?: number;
  /** Design shear stress in Pa (0.1-10 Pa physiological) */
  design_shear_stress_pa?: number;

  // === Cell Seeding ===
  /** Cell seeding density per cm^2 (1e4-1e6 cells/cm^2) */
  cell_seeding_density_per_cm2?: number;
  /** Enable cell traps */
  enable_cell_traps?: boolean;
  /** Trap diameter in micrometers (for single-cell capture) */
  trap_diameter_um?: number;
  /** Number of traps */
  trap_count?: number;

  // === Chip Dimensions ===
  /** Chip length in millimeters */
  chip_length_mm?: number;
  /** Chip width in millimeters */
  chip_width_mm?: number;
  /** Chip thickness in millimeters */
  chip_thickness_mm?: number;
  /** @deprecated Legacy param - use chip_length_mm etc */
  chip_size_mm?: { x: number; y: number; z: number };

  // === Quality/Resolution ===
  /** Random seed for reproducibility */
  seed?: number;
}

export interface GradientScaffoldParams extends BaseParams {
  // === Gradient Direction and Type ===
  /** Gradient direction: 'linear' | 'radial' | 'axial' */
  gradient_direction?: GradientDirection;
  /** Axis for linear gradient */
  gradient_axis?: 'x' | 'y' | 'z';
  /** Gradient function type: 'linear' | 'exponential' | 'sigmoid' */
  gradient_type?: GradientFunctionType;

  // === Porosity Gradient ===
  /** Starting porosity (0.3-0.95 typical) */
  starting_porosity?: number;
  /** @deprecated Legacy alias for starting_porosity */
  start_porosity?: number;
  /** Ending porosity (0.3-0.95 typical) */
  ending_porosity?: number;
  /** @deprecated Legacy alias for ending_porosity */
  end_porosity?: number;
  /** Exponent for exponential gradients */
  porosity_exponent?: number;

  // === Pore Size Gradient ===
  /** Minimum pore size in micrometers (50-150 um for bone-like) */
  min_pore_size_um?: number;
  /** Maximum pore size in micrometers (200-500 um for cartilage-like) */
  max_pore_size_um?: number;
  /** Pore shape */
  pore_shape?: 'spherical' | 'ellipsoidal' | 'cylindrical';
  /** Pore aspect ratio (for ellipsoidal pores) */
  pore_aspect_ratio?: number;
  /** Reference pore size in millimeters */
  pore_base_size_mm?: number;

  // === Zone Architecture ===
  /** Number of discrete zones (2-5) */
  num_zones?: number;
  /** Transition zone width in millimeters (0.5-3 mm) */
  transition_zone_width_mm?: number;
  /** Enable discrete zones */
  enable_discrete_zones?: boolean;
  /** Custom zone boundaries (normalized 0-1) */
  zone_boundaries?: number[];

  // === Stiffness Gradient ===
  /** Enable stiffness gradient */
  enable_stiffness_gradient?: boolean;
  /** Minimum stiffness in Pa (1-50 kPa cartilage-like) */
  stiffness_gradient_min_pa?: number;
  /** Maximum stiffness in Pa (10-200 kPa bone-like) */
  stiffness_gradient_max_pa?: number;
  /** Stiffness correlation: 'direct' | 'inverse' */
  stiffness_correlation?: 'direct' | 'inverse';

  // === Scaffold Geometry ===
  /** Total thickness along gradient in millimeters */
  scaffold_thickness_mm?: number;
  /** Scaffold shape */
  scaffold_shape?: 'rectangular' | 'cylindrical' | 'custom';
  /** Bounding box dimensions in millimeters */
  bounding_box_mm?: { x: number; y: number; z: number };
  /** @deprecated Legacy alias for bounding_box_mm */
  dimensions_mm?: { x: number; y: number; z: number };
  /** Scaffold diameter for cylindrical shape in millimeters */
  scaffold_diameter_mm?: number;
  /** Base spacing between pores in millimeters */
  grid_spacing_mm?: number;

  // === Stochastic Variation ===
  /** Enable stochastic variation */
  enable_stochastic_variation?: boolean;
  /** Random position jitter (0-1) */
  position_noise?: number;
  /** Random size variation (0-1) */
  size_variance?: number;

  // === Interconnectivity ===
  /** Enable interconnections between pores */
  enable_interconnections?: boolean;
  /** Interconnection diameter relative to pore size */
  interconnection_diameter_ratio?: number;
  /** Minimum interconnections per pore */
  min_interconnections_per_pore?: number;

  // === Quality/Resolution ===
  /** Random seed for reproducibility */
  seed?: number;
}

export interface PerfusableNetworkParams extends BaseParams {
  // === Multi-scale Vessel Hierarchy ===
  /** Large vessel diameter in millimeters (1-10 mm) */
  large_vessel_diameter_mm?: number;
  /** @deprecated Legacy alias for large_vessel_diameter_mm */
  inlet_diameter_mm?: number;
  /** Arteriole diameter in micrometers (50-300 um) */
  arteriole_diameter_um?: number;
  /** Capillary diameter in micrometers (5-10 um) */
  capillary_diameter_um?: number;
  /** Venule diameter in micrometers (30-200 um) */
  venule_diameter_um?: number;

  // === Branching Parameters ===
  /** Number of branching generations (3-12 typical) */
  num_branching_generations?: number;
  /** @deprecated Legacy alias for num_branching_generations */
  branch_generations?: number;
  /** Murray's law ratio (~1.26 for equal splitting) */
  murrays_law_ratio?: number;
  /** @deprecated Legacy alias for murrays_law_ratio */
  murray_ratio?: number;
  /** Bifurcation angle in degrees (60-75 deg physiological) */
  bifurcation_angle_deg?: number;
  /** @deprecated Legacy alias for bifurcation_angle_deg */
  branching_angle_deg?: number;
  /** Bifurcation asymmetry (0-1) */
  bifurcation_asymmetry?: number;
  /** Enable trifurcation */
  enable_trifurcation?: boolean;

  // === Capillary Network ===
  /** Capillary density per mm^2 (500-3000 per mm^2) */
  capillary_density_per_mm2?: number;
  /** Maximum distance from vessel in micrometers */
  max_cell_distance_um?: number;
  /** Enable capillary bed */
  enable_capillary_bed?: boolean;
  /** Capillary tortuosity/waviness (0-1) */
  capillary_tortuosity?: number;

  // === Flow Parameters ===
  /** Design flow rate in mL/min (0.1-10 mL/min typical) */
  design_flow_rate_ml_min?: number;
  /** Target wall shear stress in Pa (0.5-5 Pa physiological) */
  target_wall_shear_stress_pa?: number;

  // === Vessel Wall ===
  /** Vessel wall thickness ratio */
  vessel_wall_thickness_ratio?: number;
  /** Enable hollow vessels */
  enable_hollow_vessels?: boolean;
  /** Minimum wall thickness in micrometers */
  min_wall_thickness_um?: number;

  // === Network Topology ===
  /** Network topology: 'hierarchical' | 'anastomosing' | 'parallel' | 'loop' */
  network_topology?: 'hierarchical' | 'anastomosing' | 'parallel' | 'loop';
  /** Network type: 'arterial' | 'venous' | 'capillary' | 'mixed' */
  network_type?: NetworkType;
  /** Enable arterio-venous separation */
  enable_arterio_venous_separation?: boolean;
  /** Anastomosis density for cross-connections (0-1) */
  anastomosis_density?: number;

  // === Scaffold Geometry ===
  /** Bounding box dimensions in millimeters */
  bounding_box_mm?: { x: number; y: number; z: number };
  /** Inlet position */
  inlet_position?: 'top' | 'bottom' | 'side';
  /** Outlet position */
  outlet_position?: 'top' | 'bottom' | 'side' | 'distributed';

  // === Organic Variation ===
  /** Enable organic variation */
  enable_organic_variation?: boolean;
  /** Position noise (0-1) */
  position_noise?: number;
  /** Angle noise (0-1) */
  angle_noise?: number;
  /** Diameter variance (0-1) */
  diameter_variance?: number;

  // === Quality/Resolution ===
  /** Random seed for reproducibility */
  seed?: number;
}

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULT_ORGAN_ON_CHIP: OrganOnChipParams = {
  resolution: 8,
  // === Main Channel Geometry ===
  main_channel_width_um: 200.0,
  main_channel_height_um: 100.0,
  main_channel_length_mm: 10.0,
  channel_width_mm: 0.3,
  channel_depth_mm: 0.1,
  chamber_size_mm: { x: 3.0, y: 2.0, z: 0.15 },

  // === Microchannel Network ===
  microchannel_width_um: 100.0,
  microchannel_height_um: 50.0,
  enable_microchannels: false,
  microchannel_count: 8,
  microchannel_spacing_um: 150.0,

  // === Chamber Configuration ===
  chamber_width_mm: 5.0,
  chamber_height_um: 100.0,
  chamber_length_mm: 5.0,
  chamber_volume_nl: 500.0,
  num_chambers: 2,
  chamber_count: 2,
  chamber_spacing_mm: 3.0,

  // === Membrane Configuration ===
  enable_membrane: false,
  membrane_thickness_um: 10.0,
  membrane_pore_size_um: 0.4,
  membrane_pore_density_per_cm2: 1e6,
  membrane_porosity: 0.01,

  // === Inlet/Outlet Ports ===
  num_inlets: 2,
  inlet_count: 2,
  num_outlets: 2,
  port_diameter_mm: 1.6,
  port_depth_mm: 2.0,

  // === Multi-layer Architecture ===
  num_layers: 1,
  layer_height_mm: 2.0,
  enable_interlayer_vias: false,
  via_diameter_mm: 0.5,

  // === Flow Parameters ===
  design_flow_rate_ul_min: 10.0,
  design_shear_stress_pa: 1.0,

  // === Cell Seeding ===
  cell_seeding_density_per_cm2: 1e5,
  enable_cell_traps: false,
  trap_diameter_um: 20.0,
  trap_count: 0,

  // === Chip Dimensions ===
  chip_length_mm: 15.0,
  chip_width_mm: 10.0,
  chip_thickness_mm: 2.0,
  chip_size_mm: { x: 15.0, y: 10.0, z: 2.0 },

  // === Quality/Resolution ===
  seed: 42,
};

export const DEFAULT_GRADIENT_SCAFFOLD: GradientScaffoldParams = {
  resolution: 12,
  // === Gradient Direction and Type ===
  gradient_direction: GradientDirection.Z,
  gradient_axis: 'z',
  gradient_type: GradientFunctionType.LINEAR,

  // === Porosity Gradient ===
  starting_porosity: 0.2,
  start_porosity: 0.2,
  ending_porosity: 0.8,
  end_porosity: 0.8,
  porosity_exponent: 2.0,

  // === Pore Size Gradient ===
  min_pore_size_um: 100.0,
  max_pore_size_um: 400.0,
  pore_shape: 'spherical',
  pore_aspect_ratio: 1.0,
  pore_base_size_mm: 0.5,

  // === Zone Architecture ===
  num_zones: 3,
  transition_zone_width_mm: 1.0,
  enable_discrete_zones: false,
  zone_boundaries: [0.33, 0.67],

  // === Stiffness Gradient ===
  enable_stiffness_gradient: false,
  stiffness_gradient_min_pa: 30000.0,
  stiffness_gradient_max_pa: 300000000.0,
  stiffness_correlation: 'direct',

  // === Scaffold Geometry ===
  scaffold_thickness_mm: 10.0,
  scaffold_shape: 'rectangular',
  bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
  dimensions_mm: { x: 10.0, y: 10.0, z: 10.0 },
  scaffold_diameter_mm: 10.0,
  grid_spacing_mm: 1.5,

  // === Stochastic Variation ===
  enable_stochastic_variation: false,
  position_noise: 0.0,
  size_variance: 0.0,

  // === Interconnectivity ===
  enable_interconnections: true,
  interconnection_diameter_ratio: 0.3,
  min_interconnections_per_pore: 3,

  // === Quality/Resolution ===
  seed: 42,
};

export const DEFAULT_PERFUSABLE_NETWORK: PerfusableNetworkParams = {
  resolution: 12,
  // === Multi-scale Vessel Hierarchy ===
  large_vessel_diameter_mm: 2.0,
  inlet_diameter_mm: 2.0,
  arteriole_diameter_um: 100.0,
  capillary_diameter_um: 8.0,
  venule_diameter_um: 80.0,

  // === Branching Parameters ===
  num_branching_generations: 3,
  branch_generations: 3,
  murrays_law_ratio: 0.79,
  murray_ratio: 0.79,
  bifurcation_angle_deg: 30.0,
  branching_angle_deg: 30.0,
  bifurcation_asymmetry: 0.0,
  enable_trifurcation: false,

  // === Capillary Network ===
  capillary_density_per_mm2: 1000.0,
  max_cell_distance_um: 200.0,
  enable_capillary_bed: false,
  capillary_tortuosity: 0.0,

  // === Flow Parameters ===
  design_flow_rate_ml_min: 1.0,
  target_wall_shear_stress_pa: 1.5,

  // === Vessel Wall ===
  vessel_wall_thickness_ratio: 0.1,
  enable_hollow_vessels: false,
  min_wall_thickness_um: 40.0,

  // === Network Topology ===
  network_topology: 'hierarchical',
  network_type: NetworkType.ARTERIAL,
  enable_arterio_venous_separation: false,
  anastomosis_density: 0.0,

  // === Scaffold Geometry ===
  bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
  inlet_position: 'top',
  outlet_position: 'bottom',

  // === Organic Variation ===
  enable_organic_variation: false,
  position_noise: 0.0,
  angle_noise: 0.0,
  diameter_variance: 0.0,

  // === Quality/Resolution ===
  seed: 42,
};
