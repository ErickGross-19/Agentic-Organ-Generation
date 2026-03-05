/**
 * Organ-Specific Parameters
 */

import { BaseParams } from './base';

// ============================================================================
// Organ-Specific Parameters
// ============================================================================

export interface HepaticLobuleParams extends BaseParams {
  // === Basic Geometry ===
  num_lobules: number;
  lobule_radius: number;
  lobule_height: number;
  wall_thickness: number;

  // === Central Vein ===
  central_vein_radius: number;
  cv_entrance_length?: number;
  cv_entrance_radius?: number;
  cv_entrance_count?: number;
  cv_entrance_z_randomness?: number;
  cv_entrance_min_spacing?: number;

  // === Portal Triad ===
  portal_vein_radius: number;
  hepatic_artery_radius?: number;
  bile_duct_radius?: number;
  show_bile_ducts?: boolean;
  triad_wall_distance?: number;
  entrance_length?: number;
  entrance_radius?: number;
  entrance_count?: number;
  entrance_z_randomness?: number;
  entrance_min_spacing?: number;

  // === Sinusoids ===
  show_sinusoids: boolean;
  sinusoid_radius: number;
  sinusoid_count?: number;
  sinusoid_levels?: number;

  // === Organic Variation ===
  corner_noise?: number;
  angle_noise?: number;
  stretch_noise?: number;
  size_variance?: number;
  edge_curve?: number;
  seed?: number;

  // === Collectors ===
  show_hepatic_collector?: boolean;
  show_portal_collector?: boolean;
  hepatic_collector_height?: number;
  portal_collector_height?: number;
}

export interface CardiacPatchParams extends BaseParams {
  // === Cardiomyocyte Dimensions ===
  cardiomyocyte_length?: number;
  cardiomyocyte_diameter?: number;
  sarcomere_length?: number;

  // === Fiber Architecture ===
  fiber_diameter: number;
  fiber_spacing: number;
  fiber_alignment_degree?: number;

  // === Layer Configuration ===
  epicardial_helix_angle?: number;
  endocardial_helix_angle?: number;
  layer_count: number;
  enable_helix_gradient?: boolean;

  // === Patch Dimensions ===
  patch_length?: number;
  patch_width?: number;
  patch_thickness?: number;
  /** @deprecated Use patch_length, patch_width, patch_thickness instead */
  patch_size?: { x: number; y: number; z: number };
  /** @deprecated Use epicardial_helix_angle instead */
  alignment_angle?: number;

  // === Porosity ===
  porosity?: number;
  pore_size?: number;
  enable_interconnected_pores?: boolean;
  pore_interconnectivity?: number;

  // === Vascularization Features ===
  enable_capillary_channels?: boolean;
  capillary_diameter?: number;
  capillary_density?: number;
  capillary_spacing?: number;

  // === Mechanical Features ===
  enable_cross_fibers?: boolean;
  cross_fiber_ratio?: number;
  enable_fiber_crimping?: boolean;
  crimp_amplitude?: number;
  crimp_wavelength?: number;

  // === Electrical Guidance ===
  enable_conduction_channels?: boolean;
  conduction_channel_diameter?: number;
  conduction_channel_spacing?: number;

  // === Stochastic Variation ===
  position_noise?: number;
  diameter_variance?: number;
  angle_noise?: number;
  seed?: number;
}

export interface KidneyTubuleParams extends BaseParams {
  // === Tubule Geometry ===
  tubule_outer_diameter?: number;
  tubule_inner_diameter?: number;
  tubule_length?: number;
  /** @deprecated Use tubule_outer_diameter instead */
  tubule_diameter?: number;
  /** @deprecated Use tubule_length instead */
  length?: number;
  /** @deprecated Calculated from outer-inner diameter */
  wall_thickness?: number;

  // === Epithelial Features ===
  epithelial_cell_height?: number;
  microvilli_height?: number;
  enable_brush_border_texture?: boolean;
  brush_border_density?: number;

  // === Basement Membrane ===
  enable_basement_membrane?: boolean;
  basement_membrane_thickness?: number;

  // === Convolution Parameters ===
  convolution_amplitude: number;
  convolution_frequency: number;
  convolution_phase_xy?: number;
  enable_3d_convolution?: boolean;

  // === Scaffold Structure ===
  scaffold_porosity?: number;
  wall_porosity?: number;
  pore_size?: number;

  // === Peritubular Capillaries ===
  enable_peritubular_capillaries?: boolean;
  capillary_diameter?: number;
  capillary_spacing?: number;
  capillary_wrap_angle?: number;

  // === Segment Types ===
  tubule_segment_type?: 'proximal' | 'distal' | 'collecting' | 'loop_of_henle';
  enable_segment_transitions?: boolean;
  transition_length?: number;

  // === Tubule Network ===
  enable_branching?: boolean;
  branch_count?: number;
  branch_angle?: number;
  branch_diameter_ratio?: number;

  // === Cell Attachment Features ===
  enable_cell_attachment_sites?: boolean;
  attachment_site_diameter?: number;
  attachment_site_spacing?: number;
  attachment_site_depth?: number;

  // === Stochastic Variation ===
  diameter_variance?: number;
  position_noise?: number;
  seed?: number;
}

export interface LungAlveoliParams extends BaseParams {
  // === Alveolar Geometry ===
  alveolar_diameter?: number;
  alveolar_wall_thickness?: number;
  alveolar_depth?: number;
  alveoli_per_duct?: number;
  /** @deprecated Use alveolar_diameter instead */
  alveoli_diameter?: number;

  // === Airway Tree ===
  total_branching_generations?: number;
  scaffold_generations?: number;
  /** @deprecated Use scaffold_generations instead */
  branch_generations?: number;
  terminal_bronchiole_diameter?: number;
  airway_diameter: number;
  branch_angle: number;
  branch_length_ratio?: number;

  // === Murray's Law Branching ===
  branching_ratio?: number;
  enable_asymmetric_branching?: boolean;
  asymmetry_factor?: number;

  // === Porosity and Structure ===
  porosity?: number;
  pore_interconnectivity?: number;
  enable_pores_of_kohn?: boolean;
  pore_of_kohn_diameter?: number;
  pores_per_alveolus?: number;

  // === Blood-Air Barrier ===
  enable_blood_air_barrier?: boolean;
  type_1_cell_coverage?: number;
  type_2_cell_coverage?: number;
  type_2_cell_diameter?: number;

  // === Capillary Network ===
  enable_capillary_network?: boolean;
  capillary_diameter?: number;
  capillary_density?: number;
  capillary_spacing?: number;

  // === Alveolar Duct Features ===
  enable_alveolar_ducts?: boolean;
  alveolar_duct_length?: number;
  alveolar_duct_diameter?: number;

  // === Scaffold Bounding ===
  bounding_box: { x: number; y: number; z: number };

  // === Surfactant Layer ===
  surfactant_layer_thickness?: number;

  // === Stochastic Variation ===
  size_variance?: number;
  position_noise?: number;
  angle_noise?: number;
  seed?: number;
}

export interface PancreaticIsletParams extends BaseParams {
  // === Islet Geometry ===
  islet_diameter: number;
  islet_count?: number;
  islet_spacing?: number;
  /** @deprecated Use islet_count instead */
  cluster_count?: number;
  /** @deprecated Use islet_spacing instead */
  spacing?: number;

  // === Cell Type Fractions ===
  beta_cell_fraction?: number;
  alpha_cell_fraction?: number;
  delta_cell_fraction?: number;
  pp_cell_fraction?: number;
  enable_cell_distribution_markers?: boolean;

  // === Encapsulation Capsule ===
  enable_capsule?: boolean;
  capsule_thickness?: number;
  capsule_diameter?: number;
  capsule_porosity?: number;
  capsule_pore_size?: number;

  // === Islet Shell Structure ===
  shell_thickness: number;
  core_porosity?: number;
  shell_porosity?: number;
  enable_core_shell_architecture?: boolean;

  // === Porosity and Diffusion ===
  pore_size: number;
  pore_density?: number;
  oxygen_diffusion_coefficient?: number;
  glucose_diffusion_coefficient?: number;

  // === Vascularization ===
  enable_vascular_channels?: boolean;
  vascular_channel_diameter?: number;
  vascular_channel_count?: number;
  vascular_channel_pattern?: 'radial' | 'parallel';

  // === Viability Support ===
  islet_viability?: number;
  max_diffusion_distance?: number;

  // === ECM Components ===
  enable_ecm_coating?: boolean;
  ecm_thickness?: number;
  ecm_type?: 'collagen' | 'laminin' | 'matrigel';

  // === Multi-islet Configuration ===
  cluster_pattern?: 'hexagonal' | 'random' | 'linear';
  enable_inter_islet_connections?: boolean;
  connection_diameter?: number;

  // === Stochastic Variation ===
  size_variance?: number;
  position_noise?: number;
  seed?: number;
}

export interface LiverSinusoidParams extends BaseParams {
  // === Sinusoid Geometry ===
  sinusoid_diameter: number;
  sinusoid_length?: number;
  sinusoid_wall_thickness?: number;
  /** @deprecated Use sinusoid_length instead */
  length?: number;

  // === Fenestration Parameters ===
  fenestration_diameter?: number;
  fenestration_density: number;
  fenestration_pattern?: 'clustered' | 'random';
  sieve_plate_count?: number;
  fenestration_porosity?: number;
  /** @deprecated Use fenestration_diameter instead */
  fenestration_size?: number;

  // === Space of Disse ===
  enable_space_of_disse?: boolean;
  space_of_disse_thickness?: number;
  space_of_disse_porosity?: number;

  // === Hepatocyte Zone ===
  enable_hepatocyte_zone?: boolean;
  hepatocyte_diameter?: number;
  hepatocyte_spacing?: number;
  hepatocytes_per_sinusoid?: number;

  // === Cell Type Zones ===
  enable_kupffer_cell_zones?: boolean;
  kupffer_cell_density?: number;
  enable_stellate_cell_zones?: boolean;
  stellate_cell_spacing?: number;

  // === Sinusoid Network ===
  sinusoid_count?: number;
  sinusoid_spacing?: number;
  network_pattern?: 'parallel' | 'branching' | 'anastomosing';
  enable_central_vein_connection?: boolean;
  central_vein_diameter?: number;

  // === Scaffold Structure ===
  scaffold_length?: number;
  enable_scaffold_shell?: boolean;
  shell_thickness?: number;
  shell_porosity?: number;

  // === Bile Canaliculi ===
  enable_bile_canaliculi?: boolean;
  canaliculus_diameter?: number;
  canaliculus_spacing?: number;

  // === ECM Components ===
  enable_ecm_fibers?: boolean;
  ecm_fiber_diameter?: number;
  ecm_fiber_density?: number;

  // === Stochastic Variation ===
  diameter_variance?: number;
  fenestration_variance?: number;
  position_noise?: number;
  seed?: number;
}

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULT_HEPATIC_LOBULE: HepaticLobuleParams = {
  resolution: 8,
  // === Basic Geometry ===
  num_lobules: 7,
  lobule_radius: 1.5,
  lobule_height: 3.0,
  wall_thickness: 0.1,
  // === Central Vein ===
  central_vein_radius: 0.15,
  cv_entrance_length: 0.25,
  cv_entrance_radius: 0.04,
  cv_entrance_count: 5,
  cv_entrance_z_randomness: 0.0,
  cv_entrance_min_spacing: 0.1,
  // === Portal Triad ===
  portal_vein_radius: 0.12,
  hepatic_artery_radius: 0.06,
  bile_duct_radius: 0.05,
  show_bile_ducts: true,
  triad_wall_distance: 0.0,
  entrance_length: 0.25,
  entrance_radius: 0.04,
  entrance_count: 5,
  entrance_z_randomness: 0.0,
  entrance_min_spacing: 0.1,
  // === Sinusoids ===
  show_sinusoids: false,
  sinusoid_radius: 0.012,
  sinusoid_count: 6,
  sinusoid_levels: 3,
  // === Organic Variation ===
  corner_noise: 0.0,
  angle_noise: 0.0,
  stretch_noise: 0.0,
  size_variance: 0.0,
  edge_curve: 0.0,
  seed: 42,
  // === Collectors ===
  show_hepatic_collector: false,
  show_portal_collector: false,
  hepatic_collector_height: 2.0,
  portal_collector_height: 2.0,
};

export const DEFAULT_CARDIAC_PATCH: CardiacPatchParams = {
  resolution: 8,
  // === Cardiomyocyte Dimensions ===
  cardiomyocyte_length: 100.0,
  cardiomyocyte_diameter: 25.0,
  sarcomere_length: 1.8,
  // === Fiber Architecture ===
  fiber_diameter: 80.0,
  fiber_spacing: 150.0,
  fiber_alignment_degree: 0.9,
  // === Layer Configuration ===
  epicardial_helix_angle: -45.0,
  endocardial_helix_angle: 45.0,
  layer_count: 3,
  enable_helix_gradient: true,
  // === Patch Dimensions ===
  patch_length: 10.0,
  patch_width: 10.0,
  patch_thickness: 2.0,
  patch_size: { x: 10.0, y: 10.0, z: 2.0 },
  alignment_angle: -60.0,
  // === Porosity ===
  porosity: 0.75,
  pore_size: 50.0,
  enable_interconnected_pores: true,
  pore_interconnectivity: 0.85,
  // === Vascularization Features ===
  enable_capillary_channels: true,
  capillary_diameter: 8.0,
  capillary_density: 3000.0,
  capillary_spacing: 20.0,
  // === Mechanical Features ===
  enable_cross_fibers: true,
  cross_fiber_ratio: 0.3,
  enable_fiber_crimping: false,
  crimp_amplitude: 5.0,
  crimp_wavelength: 50.0,
  // === Electrical Guidance ===
  enable_conduction_channels: false,
  conduction_channel_diameter: 100.0,
  conduction_channel_spacing: 500.0,
  // === Stochastic Variation ===
  position_noise: 0.0,
  diameter_variance: 0.0,
  angle_noise: 0.0,
  seed: 42,
};

export const DEFAULT_KIDNEY_TUBULE: KidneyTubuleParams = {
  resolution: 12,
  // === Tubule Geometry ===
  tubule_outer_diameter: 500.0,
  tubule_inner_diameter: 350.0,
  tubule_length: 15.0,
  tubule_diameter: 500.0,
  length: 15.0,
  wall_thickness: 75.0,
  // === Epithelial Features ===
  epithelial_cell_height: 20.0,
  microvilli_height: 2.7,
  enable_brush_border_texture: false,
  brush_border_density: 0.8,
  // === Basement Membrane ===
  enable_basement_membrane: true,
  basement_membrane_thickness: 0.3,
  // === Convolution Parameters ===
  convolution_amplitude: 200.0,
  convolution_frequency: 2.0,
  convolution_phase_xy: 0.0,
  enable_3d_convolution: true,
  // === Scaffold Structure ===
  scaffold_porosity: 0.7,
  wall_porosity: 0.3,
  pore_size: 10.0,
  // === Peritubular Capillaries ===
  enable_peritubular_capillaries: false,
  capillary_diameter: 8.0,
  capillary_spacing: 50.0,
  capillary_wrap_angle: 180.0,
  // === Segment Types ===
  tubule_segment_type: 'proximal',
  enable_segment_transitions: false,
  transition_length: 0.5,
  // === Tubule Network ===
  enable_branching: false,
  branch_count: 0,
  branch_angle: 45.0,
  branch_diameter_ratio: 0.7,
  // === Cell Attachment Features ===
  enable_cell_attachment_sites: false,
  attachment_site_diameter: 5.0,
  attachment_site_spacing: 20.0,
  attachment_site_depth: 2.0,
  // === Stochastic Variation ===
  diameter_variance: 0.0,
  position_noise: 0.0,
  seed: 42,
};

export const DEFAULT_LUNG_ALVEOLI: LungAlveoliParams = {
  resolution: 10,
  // === Alveolar Geometry ===
  alveolar_diameter: 200.0,
  alveolar_wall_thickness: 0.6,
  alveolar_depth: 150.0,
  alveoli_per_duct: 6,
  alveoli_diameter: 200.0,
  // === Airway Tree ===
  total_branching_generations: 23,
  scaffold_generations: 3,
  branch_generations: 3,
  terminal_bronchiole_diameter: 750.0,
  airway_diameter: 1.5,
  branch_angle: 35.0,
  branch_length_ratio: 0.8,
  // === Murray's Law Branching ===
  branching_ratio: 0.79,
  enable_asymmetric_branching: false,
  asymmetry_factor: 0.1,
  // === Porosity and Structure ===
  porosity: 0.85,
  pore_interconnectivity: 0.95,
  enable_pores_of_kohn: true,
  pore_of_kohn_diameter: 10.0,
  pores_per_alveolus: 3,
  // === Blood-Air Barrier ===
  enable_blood_air_barrier: false,
  type_1_cell_coverage: 0.95,
  type_2_cell_coverage: 0.06,
  type_2_cell_diameter: 9.0,
  // === Capillary Network ===
  enable_capillary_network: false,
  capillary_diameter: 8.0,
  capillary_density: 2800.0,
  capillary_spacing: 15.0,
  // === Alveolar Duct Features ===
  enable_alveolar_ducts: true,
  alveolar_duct_length: 1.0,
  alveolar_duct_diameter: 400.0,
  // === Scaffold Bounding ===
  bounding_box: { x: 8.0, y: 8.0, z: 8.0 },
  // === Surfactant Layer ===
  surfactant_layer_thickness: 0.15,
  // === Stochastic Variation ===
  size_variance: 0.0,
  position_noise: 0.0,
  angle_noise: 0.0,
  seed: 42,
};

export const DEFAULT_PANCREATIC_ISLET: PancreaticIsletParams = {
  resolution: 12,
  // === Islet Geometry ===
  islet_diameter: 150.0,
  islet_count: 3,
  islet_spacing: 300.0,
  cluster_count: 3,
  spacing: 300.0,
  // === Cell Type Fractions ===
  beta_cell_fraction: 0.50,
  alpha_cell_fraction: 0.35,
  delta_cell_fraction: 0.07,
  pp_cell_fraction: 0.02,
  enable_cell_distribution_markers: false,
  // === Encapsulation Capsule ===
  enable_capsule: true,
  capsule_thickness: 100.0,
  capsule_diameter: 500.0,
  capsule_porosity: 0.4,
  capsule_pore_size: 12.0,
  // === Islet Shell Structure ===
  shell_thickness: 40.0,
  core_porosity: 0.6,
  shell_porosity: 0.5,
  enable_core_shell_architecture: true,
  // === Porosity and Diffusion ===
  pore_size: 20.0,
  pore_density: 0.1,
  oxygen_diffusion_coefficient: 2.0e-9,
  glucose_diffusion_coefficient: 6.7e-10,
  // === Vascularization ===
  enable_vascular_channels: false,
  vascular_channel_diameter: 20.0,
  vascular_channel_count: 4,
  vascular_channel_pattern: 'radial',
  // === Viability Support ===
  islet_viability: 0.85,
  max_diffusion_distance: 150.0,
  // === ECM Components ===
  enable_ecm_coating: false,
  ecm_thickness: 5.0,
  ecm_type: 'collagen',
  // === Multi-islet Configuration ===
  cluster_pattern: 'hexagonal',
  enable_inter_islet_connections: false,
  connection_diameter: 30.0,
  // === Stochastic Variation ===
  size_variance: 0.0,
  position_noise: 0.0,
  seed: 42,
};

export const DEFAULT_LIVER_SINUSOID: LiverSinusoidParams = {
  resolution: 12,
  // === Sinusoid Geometry ===
  sinusoid_diameter: 12.0,
  sinusoid_length: 250.0,
  sinusoid_wall_thickness: 0.5,
  length: 250.0,
  // === Fenestration Parameters ===
  fenestration_diameter: 150.0,
  fenestration_density: 10.0,
  fenestration_pattern: 'clustered',
  sieve_plate_count: 8,
  fenestration_porosity: 0.06,
  fenestration_size: 150.0,
  // === Space of Disse ===
  enable_space_of_disse: true,
  space_of_disse_thickness: 0.5,
  space_of_disse_porosity: 0.8,
  // === Hepatocyte Zone ===
  enable_hepatocyte_zone: false,
  hepatocyte_diameter: 25.0,
  hepatocyte_spacing: 5.0,
  hepatocytes_per_sinusoid: 12,
  // === Cell Type Zones ===
  enable_kupffer_cell_zones: false,
  kupffer_cell_density: 0.15,
  enable_stellate_cell_zones: false,
  stellate_cell_spacing: 50.0,
  // === Sinusoid Network ===
  sinusoid_count: 1,
  sinusoid_spacing: 50.0,
  network_pattern: 'parallel',
  enable_central_vein_connection: false,
  central_vein_diameter: 30.0,
  // === Scaffold Structure ===
  scaffold_length: 2.0,
  enable_scaffold_shell: false,
  shell_thickness: 50.0,
  shell_porosity: 0.5,
  // === Bile Canaliculi ===
  enable_bile_canaliculi: false,
  canaliculus_diameter: 1.0,
  canaliculus_spacing: 25.0,
  // === ECM Components ===
  enable_ecm_fibers: false,
  ecm_fiber_diameter: 0.12,
  ecm_fiber_density: 0.2,
  // === Stochastic Variation ===
  diameter_variance: 0.0,
  fenestration_variance: 0.0,
  position_noise: 0.0,
  seed: 42,
};
