/**
 * Skeletal Tissue Parameters
 */

import { BaseParams, GradientType, OsteonPattern } from './base';

// ============================================================================
// Skeletal Tissue Parameters
// ============================================================================

export interface TrabecularBoneParams extends BaseParams {
  // Basic geometry
  bounding_box: { x: number; y: number; z: number };
  seed: number;

  // Pore characteristics
  pore_size_um: number;
  pore_size_min_um?: number;
  pore_size_max_um?: number;
  pore_size_variance?: number;

  // Trabecular architecture
  trabecular_thickness_um?: number;
  trabecular_spacing_um?: number;
  strut_thickness_um: number;
  strut_diameter_um?: number;

  // Porosity and volume metrics
  porosity: number;
  bone_volume_fraction?: number;
  connectivity_density?: number;

  // Structural anisotropy
  anisotropy_ratio: number;
  degree_anisotropy?: number;
  primary_orientation_deg?: number;
  fabric_tensor_eigenratio?: number;

  // Randomization
  randomization_factor?: number;
  position_noise?: number;
  thickness_variance?: number;
  connectivity_variance?: number;

  // Rod vs plate morphology
  structure_model_index?: number;
  enable_plate_like_regions?: boolean;
  plate_fraction?: number;

  // Surface characteristics
  surface_roughness?: number;
  enable_resorption_pits?: boolean;
  resorption_pit_density?: number;

  // Gradient options
  enable_density_gradient?: boolean;
  gradient_direction?: 'x' | 'y' | 'z';
  gradient_start_porosity?: number;
  gradient_end_porosity?: number;
}

export interface OsteochondralParams extends BaseParams {
  // Basic geometry
  diameter: number;
  seed?: number;

  // Zone depths (mm)
  cartilage_depth: number;
  calcified_cartilage_depth?: number;
  subchondral_plate_depth?: number;
  bone_depth: number;
  transition_width: number;

  // Cartilage zone ratios
  superficial_zone_ratio?: number;
  middle_zone_ratio?: number;
  deep_zone_ratio?: number;

  // Porosity per zone
  cartilage_porosity: number;
  calcified_cartilage_porosity?: number;
  subchondral_plate_porosity?: number;
  bone_porosity: number;

  // Pore size gradients (mm)
  superficial_pore_size?: number;
  middle_pore_size?: number;
  deep_pore_size?: number;
  bone_pore_size?: number;

  // Gradient and transition properties
  gradient_type: GradientType;
  gradient_sharpness?: number;

  // Tidemark and cement line features
  enable_tidemark?: boolean;
  tidemark_thickness?: number;
  enable_cement_line?: boolean;
  cement_line_thickness?: number;

  // Collagen fiber orientation by zone
  superficial_fiber_angle_deg?: number;
  deep_fiber_angle_deg?: number;

  // Channel features
  enable_vascular_channels?: boolean;
  vascular_channel_diameter?: number;
  vascular_channel_spacing?: number;

  // Subchondral bone properties
  subchondral_trabecular_thickness?: number;
  subchondral_connectivity?: number;

  // Mechanical property indicators
  enable_stiffness_gradient?: boolean;
  cartilage_modulus_indicator?: number;
  bone_modulus_indicator?: number;

  // Randomization
  position_noise?: number;
  pore_size_variance?: number;
}

export interface ArticularCartilageParams extends BaseParams {
  // Basic geometry
  diameter: number;
  seed?: number;

  // Overall thickness
  total_thickness: number;
  total_thickness_um?: number;

  // Zone thickness ratios (must sum to 1.0)
  superficial_thickness_ratio?: number;
  middle_thickness_ratio?: number;
  deep_thickness_ratio?: number;
  zone_ratios: [number, number, number];

  // Collagen fiber orientation per zone (degrees from surface)
  superficial_fiber_orientation_deg?: number;
  middle_fiber_orientation_deg?: number;
  deep_fiber_orientation_deg?: number;
  fiber_orientation_variance_deg?: number;

  // Pore characteristics per zone (mm)
  superficial_pore_size?: number;
  middle_pore_size?: number;
  deep_pore_size?: number;
  pore_gradient: [number, number, number];

  // Porosity per zone
  superficial_porosity?: number;
  middle_porosity?: number;
  deep_porosity?: number;

  // Cell density per zone (cells per mm3)
  superficial_cell_density?: number;
  middle_cell_density?: number;
  deep_cell_density?: number;

  // Proteoglycan content indicators (relative units)
  superficial_proteoglycan?: number;
  middle_proteoglycan?: number;
  deep_proteoglycan?: number;

  // Collagen fiber properties
  collagen_fiber_diameter_um?: number;
  collagen_fiber_spacing_um?: number;
  enable_fiber_bundles?: boolean;

  // Channel features for nutrient diffusion
  enable_vertical_channels?: boolean;
  vertical_channel_diameter?: number;
  vertical_channel_spacing?: number;
  enable_horizontal_channels?: boolean;

  // Surface characteristics
  surface_roughness?: number;
  enable_surface_texture?: boolean;

  // Tidemark (calcified cartilage interface)
  enable_tidemark_layer?: boolean;
  tidemark_thickness?: number;
  tidemark_porosity?: number;

  // Randomization and organic appearance
  position_noise?: number;
  pore_size_variance?: number;
  zone_boundary_blur?: number;
}

export interface MeniscusParams extends BaseParams {
  // Basic geometry (mm)
  outer_radius: number;
  inner_radius: number;
  outer_diameter?: number;
  inner_diameter?: number;
  thickness?: number;
  height: number;
  arc_degrees: number;
  seed?: number;

  // Wedge geometry
  wedge_angle_deg: number;
  inner_edge_height_ratio?: number;

  // Radial zones (from inner to outer)
  zone_count: number;
  outer_zone_thickness_ratio?: number;
  middle_zone_thickness_ratio?: number;
  inner_zone_thickness_ratio?: number;

  // Circumferential fiber properties (main load-bearing fibers)
  circumferential_bundle_diameter_um?: number;
  circumferential_bundle_spacing_um?: number;
  circumferential_fiber_density?: number;

  // Radial (tie) fiber properties
  radial_bundle_diameter_um?: number;
  radial_bundle_spacing_um?: number;
  radial_fiber_density?: number;
  enable_radial_tie_fibers?: boolean;

  // General fiber properties
  fiber_diameter: number;
  fiber_orientation_variance_deg?: number;

  // Porosity per zone
  outer_zone_porosity?: number;
  middle_zone_porosity?: number;
  inner_zone_porosity?: number;
  porosity?: number;

  // Vascular features (outer 1/3 is vascular)
  enable_vascular_channels?: boolean;
  vascular_channel_diameter?: number;
  vascular_channel_spacing?: number;
  vascular_penetration_depth?: number;

  // Horn attachments
  enable_anterior_horn?: boolean;
  enable_posterior_horn?: boolean;
  horn_insertion_width?: number;
  horn_insertion_depth?: number;

  // Surface properties
  surface_roughness?: number;
  enable_femoral_surface?: boolean;
  enable_tibial_surface?: boolean;
  femoral_curvature_radius?: number;

  // Collagen organization
  enable_lamellar_structure?: boolean;
  lamella_count?: number;
  interlaminar_spacing_um?: number;

  // Cell lacunae (for biological appearance)
  enable_cell_lacunae?: boolean;
  lacuna_density?: number;
  lacuna_diameter_um?: number;

  // Randomization
  position_noise?: number;
  thickness_variance?: number;
}

export interface TendonLigamentParams extends BaseParams {
  // Basic geometry (mm)
  length: number;
  width: number;
  thickness: number;
  seed?: number;

  // Fascicle organization (highest level bundles)
  bundle_count: number;
  fascicle_diameter?: number;
  fascicle_spacing?: number;
  enable_fascicle_boundaries?: boolean;

  // Fiber properties (within fascicles)
  fiber_diameter: number;
  fiber_diameter_um?: number;
  fiber_spacing: number;
  fiber_spacing_um?: number;
  fibers_per_fascicle?: number;

  // Fibril properties (within fibers)
  fibril_diameter_um?: number;
  fibril_spacing_um?: number;
  enable_fibril_detail?: boolean;

  // Crimp pattern (sinusoidal wave)
  crimp_amplitude: number;
  crimp_amplitude_um?: number;
  crimp_wavelength: number;
  crimp_wavelength_um?: number;
  crimp_angle_deg?: number;
  crimp_variance?: number;

  // Endotenon (interfascicular matrix)
  enable_endotenon?: boolean;
  endotenon_thickness?: number;
  endotenon_porosity?: number;

  // Epitenon (outer sheath)
  enable_epitenon?: boolean;
  epitenon_thickness?: number;
  epitenon_porosity?: number;

  // Paratenon (sliding layer)
  enable_paratenon?: boolean;
  paratenon_thickness?: number;

  // Collagen alignment
  primary_fiber_angle_deg?: number;
  fiber_angle_variance_deg?: number;
  enable_cross_links?: boolean;
  cross_link_density?: number;

  // Vascular features
  enable_vascular_channels?: boolean;
  vascular_channel_diameter?: number;
  vascular_channel_spacing?: number;
  vascular_channel_pattern?: 'longitudinal' | 'spiral';

  // Insertion zones (enthesis)
  enable_enthesis_transition?: boolean;
  enthesis_length?: number;
  enthesis_mineralization_gradient?: number;

  // Mechanical property indicators
  target_stiffness_mpa?: number;
  target_ultimate_stress_mpa?: number;

  // Porosity
  porosity?: number;
  pore_size_um?: number;

  // Surface characteristics
  surface_roughness?: number;
  enable_surface_texture?: boolean;

  // Randomization
  position_noise?: number;
  diameter_variance?: number;
  spacing_variance?: number;
}

export interface IntervertebralDiscParams extends BaseParams {
  // Basic geometry (mm)
  disc_diameter: number;
  disc_height: number;
  seed?: number;

  // Nucleus pulposus (NP) - central gel-like region
  np_diameter: number;
  nucleus_percentage?: number;
  np_height_ratio?: number;
  np_porosity: number;
  np_water_content?: number;
  np_proteoglycan_content?: number;
  np_pore_size?: number;

  // Annulus fibrosus (AF) - outer fibrous ring
  af_ring_count: number;
  num_lamellae?: number;
  annulus_percentage?: number;
  af_porosity?: number;
  af_water_content?: number;

  // Fiber properties
  fiber_diameter: number;
  fiber_diameter_um?: number;
  fiber_angle?: number;
  af_layer_angle: number;
  inner_af_fiber_angle_deg?: number;
  outer_af_fiber_angle_deg?: number;
  fiber_angle_variance_deg?: number;

  // Lamella properties
  lamella_thickness?: number;
  interlaminar_spacing?: number;
  enable_interlaminar_connections?: boolean;

  // Mechanical property indicators
  np_stiffness_kpa?: number;
  af_stiffness_kpa?: number;
  stiffness_gradient?: boolean;

  // Cartilaginous endplates
  enable_endplates?: boolean;
  endplate_thickness?: number;
  endplate_porosity?: number;
  endplate_pore_size?: number;

  // Nutrient pathway features
  enable_nutrient_channels?: boolean;
  nutrient_channel_diameter?: number;
  nutrient_channel_density?: number;
  enable_endplate_pores?: boolean;

  // Transition zone (NP-AF interface)
  transition_zone_width?: number;
  transition_gradient_type?: 'linear' | 'sigmoid' | 'step';

  // Vascular features (outer AF only)
  enable_outer_vascular?: boolean;
  vascular_channel_diameter?: number;
  vascular_penetration_depth?: number;

  // Notochordal remnants
  enable_notochordal_cells?: boolean;
  notochordal_region_radius?: number;

  // Age-related changes
  degeneration_level?: number;
  enable_fissures?: boolean;
  fissure_count?: number;

  // Surface properties
  surface_roughness?: number;

  // Randomization
  position_noise?: number;
  fiber_variance?: number;
  lamella_variance?: number;
}

export interface HaversianBoneParams extends BaseParams {
  // Basic geometry
  bounding_box: { x: number; y: number; z: number };
  seed?: number;

  // Haversian canal (central vascular channel)
  haversian_canal_diameter_um?: number;
  canal_diameter_um: number;
  canal_wall_thickness_um?: number;
  enable_haversian_vessels?: boolean;

  // Osteon (Haversian system) properties
  osteon_diameter_um?: number;
  canal_spacing_um: number;
  osteon_spacing_um?: number;
  osteon_pattern: OsteonPattern;

  // Volkmann's canals (transverse/oblique connecting canals)
  enable_volkmann_canals?: boolean;
  volkmann_canal_diameter_um?: number;
  volkmann_canal_density?: number;
  volkmann_angle_deg?: number;

  // Concentric lamellae (rings around Haversian canal)
  num_concentric_lamellae?: number;
  lamella_thickness_um?: number;
  interlamellar_spacing_um?: number;
  enable_lamellar_detail?: boolean;

  // Lacunae (osteocyte spaces)
  enable_lacunae?: boolean;
  lacuna_density?: number;
  lacuna_length_um?: number;
  lacuna_width_um?: number;

  // Canaliculi (connecting channels from lacunae)
  enable_canaliculi?: boolean;
  canaliculus_diameter_um?: number;
  canaliculi_per_lacuna?: number;

  // Cement line (osteon boundary)
  enable_cement_line?: boolean;
  cement_line_thickness_um?: number;

  // Interstitial lamellae (between osteons)
  enable_interstitial_lamellae?: boolean;
  interstitial_fraction?: number;

  // Cortical bone properties
  cortical_thickness: number;
  cortical_porosity?: number;
  bone_mineral_density?: number;

  // Periosteal and endosteal surfaces
  enable_periosteal_surface?: boolean;
  enable_endosteal_surface?: boolean;
  periosteal_layer_thickness_um?: number;
  endosteal_layer_thickness_um?: number;

  // Remodeling features
  enable_resorption_spaces?: boolean;
  resorption_space_diameter_um?: number;
  resorption_density?: number;

  // Mechanical orientation
  primary_orientation_deg?: number;
  orientation_variance_deg?: number;

  // Surface characteristics
  surface_roughness?: number;

  // Randomization
  position_noise?: number;
  diameter_variance?: number;
  lamella_count_variance?: number;
}

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULT_TRABECULAR_BONE: TrabecularBoneParams = {
  resolution: 8,
  // Basic geometry
  bounding_box: { x: 10.0, y: 10.0, z: 10.0 },
  seed: 42,
  // Pore characteristics
  pore_size_um: 400.0,
  pore_size_min_um: 100.0,
  pore_size_max_um: 600.0,
  pore_size_variance: 0.2,
  // Trabecular architecture
  trabecular_thickness_um: 200.0,
  trabecular_spacing_um: 500.0,
  strut_thickness_um: 200.0,
  strut_diameter_um: 250.0,
  // Porosity and volume metrics
  porosity: 0.75,
  bone_volume_fraction: 0.25,
  connectivity_density: 5.0,
  // Structural anisotropy
  anisotropy_ratio: 1.5,
  degree_anisotropy: 1.8,
  primary_orientation_deg: 0.0,
  fabric_tensor_eigenratio: 1.5,
  // Randomization
  randomization_factor: 0.2,
  position_noise: 0.3,
  thickness_variance: 0.15,
  connectivity_variance: 0.2,
  // Rod vs plate morphology
  structure_model_index: 1.5,
  enable_plate_like_regions: true,
  plate_fraction: 0.3,
  // Surface characteristics
  surface_roughness: 0.1,
  enable_resorption_pits: false,
  resorption_pit_density: 0.05,
  // Gradient options
  enable_density_gradient: false,
  gradient_direction: 'z',
  gradient_start_porosity: 0.6,
  gradient_end_porosity: 0.9,
};

export const DEFAULT_OSTEOCHONDRAL: OsteochondralParams = {
  resolution: 16,
  // Basic geometry
  diameter: 8.0,
  seed: 42,
  // Zone depths
  cartilage_depth: 2.5,
  calcified_cartilage_depth: 0.2,
  subchondral_plate_depth: 0.3,
  bone_depth: 3.0,
  transition_width: 1.0,
  // Cartilage zone ratios
  superficial_zone_ratio: 0.15,
  middle_zone_ratio: 0.50,
  deep_zone_ratio: 0.35,
  // Porosity per zone
  cartilage_porosity: 0.85,
  calcified_cartilage_porosity: 0.65,
  subchondral_plate_porosity: 0.10,
  bone_porosity: 0.70,
  // Pore size gradients
  superficial_pore_size: 0.15,
  middle_pore_size: 0.25,
  deep_pore_size: 0.35,
  bone_pore_size: 0.50,
  // Gradient properties
  gradient_type: GradientType.LINEAR,
  gradient_sharpness: 1.0,
  // Tidemark and cement line
  enable_tidemark: true,
  tidemark_thickness: 0.01,
  enable_cement_line: true,
  cement_line_thickness: 0.005,
  // Fiber orientation
  superficial_fiber_angle_deg: 0.0,
  deep_fiber_angle_deg: 90.0,
  // Channel features
  enable_vascular_channels: true,
  vascular_channel_diameter: 0.1,
  vascular_channel_spacing: 0.8,
  // Subchondral bone properties
  subchondral_trabecular_thickness: 0.15,
  subchondral_connectivity: 0.8,
  // Mechanical indicators
  enable_stiffness_gradient: false,
  cartilage_modulus_indicator: 1.0,
  bone_modulus_indicator: 20.0,
  // Randomization
  position_noise: 0.1,
  pore_size_variance: 0.15,
};

export const DEFAULT_ARTICULAR_CARTILAGE: ArticularCartilageParams = {
  resolution: 16,
  // Basic geometry
  diameter: 8.0,
  seed: 42,
  // Overall thickness
  total_thickness: 2.5,
  total_thickness_um: 2500.0,
  // Zone thickness ratios
  superficial_thickness_ratio: 0.15,
  middle_thickness_ratio: 0.50,
  deep_thickness_ratio: 0.35,
  zone_ratios: [0.15, 0.50, 0.35],
  // Collagen fiber orientation
  superficial_fiber_orientation_deg: 0.0,
  middle_fiber_orientation_deg: 45.0,
  deep_fiber_orientation_deg: 90.0,
  fiber_orientation_variance_deg: 15.0,
  // Pore characteristics
  superficial_pore_size: 0.10,
  middle_pore_size: 0.20,
  deep_pore_size: 0.30,
  pore_gradient: [0.10, 0.20, 0.30],
  // Porosity
  superficial_porosity: 0.70,
  middle_porosity: 0.80,
  deep_porosity: 0.85,
  // Cell density
  superficial_cell_density: 10000.0,
  middle_cell_density: 5000.0,
  deep_cell_density: 3000.0,
  // Proteoglycan content
  superficial_proteoglycan: 0.3,
  middle_proteoglycan: 0.7,
  deep_proteoglycan: 1.0,
  // Collagen fibers
  collagen_fiber_diameter_um: 50.0,
  collagen_fiber_spacing_um: 100.0,
  enable_fiber_bundles: true,
  // Channels
  enable_vertical_channels: true,
  vertical_channel_diameter: 0.15,
  vertical_channel_spacing: 0.5,
  enable_horizontal_channels: false,
  // Surface
  surface_roughness: 0.05,
  enable_surface_texture: true,
  // Tidemark
  enable_tidemark_layer: false,
  tidemark_thickness: 0.1,
  tidemark_porosity: 0.60,
  // Randomization
  position_noise: 0.15,
  pore_size_variance: 0.2,
  zone_boundary_blur: 0.1,
};

export const DEFAULT_MENISCUS: MeniscusParams = {
  resolution: 32,
  // Basic geometry
  outer_radius: 22.5,
  inner_radius: 10.0,
  outer_diameter: 45.0,
  inner_diameter: 20.0,
  thickness: 8.0,
  height: 8.0,
  arc_degrees: 300.0,
  seed: 42,
  // Wedge geometry
  wedge_angle_deg: 20.0,
  inner_edge_height_ratio: 0.3,
  // Radial zones
  zone_count: 3,
  outer_zone_thickness_ratio: 0.33,
  middle_zone_thickness_ratio: 0.33,
  inner_zone_thickness_ratio: 0.34,
  // Circumferential fibers
  circumferential_bundle_diameter_um: 100.0,
  circumferential_bundle_spacing_um: 200.0,
  circumferential_fiber_density: 0.8,
  // Radial fibers
  radial_bundle_diameter_um: 20.0,
  radial_bundle_spacing_um: 100.0,
  radial_fiber_density: 0.6,
  enable_radial_tie_fibers: true,
  // General fiber properties
  fiber_diameter: 0.2,
  fiber_orientation_variance_deg: 15.0,
  // Porosity
  outer_zone_porosity: 0.65,
  middle_zone_porosity: 0.55,
  inner_zone_porosity: 0.45,
  porosity: 0.61,
  // Vascular features
  enable_vascular_channels: true,
  vascular_channel_diameter: 0.1,
  vascular_channel_spacing: 0.8,
  vascular_penetration_depth: 0.33,
  // Horn attachments
  enable_anterior_horn: false,
  enable_posterior_horn: false,
  horn_insertion_width: 5.0,
  horn_insertion_depth: 3.0,
  // Surface properties
  surface_roughness: 0.05,
  enable_femoral_surface: true,
  enable_tibial_surface: true,
  femoral_curvature_radius: 30.0,
  // Collagen organization
  enable_lamellar_structure: true,
  lamella_count: 6,
  interlaminar_spacing_um: 50.0,
  // Cell lacunae
  enable_cell_lacunae: false,
  lacuna_density: 500.0,
  lacuna_diameter_um: 15.0,
  // Randomization
  position_noise: 0.1,
  thickness_variance: 0.1,
};

export const DEFAULT_TENDON_LIGAMENT: TendonLigamentParams = {
  resolution: 12,
  // Basic geometry
  length: 30.0,
  width: 8.0,
  thickness: 3.0,
  seed: 42,
  // Fascicle organization
  bundle_count: 5,
  fascicle_diameter: 1.5,
  fascicle_spacing: 0.3,
  enable_fascicle_boundaries: true,
  // Fiber properties
  fiber_diameter: 0.15,
  fiber_diameter_um: 150.0,
  fiber_spacing: 0.4,
  fiber_spacing_um: 400.0,
  fibers_per_fascicle: 20,
  // Fibril properties
  fibril_diameter_um: 5.0,
  fibril_spacing_um: 10.0,
  enable_fibril_detail: false,
  // Crimp pattern
  crimp_amplitude: 0.3,
  crimp_amplitude_um: 300.0,
  crimp_wavelength: 2.0,
  crimp_wavelength_um: 300.0,
  crimp_angle_deg: 10.0,
  crimp_variance: 0.1,
  // Endotenon
  enable_endotenon: true,
  endotenon_thickness: 0.05,
  endotenon_porosity: 0.5,
  // Epitenon
  enable_epitenon: true,
  epitenon_thickness: 0.1,
  epitenon_porosity: 0.3,
  // Paratenon
  enable_paratenon: false,
  paratenon_thickness: 0.2,
  // Collagen alignment
  primary_fiber_angle_deg: 0.0,
  fiber_angle_variance_deg: 5.0,
  enable_cross_links: true,
  cross_link_density: 0.1,
  // Vascular features
  enable_vascular_channels: true,
  vascular_channel_diameter: 0.08,
  vascular_channel_spacing: 1.5,
  vascular_channel_pattern: 'longitudinal',
  // Enthesis
  enable_enthesis_transition: false,
  enthesis_length: 3.0,
  enthesis_mineralization_gradient: 0.5,
  // Mechanical indicators
  target_stiffness_mpa: 1000.0,
  target_ultimate_stress_mpa: 100.0,
  // Porosity
  porosity: 0.15,
  pore_size_um: 50.0,
  // Surface
  surface_roughness: 0.02,
  enable_surface_texture: true,
  // Randomization
  position_noise: 0.05,
  diameter_variance: 0.1,
  spacing_variance: 0.1,
};

export const DEFAULT_INTERVERTEBRAL_DISC: IntervertebralDiscParams = {
  resolution: 16,
  // Basic geometry
  disc_diameter: 40.0,
  disc_height: 8.5,
  seed: 42,
  // Nucleus pulposus
  np_diameter: 16.0,
  nucleus_percentage: 0.40,
  np_height_ratio: 0.85,
  np_porosity: 0.90,
  np_water_content: 0.80,
  np_proteoglycan_content: 0.8,
  np_pore_size: 0.3,
  // Annulus fibrosus
  af_ring_count: 3,
  num_lamellae: 20,
  annulus_percentage: 0.60,
  af_porosity: 0.65,
  af_water_content: 0.60,
  // Fiber properties
  fiber_diameter: 0.2,
  fiber_diameter_um: 200.0,
  fiber_angle: 30.0,
  af_layer_angle: 30.0,
  inner_af_fiber_angle_deg: 45.0,
  outer_af_fiber_angle_deg: 65.0,
  fiber_angle_variance_deg: 5.0,
  // Lamella properties
  lamella_thickness: 0.15,
  interlaminar_spacing: 0.02,
  enable_interlaminar_connections: true,
  // Mechanical indicators
  np_stiffness_kpa: 2.0,
  af_stiffness_kpa: 100.0,
  stiffness_gradient: true,
  // Endplates
  enable_endplates: true,
  endplate_thickness: 0.7,
  endplate_porosity: 0.55,
  endplate_pore_size: 0.1,
  // Nutrient pathways
  enable_nutrient_channels: true,
  nutrient_channel_diameter: 0.05,
  nutrient_channel_density: 0.3,
  enable_endplate_pores: true,
  // Transition zone
  transition_zone_width: 0.5,
  transition_gradient_type: 'linear',
  // Vascular
  enable_outer_vascular: true,
  vascular_channel_diameter: 0.1,
  vascular_penetration_depth: 0.2,
  // Notochordal
  enable_notochordal_cells: false,
  notochordal_region_radius: 2.0,
  // Age-related
  degeneration_level: 0.0,
  enable_fissures: false,
  fissure_count: 0,
  // Surface
  surface_roughness: 0.05,
  // Randomization
  position_noise: 0.1,
  fiber_variance: 0.1,
  lamella_variance: 0.05,
};

export const DEFAULT_HAVERSIAN_BONE: HaversianBoneParams = {
  resolution: 8,
  // Basic geometry
  bounding_box: { x: 5.0, y: 5.0, z: 8.0 },
  seed: 42,
  // Haversian canal
  haversian_canal_diameter_um: 50.0,
  canal_diameter_um: 50.0,
  canal_wall_thickness_um: 5.0,
  enable_haversian_vessels: true,
  // Osteon properties
  osteon_diameter_um: 225.0,
  canal_spacing_um: 400.0,
  osteon_spacing_um: 250.0,
  osteon_pattern: OsteonPattern.HEXAGONAL,
  // Volkmann's canals
  enable_volkmann_canals: true,
  volkmann_canal_diameter_um: 75.0,
  volkmann_canal_density: 0.5,
  volkmann_angle_deg: 60.0,
  // Concentric lamellae
  num_concentric_lamellae: 8,
  lamella_thickness_um: 5.0,
  interlamellar_spacing_um: 1.0,
  enable_lamellar_detail: true,
  // Lacunae
  enable_lacunae: true,
  lacuna_density: 15000.0,
  lacuna_length_um: 20.0,
  lacuna_width_um: 8.0,
  // Canaliculi
  enable_canaliculi: false,
  canaliculus_diameter_um: 0.5,
  canaliculi_per_lacuna: 50,
  // Cement line
  enable_cement_line: true,
  cement_line_thickness_um: 3.0,
  // Interstitial lamellae
  enable_interstitial_lamellae: true,
  interstitial_fraction: 0.2,
  // Cortical properties
  cortical_thickness: 5.0,
  cortical_porosity: 0.08,
  bone_mineral_density: 1.8,
  // Surfaces
  enable_periosteal_surface: true,
  enable_endosteal_surface: true,
  periosteal_layer_thickness_um: 50.0,
  endosteal_layer_thickness_um: 30.0,
  // Remodeling
  enable_resorption_spaces: false,
  resorption_space_diameter_um: 200.0,
  resorption_density: 0.02,
  // Orientation
  primary_orientation_deg: 0.0,
  orientation_variance_deg: 10.0,
  // Surface
  surface_roughness: 0.05,
  // Randomization
  position_noise: 0.1,
  diameter_variance: 0.15,
  lamella_count_variance: 2,
};
