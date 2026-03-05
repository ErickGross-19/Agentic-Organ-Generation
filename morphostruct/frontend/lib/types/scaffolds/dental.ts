/**
 * Dental/Craniofacial Parameters
 */

import { BaseParams, CurveType } from './base';

// ============================================================================
// Dental/Craniofacial Parameters
// ============================================================================

export interface DentinPulpParams extends BaseParams {
  // === Basic Geometry ===
  tooth_height: number;
  crown_diameter: number;
  crown_height?: number;
  root_length: number;
  root_diameter: number;

  // === Dentin Tubule Parameters (Microstructure) ===
  tubule_diameter_dej?: number;
  tubule_diameter_pulp?: number;
  tubule_density_dej?: number;
  tubule_density_pulp?: number;
  enable_tubule_representation?: boolean;
  tubule_resolution?: number;

  // === Pulp Chamber ===
  pulp_chamber_height?: number;
  pulp_chamber_width?: number;
  pulp_chamber_size: number;
  pulp_horn_count?: number;
  pulp_horn_height?: number;
  root_canal_taper?: number;

  // === Dentin-Enamel Junction (DEJ) ===
  dej_scallop_size?: number;
  dej_scallop_count?: number;
  dej_width?: number;
  enable_dej_texture?: boolean;

  // === Dentin Layers ===
  dentin_thickness: number;
  peritubular_dentin_ratio?: number;
  intertubular_dentin_ratio?: number;
  mantle_dentin_thickness?: number;

  // === Enamel Interface ===
  enamel_interface_roughness?: number;
  enamel_thickness_occlusal?: number;
  enamel_thickness_cervical?: number;
  enable_enamel_shell?: boolean;

  // === Root Features ===
  root_count?: number;
  root_furcation_height?: number;
  cementum_thickness?: number;
  enable_cementum?: boolean;

  // === Generation Settings ===
  seed?: number;
  randomness?: number;
  detail_level?: 'low' | 'medium' | 'high';
}

export interface EarAuricleParams extends BaseParams {
  // === Overall Dimensions ===
  overall_height?: number;
  overall_width?: number;
  overall_depth?: number;
  scale_factor: number;

  // === Cartilage Framework ===
  cartilage_thickness?: number;
  skin_thickness?: number;
  thickness: number;

  // === Structural Elements ===
  strut_width?: number;
  strut_spacing?: number;
  pore_size?: number;
  pore_shape?: 'circular' | 'hexagonal';

  // === Helix (Outer Rim) ===
  helix_definition: number;
  helix_curvature?: number;
  helix_width?: number;
  helix_thickness_factor?: number;

  // === Antihelix (Inner Ridge) ===
  antihelix_depth: number;
  antihelix_curvature?: number;
  antihelix_bifurcation?: boolean;
  crura_angle?: number;

  // === Concha (Bowl) ===
  concha_depth?: number;
  concha_diameter?: number;
  cymba_conchae_ratio?: number;

  // === Tragus and Antitragus ===
  tragus_width?: number;
  tragus_height?: number;
  tragus_projection?: number;
  antitragus_size?: number;

  // === Lobule (Earlobe) ===
  lobule_length?: number;
  lobule_width?: number;
  lobule_thickness?: number;
  lobule_attached?: boolean;

  // === Mechanical Properties ===
  mechanical_modulus_ratio?: number;
  target_porosity?: number;

  // === Surface Features ===
  enable_surface_texture?: boolean;
  texture_roughness?: number;

  // === Generation Settings ===
  seed?: number;
  randomness?: number;
  detail_level?: 'low' | 'medium' | 'high';
}

export interface NasalSeptumParams extends BaseParams {
  // === Thickness Profile (varies across septum) ===
  thickness_min?: number;
  thickness_max?: number;
  thickness_base?: number;
  thickness_mid?: number;
  thickness_anterior?: number;
  thickness: number;
  enable_thickness_gradient?: boolean;

  // === Overall Dimensions ===
  height: number;
  length: number;
  surface_area_target?: number;
  width_superior?: number;
  width_inferior?: number;

  // === Quadrangular Cartilage Shape ===
  anterior_height?: number;
  posterior_height?: number;
  dorsal_length?: number;
  basal_length?: number;

  // === Curvature and Deviation ===
  curvature_radius: number;
  curvature_secondary?: number;
  curve_type: CurveType;
  deviation_angle?: number;
  deviation_location?: number;

  // === Cartilage Properties ===
  cartilage_porosity?: number;
  pore_size?: number;
  enable_porous_structure?: boolean;

  // === Cell Seeding Ratios ===
  cell_ratio_adsc_chondrocyte?: number;
  enable_cell_guidance_channels?: boolean;

  // === Three-Layer Architecture ===
  three_layer_structure?: boolean;
  perichondrium_thickness?: number;
  core_cartilage_ratio?: number;

  // === Surface Features ===
  enable_mucosal_texture?: boolean;
  mucosal_groove_depth?: number;
  enable_vascular_channels?: boolean;
  vascular_channel_diameter?: number;
  vascular_channel_spacing?: number;

  // === Edges and Margins ===
  edge_rounding?: number;
  enable_suture_holes?: boolean;
  suture_hole_diameter?: number;
  suture_hole_spacing?: number;

  // === Generation Settings ===
  seed?: number;
  randomness?: number;
  detail_level?: 'low' | 'medium' | 'high';
}

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULT_DENTIN_PULP: DentinPulpParams = {
  resolution: 16,
  // === Basic Geometry ===
  tooth_height: 10.0,
  crown_diameter: 7.0,
  crown_height: 5.0,
  root_length: 12.0,
  root_diameter: 3.0,

  // === Dentin Tubule Parameters (Microstructure) ===
  tubule_diameter_dej: 1.0,
  tubule_diameter_pulp: 3.7,
  tubule_density_dej: 1.9e6,
  tubule_density_pulp: 5.75e4,
  enable_tubule_representation: false,
  tubule_resolution: 4,

  // === Pulp Chamber ===
  pulp_chamber_height: 2.76,
  pulp_chamber_width: 2.5,
  pulp_chamber_size: 0.4,
  pulp_horn_count: 2,
  pulp_horn_height: 0.5,
  root_canal_taper: 0.06,

  // === Dentin-Enamel Junction (DEJ) ===
  dej_scallop_size: 75.0,
  dej_scallop_count: 12,
  dej_width: 9.0,
  enable_dej_texture: true,

  // === Dentin Layers ===
  dentin_thickness: 2.0,
  peritubular_dentin_ratio: 0.15,
  intertubular_dentin_ratio: 0.85,
  mantle_dentin_thickness: 0.025,

  // === Enamel Interface ===
  enamel_interface_roughness: 0.5,
  enamel_thickness_occlusal: 2.5,
  enamel_thickness_cervical: 0.5,
  enable_enamel_shell: false,

  // === Root Features ===
  root_count: 1,
  root_furcation_height: 3.0,
  cementum_thickness: 0.1,
  enable_cementum: false,

  // === Generation Settings ===
  seed: 42,
  randomness: 0.1,
  detail_level: 'medium',
};

export const DEFAULT_EAR_AURICLE: EarAuricleParams = {
  resolution: 16,
  // === Overall Dimensions ===
  overall_height: 60.0,
  overall_width: 35.0,
  overall_depth: 20.0,
  scale_factor: 1.0,

  // === Cartilage Framework ===
  cartilage_thickness: 1.5,
  skin_thickness: 1.0,
  thickness: 2.0,

  // === Structural Elements ===
  strut_width: 0.5,
  strut_spacing: 1.5,
  pore_size: 0.2,
  pore_shape: 'circular',

  // === Helix (Outer Rim) ===
  helix_definition: 0.7,
  helix_curvature: 0.6,
  helix_width: 8.0,
  helix_thickness_factor: 1.2,

  // === Antihelix (Inner Ridge) ===
  antihelix_depth: 0.3,
  antihelix_curvature: 0.5,
  antihelix_bifurcation: true,
  crura_angle: 30.0,

  // === Concha (Bowl) ===
  concha_depth: 15.0,
  concha_diameter: 20.0,
  cymba_conchae_ratio: 0.4,

  // === Tragus and Antitragus ===
  tragus_width: 8.0,
  tragus_height: 10.0,
  tragus_projection: 5.0,
  antitragus_size: 6.0,

  // === Lobule (Earlobe) ===
  lobule_length: 18.0,
  lobule_width: 15.0,
  lobule_thickness: 4.0,
  lobule_attached: false,

  // === Mechanical Properties ===
  mechanical_modulus_ratio: 0.5,
  target_porosity: 0.7,

  // === Surface Features ===
  enable_surface_texture: true,
  texture_roughness: 0.3,

  // === Generation Settings ===
  seed: 42,
  randomness: 0.1,
  detail_level: 'medium',
};

export const DEFAULT_NASAL_SEPTUM: NasalSeptumParams = {
  resolution: 16,
  // === Thickness Profile ===
  thickness_min: 0.77,
  thickness_max: 3.03,
  thickness_base: 2.6,
  thickness_mid: 0.85,
  thickness_anterior: 1.5,
  thickness: 1.5,
  enable_thickness_gradient: true,

  // === Overall Dimensions ===
  height: 31.0,
  length: 30.0,
  surface_area_target: 8.18,
  width_superior: 25.0,
  width_inferior: 35.0,

  // === Quadrangular Cartilage Shape ===
  anterior_height: 28.0,
  posterior_height: 20.0,
  dorsal_length: 25.0,
  basal_length: 30.0,

  // === Curvature and Deviation ===
  curvature_radius: 75.0,
  curvature_secondary: 150.0,
  curve_type: CurveType.SINGLE,
  deviation_angle: 5.0,
  deviation_location: 0.5,

  // === Cartilage Properties ===
  cartilage_porosity: 0.65,
  pore_size: 0.3,
  enable_porous_structure: true,

  // === Cell Seeding Ratios ===
  cell_ratio_adsc_chondrocyte: 0.25,
  enable_cell_guidance_channels: false,

  // === Three-Layer Architecture ===
  three_layer_structure: true,
  perichondrium_thickness: 0.1,
  core_cartilage_ratio: 0.8,

  // === Surface Features ===
  enable_mucosal_texture: false,
  mucosal_groove_depth: 0.05,
  enable_vascular_channels: false,
  vascular_channel_diameter: 0.2,
  vascular_channel_spacing: 2.0,

  // === Edges and Margins ===
  edge_rounding: 0.5,
  enable_suture_holes: false,
  suture_hole_diameter: 1.0,
  suture_hole_spacing: 5.0,

  // === Generation Settings ===
  seed: 42,
  randomness: 0.1,
  detail_level: 'medium',
};
