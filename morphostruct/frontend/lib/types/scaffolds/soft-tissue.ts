/**
 * Soft Tissue Parameters
 */

import { BaseParams, MuscleArchitecture } from './base';

// ============================================================================
// Soft Tissue Parameters
// ============================================================================

export interface MultilayerSkinParams extends BaseParams {
  // Basic Geometry
  diameter_mm: number;

  // Epidermis Layer
  epidermis_thickness_um: number;
  keratinocyte_layers?: number;

  // Dermis Layers
  dermis_thickness_mm?: number;
  papillary_dermis_thickness_um: number;
  reticular_dermis_thickness_mm: number;

  // Hypodermis
  hypodermis_thickness_mm: number;

  // Rete Ridges
  enable_rete_ridges: boolean;
  rete_ridge_height_um: number;
  rete_ridge_width_um: number;
  rete_ridge_spacing_um: number;
  rete_ridge_depth_variance?: number;

  // Dermal Papillae
  enable_dermal_papillae?: boolean;
  papillae_density_per_mm2?: number;
  papillae_height_um?: number;
  papillae_diameter_um?: number;

  // Hair Follicles
  enable_hair_follicles: boolean;
  hair_follicle_density_per_cm2: number;
  hair_follicle_diameter_um: number;
  hair_follicle_depth_mm: number;

  // Sweat Glands
  enable_sweat_glands: boolean;
  sweat_gland_density_per_cm2: number;
  sweat_gland_diameter_um: number;
  sweat_gland_depth_mm?: number;

  // Sebaceous Glands
  enable_sebaceous_glands?: boolean;
  sebaceous_gland_density_per_cm2?: number;
  sebaceous_gland_diameter_um?: number;

  // Vascular Network
  enable_vascular_channels: boolean;
  vascular_channel_diameter_mm: number;
  vascular_channel_count?: number;
  vascular_channel_spacing_mm: number;

  // Collagen Architecture
  enable_collagen_orientation?: boolean;
  papillary_collagen_diameter_um?: number;
  reticular_collagen_diameter_um?: number;
  collagen_alignment?: number;

  // Porosity
  dermal_porosity: number;
  pore_diameter_um: number;
  pore_gradient?: [number, number, number];
  pore_interconnectivity?: number;

  // Surface Features
  enable_surface_texture?: boolean;
  surface_roughness?: number;

  // Randomization
  seed: number;
  randomness?: number;
  position_noise?: number;
}

export interface SkeletalMuscleParams extends BaseParams {
  // Basic Geometry
  length_mm: number;
  width_mm: number;
  height_mm: number;

  // Muscle Fiber Parameters
  fiber_diameter_um: number;
  fiber_spacing_um: number;
  fiber_length_mm: number;

  // Fascicle Parameters
  fascicle_diameter_mm: number;
  fascicle_count: number;
  fascicle_spacing_mm: number;

  // Sarcomere Parameters
  sarcomere_length_um: number;
  sarcomere_resolution: number;

  // Fiber Architecture
  architecture_type: MuscleArchitecture;
  pennation_angle_deg: number;
  fiber_alignment: number;
  fiber_curvature: number;

  // Connective Tissue - Endomysium
  endomysium_thickness_um: number;
  endomysium_porosity: number;

  // Connective Tissue - Perimysium
  perimysium_thickness_um: number;
  perimysium_porosity: number;
  enable_perimysium: boolean;

  // Connective Tissue - Epimysium
  epimysium_thickness_um: number;
  enable_epimysium: boolean;

  // Mechanical Properties
  contraction_force_uN: number;

  // Vascular Network
  enable_vascular_channels: boolean;
  capillary_diameter_um: number;
  capillary_density_per_mm2: number;
  arteriole_diameter_um: number;
  venule_diameter_um: number;

  // Neuromuscular Features
  enable_motor_endplates: boolean;
  motor_endplate_density_per_mm2: number;

  // Randomization
  seed: number;
  position_noise: number;
  diameter_variance: number;
}

export interface CorneaParams extends BaseParams {
  // Basic Geometry
  diameter_mm: number;
  radius_of_curvature_mm: number;

  // Total Thickness
  total_thickness_um: number;

  // Layer Thicknesses (anterior to posterior)
  epithelium_thickness_um: number;
  bowmans_layer_thickness_um: number;
  stroma_thickness_um: number;
  descemet_thickness_um: number;
  endothelium_thickness_um: number;

  // Stromal Architecture
  num_lamellae: number;
  lamella_thickness_um: number;
  lamella_angle_variation_deg: number;

  // Collagen Fibril Parameters
  collagen_fibril_diameter_nm: number;
  collagen_fibril_spacing_nm: number;

  // Keratocyte Distribution
  keratocyte_density_per_mm3: number;
  enable_keratocyte_markers: boolean;

  // Nerve Plexus
  enable_nerve_plexus: boolean;
  subbasal_nerve_density_per_mm2: number;
  stromal_nerve_bundle_count: number;

  // Optical Properties
  transparency_factor: number;
  refractive_index: number;

  // Limbal Transition
  enable_limbal_zone: boolean;
  limbal_width_mm: number;

  // Surface Curvature
  asphericity_q: number;
  posterior_radius_mm: number;

  // Randomization
  seed: number;
  thickness_variance: number;
}

export interface AdiposeParams extends BaseParams {
  // Basic Geometry
  diameter_mm: number;
  height_mm: number;

  // Adipocyte Parameters
  adipocyte_diameter_um: number;
  adipocyte_size_variance: number;
  cell_density_per_mL: number;

  // Lobule Structure
  lobule_size_mm: number;
  enable_lobules: boolean;
  lobules_per_cm2: number;

  // Septum/Connective Tissue
  septum_thickness_um: number;
  septum_porosity: number;

  // Vascular Network
  enable_vascular_channels: boolean;
  capillary_diameter_um: number;
  capillary_density_per_mm2: number;
  vascular_channel_diameter_mm: number;
  vascular_channel_spacing_mm: number;

  // Arteriole/Venule
  arteriole_diameter_um: number;
  venule_diameter_um: number;

  // Stromal Vascular Fraction
  enable_svf_channels: boolean;
  svf_channel_diameter_um: number;

  // Pore Architecture
  porosity: number;
  pore_size_um: number;
  pore_interconnectivity: number;

  // ECM Features
  enable_ecm_fibers: boolean;
  collagen_fiber_density: number;
  elastin_fiber_density: number;

  // Basement Membrane
  enable_basement_membrane: boolean;
  basement_membrane_thickness_um: number;

  // Surface Features
  enable_surface_texture: boolean;
  surface_roughness: number;

  // Mechanical Properties
  target_stiffness_kPa: number;

  // Randomization
  seed: number;
  position_noise: number;
}

// ============================================================================
// Default Values
// ============================================================================

const DEFAULT_RESOLUTION = 16;

export const DEFAULT_MULTILAYER_SKIN: MultilayerSkinParams = {
  resolution: DEFAULT_RESOLUTION,
  // Basic Geometry
  diameter_mm: 10.0,
  // Epidermis Layer
  epidermis_thickness_um: 100.0,
  keratinocyte_layers: 5,
  // Dermis Layers
  dermis_thickness_mm: 1.5,
  papillary_dermis_thickness_um: 200.0,
  reticular_dermis_thickness_mm: 1.3,
  // Hypodermis
  hypodermis_thickness_mm: 10.0,
  // Rete Ridges
  enable_rete_ridges: true,
  rete_ridge_height_um: 51.0,
  rete_ridge_width_um: 150.0,
  rete_ridge_spacing_um: 105.0,
  rete_ridge_depth_variance: 0.2,
  // Dermal Papillae
  enable_dermal_papillae: true,
  papillae_density_per_mm2: 100.0,
  papillae_height_um: 100.0,
  papillae_diameter_um: 75.0,
  // Hair Follicles
  enable_hair_follicles: false,
  hair_follicle_density_per_cm2: 130.0,
  hair_follicle_diameter_um: 70.0,
  hair_follicle_depth_mm: 2.5,
  // Sweat Glands
  enable_sweat_glands: false,
  sweat_gland_density_per_cm2: 155.0,
  sweat_gland_diameter_um: 40.0,
  sweat_gland_depth_mm: 1.5,
  // Sebaceous Glands
  enable_sebaceous_glands: false,
  sebaceous_gland_density_per_cm2: 100.0,
  sebaceous_gland_diameter_um: 100.0,
  // Vascular Network
  enable_vascular_channels: true,
  vascular_channel_diameter_mm: 0.15,
  vascular_channel_count: 4,
  vascular_channel_spacing_mm: 2.0,
  // Collagen Architecture
  enable_collagen_orientation: false,
  papillary_collagen_diameter_um: 2.0,
  reticular_collagen_diameter_um: 10.0,
  collagen_alignment: 0.3,
  // Porosity
  dermal_porosity: 0.19,
  pore_diameter_um: 219.0,
  pore_gradient: [0.15, 0.25, 0.40],
  pore_interconnectivity: 0.8,
  // Surface Features
  enable_surface_texture: false,
  surface_roughness: 0.1,
  // Randomization
  seed: 42,
  randomness: 0.1,
  position_noise: 0.15,
};

export const DEFAULT_SKELETAL_MUSCLE: SkeletalMuscleParams = {
  resolution: 8,
  // Basic Geometry
  length_mm: 20.0,
  width_mm: 10.0,
  height_mm: 5.0,
  // Muscle Fiber Parameters
  fiber_diameter_um: 50.0,
  fiber_spacing_um: 100.0,
  fiber_length_mm: 15.0,
  // Fascicle Parameters
  fascicle_diameter_mm: 5.0,
  fascicle_count: 4,
  fascicle_spacing_mm: 1.0,
  // Sarcomere Parameters
  sarcomere_length_um: 2.5,
  sarcomere_resolution: 4,
  // Fiber Architecture
  architecture_type: MuscleArchitecture.PARALLEL,
  pennation_angle_deg: 15.0,
  fiber_alignment: 0.6,
  fiber_curvature: 0.0,
  // Connective Tissue - Endomysium
  endomysium_thickness_um: 0.5,
  endomysium_porosity: 0.3,
  // Connective Tissue - Perimysium
  perimysium_thickness_um: 20.0,
  perimysium_porosity: 0.25,
  enable_perimysium: true,
  // Connective Tissue - Epimysium
  epimysium_thickness_um: 100.0,
  enable_epimysium: true,
  // Mechanical Properties
  contraction_force_uN: 443.0,
  // Vascular Network
  enable_vascular_channels: true,
  capillary_diameter_um: 8.0,
  capillary_density_per_mm2: 300,
  arteriole_diameter_um: 50.0,
  venule_diameter_um: 60.0,
  // Neuromuscular Features
  enable_motor_endplates: false,
  motor_endplate_density_per_mm2: 0.5,
  // Randomization
  seed: 42,
  position_noise: 0.1,
  diameter_variance: 0.1,
};

export const DEFAULT_CORNEA: CorneaParams = {
  resolution: 24,
  // Basic Geometry
  diameter_mm: 11.5,
  radius_of_curvature_mm: 7.8,
  // Total Thickness
  total_thickness_um: 520.0,
  // Layer Thicknesses
  epithelium_thickness_um: 50.0,
  bowmans_layer_thickness_um: 12.0,
  stroma_thickness_um: 500.0,
  descemet_thickness_um: 8.0,
  endothelium_thickness_um: 5.0,
  // Stromal Architecture
  num_lamellae: 300,
  lamella_thickness_um: 2.0,
  lamella_angle_variation_deg: 90.0,
  // Collagen Fibril Parameters
  collagen_fibril_diameter_nm: 30.0,
  collagen_fibril_spacing_nm: 42.0,
  // Keratocyte Distribution
  keratocyte_density_per_mm3: 20522.0,
  enable_keratocyte_markers: false,
  // Nerve Plexus
  enable_nerve_plexus: false,
  subbasal_nerve_density_per_mm2: 5900.0,
  stromal_nerve_bundle_count: 70,
  // Optical Properties
  transparency_factor: 1.0,
  refractive_index: 1.376,
  // Limbal Transition
  enable_limbal_zone: false,
  limbal_width_mm: 1.5,
  // Surface Curvature
  asphericity_q: -0.26,
  posterior_radius_mm: 6.5,
  // Randomization
  seed: 42,
  thickness_variance: 0.05,
};

export const DEFAULT_ADIPOSE: AdiposeParams = {
  resolution: 8,
  // Basic Geometry
  diameter_mm: 15.0,
  height_mm: 10.0,
  // Adipocyte Parameters
  adipocyte_diameter_um: 90.0,
  adipocyte_size_variance: 0.15,
  cell_density_per_mL: 40e6,
  // Lobule Structure
  lobule_size_mm: 2.0,
  enable_lobules: true,
  lobules_per_cm2: 4.0,
  // Septum/Connective Tissue
  septum_thickness_um: 50.0,
  septum_porosity: 0.3,
  // Vascular Network
  enable_vascular_channels: true,
  capillary_diameter_um: 7.0,
  capillary_density_per_mm2: 400,
  vascular_channel_diameter_mm: 0.4,
  vascular_channel_spacing_mm: 2.0,
  // Arteriole/Venule
  arteriole_diameter_um: 30.0,
  venule_diameter_um: 40.0,
  // Stromal Vascular Fraction
  enable_svf_channels: false,
  svf_channel_diameter_um: 50.0,
  // Pore Architecture
  porosity: 0.8,
  pore_size_um: 200.0,
  pore_interconnectivity: 0.9,
  // ECM Features
  enable_ecm_fibers: false,
  collagen_fiber_density: 0.3,
  elastin_fiber_density: 0.1,
  // Basement Membrane
  enable_basement_membrane: false,
  basement_membrane_thickness_um: 0.1,
  // Surface Features
  enable_surface_texture: false,
  surface_roughness: 0.2,
  // Mechanical Properties
  target_stiffness_kPa: 2.0,
  // Randomization
  seed: 42,
  position_noise: 0.2,
};
