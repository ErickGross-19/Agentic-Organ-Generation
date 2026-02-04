/**
 * Tubular Organ Parameters
 */

import { BaseParams, InnerTexture } from './base';

// ============================================================================
// Tubular Organ Parameters
// ============================================================================

export interface BloodVesselParams extends BaseParams {
  // === Basic Geometry ===
  inner_diameter_mm: number;
  wall_thickness_mm: number;
  length_mm: number;

  // === Layer Thicknesses ===
  intima_thickness_um?: number;
  media_thickness_um?: number;
  adventitia_thickness_um?: number;
  layer_ratios?: [number, number, number];

  // === Elastic Laminae (Media Layer) ===
  elastic_lamina_thickness_um?: number;
  external_elastic_lamina_thickness_um?: number;
  num_elastic_laminae?: number;
  enable_elastic_laminae?: boolean;
  fenestration_count?: number;
  fenestration_diameter_um?: number;

  // === Smooth Muscle Cells (Media) ===
  smc_length_um?: number;
  smc_orientation_angle_deg?: number;
  enable_smc_alignment?: boolean;
  smc_density_per_mm2?: number;

  // === Endothelial Features (Intima) ===
  enable_endothelial_texture?: boolean;
  endothelial_cell_size_um?: number;
  endothelial_cell_density_per_mm2?: number;
  endothelial_bump_height_um?: number;

  // === Adventitial Features ===
  enable_vasa_vasorum?: boolean;
  vasa_vasorum_diameter_um?: number;
  vasa_vasorum_spacing_um?: number;
  vasa_vasorum_density_per_mm2?: number;

  // === Bifurcation ===
  enable_bifurcation?: boolean;
  bifurcation_angle_deg?: number;
  daughter_diameter_ratio?: number;

  // === Porosity and Permeability ===
  scaffold_porosity?: number;
  pore_size_um?: number;
  enable_radial_pores?: boolean;
  enable_pore_network?: boolean;

  // === Vessel Geometry Variations ===
  taper_ratio?: number;
  vessel_taper?: number;
  tortuosity_index?: number;
  vessel_tortuosity?: number;
  tortuosity_wavelength_mm?: number;
  tortuosity_amplitude_ratio?: number;

  // === Mechanical Properties ===
  target_compliance_percent_per_100mmHg?: number;
  burst_pressure_mmHg?: number;

  // === Randomization ===
  position_noise?: number;
  random_seed?: number;
}

export interface NerveConduitParams extends BaseParams {
  // === Basic Geometry ===
  outer_diameter_mm: number;
  inner_diameter_mm?: number;
  wall_thickness_mm: number;
  length_mm: number;

  // === Channel Structure ===
  num_channels?: number;
  channel_count?: number; // Legacy alias for num_channels
  channel_diameter_um: number;
  channel_spacing_um?: number;

  // === Fascicle Architecture ===
  fascicle_diameter_um?: number;
  num_fascicles?: number;
  enable_fascicle_chambers?: boolean;
  fascicle_spacing_um?: number;

  // === Perineurium Layer ===
  perineurium_thickness_um?: number;
  enable_perineurium?: boolean;

  // === Epineurium Layer ===
  epineurium_thickness_um?: number;
  enable_epineurium?: boolean;

  // === Schwann Cell Guidance Features ===
  ridge_width_um?: number;
  groove_width_um?: number;
  groove_depth_um?: number;
  groove_spacing_um?: number;
  enable_microgrooves?: boolean;
  num_microgrooves?: number;

  // === Inner Surface Features ===
  enable_guidance_channels?: boolean;
  guidance_channel_diameter_um?: number;
  guidance_channel_pattern?: 'linear' | 'spiral';

  // === Wall Porosity ===
  wall_porosity?: number;
  pore_size_um?: number;
  inner_surface_porosity?: number;

  // === Growth Factor Delivery ===
  enable_growth_factor_reservoirs?: boolean;
  reservoir_diameter_um?: number;
  reservoir_count?: number;
  reservoir_spacing_mm?: number;
  reservoir_rings?: number;

  // === Suture Features ===
  enable_biodegradable_suture_holes?: boolean;
  suture_hole_diameter_mm?: number;
  suture_hole_count?: number;
  suture_hole_distance_from_end_mm?: number;

  // === Geometry Variations ===
  taper_ratio?: number;
  enable_flared_ends?: boolean;
  flare_angle_deg?: number;
  flare_length_mm?: number;

  // === Randomization ===
  position_noise_um?: number;
  channel_variance?: number;
  random_seed?: number;
}

export interface SpinalCordParams extends BaseParams {
  // === Basic Geometry ===
  total_diameter_mm?: number;
  cord_diameter_mm?: number; // Legacy alias for total_diameter_mm
  length_mm: number;

  // === Gray Matter Configuration ===
  gray_matter_volume_ratio?: number;
  gray_matter_ratio?: number; // Legacy alias for gray_matter_volume_ratio
  gray_matter_pattern?: 'butterfly' | 'h_shape' | 'simplified';

  // === Gray Matter Horns ===
  dorsal_horn_width_mm?: number;
  dorsal_horn_height_mm?: number;
  ventral_horn_width_mm?: number;
  ventral_horn_height_mm?: number;
  lateral_horn_present?: boolean;
  lateral_horn_size_mm?: number;

  // === Central Canal ===
  central_canal_diameter_mm?: number;
  enable_central_canal?: boolean;

  // === White Matter ===
  white_matter_thickness_mm?: number;
  enable_tract_columns?: boolean;
  dorsal_column_width_mm?: number;
  lateral_column_width_mm?: number;
  ventral_column_width_mm?: number;

  // === Anterior Fissure ===
  enable_anterior_fissure?: boolean;
  anterior_fissure_depth_mm?: number;
  anterior_fissure_width_mm?: number;

  // === Guidance Channels ===
  num_guidance_channels?: number;
  channel_count?: number; // Legacy alias for num_guidance_channels
  channel_diameter_um: number;
  channel_pattern?: 'radial' | 'grid' | 'random';
  channel_region?: 'white_matter' | 'gray_matter' | 'both';

  // === Meningeal Layers ===
  enable_pia_mater?: boolean;
  pia_mater_thickness_um?: number;
  enable_arachnoid_mater?: boolean;
  arachnoid_thickness_um?: number;
  enable_dura_mater?: boolean;
  dura_mater_thickness_mm?: number;
  subarachnoid_space_mm?: number;

  // === Root Entry/Exit Zones ===
  enable_root_entry_zones?: boolean;
  dorsal_root_diameter_mm?: number;
  ventral_root_diameter_mm?: number;
  root_spacing_mm?: number;
  root_angle_deg?: number;

  // === Vascular Features ===
  enable_anterior_spinal_artery?: boolean;
  anterior_spinal_artery_diameter_mm?: number;
  enable_posterior_spinal_arteries?: boolean;
  posterior_spinal_artery_diameter_mm?: number;

  // === Porosity ===
  scaffold_porosity?: number;
  pore_size_um?: number;

  // === Randomization ===
  position_noise?: number;
  random_seed?: number;
}

export interface BladderParams extends BaseParams {
  // === Basic Geometry ===
  dome_diameter_mm?: number;
  diameter_mm?: number; // Legacy alias for dome_diameter_mm
  wall_thickness_empty_mm?: number;
  wall_thickness_mm?: number; // Legacy alias for wall_thickness_empty_mm
  dome_curvature: number;
  dome_height_mm?: number;

  // === Layer Configuration (detailed) ===
  urothelium_thickness_um?: number;
  enable_urothelium?: boolean;
  lamina_propria_thickness_um?: number;
  enable_lamina_propria?: boolean;
  detrusor_thickness_mm?: number;
  enable_detrusor_muscle?: boolean;
  detrusor_layer_count?: number;
  serosa_thickness_um?: number;
  enable_serosa?: boolean;
  layer_count?: number;

  // === Rugae (Mucosal Folds) ===
  rugae_height_um?: number;
  rugae_spacing_mm?: number;
  enable_urothelial_folds?: boolean;
  fold_count?: number;
  fold_variance?: number;

  // === Compliance and Mechanics ===
  compliance_ml_per_cmH2O?: number;
  max_capacity_ml?: number;

  // === Trigone Region ===
  trigone_marker?: boolean;
  trigone_width_mm?: number;
  trigone_height_mm?: number;

  // === Ureteral/Urethral Openings ===
  enable_ureteral_openings?: boolean;
  ureteral_opening_diameter_mm?: number;
  ureteral_spacing_mm?: number;
  urethral_opening_diameter_mm?: number;
  enable_urethral_opening?: boolean;

  // === Vascular Network ===
  enable_suburothelial_capillaries?: boolean;
  capillary_diameter_um?: number;
  capillary_spacing_mm?: number;

  // === Muscle Bundle Features ===
  muscle_bundle_diameter_mm?: number;
  muscle_bundle_spacing_mm?: number;
  enable_muscle_bundles?: boolean;

  // === Nerve Innervation Markers ===
  enable_nerve_markers?: boolean;
  nerve_density_per_cm2?: number;

  // === Porosity ===
  scaffold_porosity?: number;
  pore_size_um?: number;
  enable_pore_network?: boolean;

  // === Randomization ===
  position_noise_mm?: number;
  random_seed?: number;
}

export interface TracheaParams extends BaseParams {
  // === Basic Geometry ===
  total_length_mm?: number;
  length_mm?: number; // Legacy alias for total_length_mm
  outer_diameter_mm: number;
  inner_diameter_mm?: number;

  // === Cartilage Ring Configuration ===
  num_cartilage_rings?: number;
  ring_count?: number; // Legacy alias for num_cartilage_rings
  ring_height_mm?: number;
  ring_width_mm?: number;
  ring_thickness_mm?: number;
  ring_gap_deg?: number;
  posterior_opening_degrees?: number;
  posterior_gap_angle_deg?: number; // Legacy alias for posterior_opening_degrees
  interring_spacing_mm?: number;
  ring_spacing_mm?: number; // Legacy alias for interring_spacing_mm

  // === Cartilage Properties ===
  cartilage_type?: 'hyaline' | 'elastic' | 'fibrocartilage';
  enable_c_shaped_cartilage?: boolean;
  cartilage_porosity?: number;

  // === Mucosal Layer ===
  mucosa_thickness_um?: number;
  enable_mucosal_layer?: boolean;
  enable_ciliated_epithelium_markers?: boolean;
  cilia_density?: number;
  enable_cilia_markers?: boolean;
  enable_goblet_cell_markers?: boolean;
  epithelium_thickness_um?: number;
  lamina_propria_thickness_um?: number;

  // === Submucosa ===
  submucosa_thickness_um?: number;
  enable_submucosa?: boolean;

  // === Submucosal Glands ===
  enable_submucosal_glands?: boolean;
  gland_diameter_um?: number;
  gland_diameter_mm?: number;
  gland_density_per_mm2?: number;

  // === Trachealis Muscle ===
  enable_trachealis_muscle?: boolean;
  trachealis_thickness_um?: number;
  trachealis_thickness_mm?: number;
  trachealis_width_deg?: number;
  trachealis_fiber_orientation_deg?: number;

  // === Perichondrium ===
  perichondrium_thickness_um?: number;
  enable_perichondrium?: boolean;

  // === Mucosal Folds ===
  mucosal_fold_depth_mm?: number;
  mucosal_fold_count?: number;
  enable_mucosal_folds?: boolean;

  // === Vascular Channels ===
  vascular_channel_diameter_um?: number;
  vascular_channel_diameter_mm?: number;
  vascular_spacing_um?: number;
  vascular_channel_spacing_mm?: number;
  enable_vascular_channels?: boolean;

  // === Annular Ligaments ===
  enable_annular_ligaments?: boolean;
  ligament_thickness_mm?: number;
  annular_ligament_thickness_mm?: number;

  // === Carina (for bifurcation) ===
  enable_carina?: boolean;
  carina_angle_deg?: number;
  left_bronchus_angle_deg?: number;
  right_bronchus_angle_deg?: number;
  bronchus_diameter_mm?: number;
  bronchus_length_mm?: number;

  // === Porosity ===
  scaffold_porosity?: number;
  pore_size_um?: number;

  // === Randomization ===
  position_noise_mm?: number;
  ring_variance_pct?: number;
  random_seed?: number;
}

export interface TubularConduitParams extends BaseParams {
  // === Basic Geometry ===
  outer_diameter_mm: number;
  wall_thickness_mm: number;
  length_mm: number;

  // === Inner Surface ===
  inner_texture?: InnerTexture;
  groove_count?: number;
  groove_depth_mm?: number;
}

export interface VascularPerfusionDishParams extends BaseParams {
  // === Network Structure ===
  /** Number of inlet branches (1-25) */
  inlets: number;
  /** Branching levels (0-8) */
  levels: number;
  /** Splits per branch (1-6) */
  splits: number;
  /** Branch spread factor (0.1-0.8) */
  spread: number;
  /** Branch radius ratio (0.5-0.95, Murray's law = 0.79) */
  ratio: number;
  /** Cone angle in degrees (10-180) */
  cone_angle: number;
  /** Branch curvature (0-1) */
  curvature: number;
  /** Height from bottom where vessels terminate (0.02-1.0) */
  bottom_height: number;

  // === Variation/Randomness ===
  /** Radius variation (0-1, as fraction) */
  radius_variation: number;
  /** Chance of flipping branch direction (0-0.5) */
  flip_chance: number;
  /** Z-axis variation factor (0-0.5) */
  z_variation: number;
  /** Angle variation factor (0-0.5) */
  angle_variation: number;
  /** Extra buffer for collision detection (0-0.3) */
  collision_buffer: number;

  // === Behavior Flags ===
  /** Use even 360° spread vs directional cone spread */
  even_spread: boolean;
  /** Use deterministic straight grid-aligned channels */
  deterministic: boolean;
  /** Force terminal branch tips to point straight down */
  tips_down: boolean;

  // === Geometry Dimensions ===
  /** Outer cylinder radius (mm) */
  outer_radius_mm: number;
  /** Total height (mm) */
  height_mm: number;
  /** Inlet branch radius (mm) */
  inlet_radius_mm: number;
  /** Height of rim around dish edge (mm, 0-3) */
  rim_height_mm: number;
  /** Flip scaffold 180° for export */
  flip_upside_down: boolean;
  /** Random seed for reproducibility */
  seed?: number;
}

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULT_BLOOD_VESSEL: BloodVesselParams = {
  // === Basic Geometry ===
  resolution: 16,
  inner_diameter_mm: 3.0,
  wall_thickness_mm: 0.5,
  length_mm: 50.0,

  // === Layer Thicknesses (anatomically accurate) ===
  intima_thickness_um: 50.0,
  media_thickness_um: 400.0,
  adventitia_thickness_um: 200.0,
  layer_ratios: [1, 8, 4],

  // === Elastic Laminae (Media Layer) ===
  elastic_lamina_thickness_um: 1.8,
  external_elastic_lamina_thickness_um: 2.0,
  num_elastic_laminae: 10,
  enable_elastic_laminae: false,
  fenestration_count: 0,
  fenestration_diameter_um: 2.0,

  // === Smooth Muscle Cells (Media) ===
  smc_length_um: 175.0,
  smc_orientation_angle_deg: 0.0,
  enable_smc_alignment: false,
  smc_density_per_mm2: 1000.0,

  // === Endothelial Features (Intima) ===
  enable_endothelial_texture: false,
  endothelial_cell_size_um: 30.0,
  endothelial_cell_density_per_mm2: 5000.0,
  endothelial_bump_height_um: 2.0,

  // === Adventitial Features ===
  enable_vasa_vasorum: false,
  vasa_vasorum_diameter_um: 50.0,
  vasa_vasorum_spacing_um: 500.0,
  vasa_vasorum_density_per_mm2: 10.0,

  // === Bifurcation ===
  enable_bifurcation: false,
  bifurcation_angle_deg: 45.0,
  daughter_diameter_ratio: 0.794,

  // === Porosity and Permeability ===
  scaffold_porosity: 0.5,
  pore_size_um: 100.0,
  enable_radial_pores: false,
  enable_pore_network: false,

  // === Vessel Geometry Variations ===
  taper_ratio: 0.0,
  vessel_taper: 0.0,
  tortuosity_index: 0.0,
  vessel_tortuosity: 0.0,
  tortuosity_wavelength_mm: 5.0,
  tortuosity_amplitude_ratio: 0.15,

  // === Mechanical Properties ===
  target_compliance_percent_per_100mmHg: 6.0,
  burst_pressure_mmHg: 2000.0,

  // === Randomization ===
  position_noise: 0.0,
  random_seed: 42,
};

export const DEFAULT_NERVE_CONDUIT: NerveConduitParams = {
  // === Basic Geometry ===
  resolution: 12,
  outer_diameter_mm: 2.5,
  inner_diameter_mm: 2.0,
  wall_thickness_mm: 0.4,
  length_mm: 30.0,

  // === Channel Structure ===
  num_channels: 50,
  channel_diameter_um: 150.0,
  channel_spacing_um: 100.0,

  // === Fascicle Architecture ===
  fascicle_diameter_um: 400.0,
  num_fascicles: 4,
  enable_fascicle_chambers: false,
  fascicle_spacing_um: 150.0,

  // === Perineurium Layer ===
  perineurium_thickness_um: 20.0,
  enable_perineurium: false,

  // === Epineurium Layer ===
  epineurium_thickness_um: 125.0,
  enable_epineurium: false,

  // === Schwann Cell Guidance Features ===
  ridge_width_um: 5.0,
  groove_width_um: 20.0,
  groove_depth_um: 3.0,
  groove_spacing_um: 25.0,
  enable_microgrooves: false,
  num_microgrooves: 0,

  // === Inner Surface Features ===
  enable_guidance_channels: true,
  guidance_channel_diameter_um: 50.0,
  guidance_channel_pattern: 'linear',

  // === Wall Porosity ===
  wall_porosity: 0.75,
  pore_size_um: 50.0,
  inner_surface_porosity: 0.3,

  // === Growth Factor Delivery ===
  enable_growth_factor_reservoirs: false,
  reservoir_diameter_um: 400.0,
  reservoir_count: 4,
  reservoir_spacing_mm: 2.5,
  reservoir_rings: 0,

  // === Suture Features ===
  enable_biodegradable_suture_holes: false,
  suture_hole_diameter_mm: 0.5,
  suture_hole_count: 4,
  suture_hole_distance_from_end_mm: 2.5,

  // === Geometry Variations ===
  taper_ratio: 1.0,
  enable_flared_ends: false,
  flare_angle_deg: 15.0,
  flare_length_mm: 1.5,

  // === Randomization ===
  position_noise_um: 0.0,
  channel_variance: 0.0,
  random_seed: 42,
};

export const DEFAULT_SPINAL_CORD: SpinalCordParams = {
  // === Basic Geometry ===
  resolution: 20,
  total_diameter_mm: 10.0,
  length_mm: 100.0,

  // === Gray Matter Configuration ===
  gray_matter_volume_ratio: 0.22,
  gray_matter_pattern: 'butterfly',

  // === Gray Matter Horns ===
  dorsal_horn_width_mm: 2.0,
  dorsal_horn_height_mm: 2.5,
  ventral_horn_width_mm: 2.5,
  ventral_horn_height_mm: 2.0,
  lateral_horn_present: false,
  lateral_horn_size_mm: 0.8,

  // === Central Canal ===
  central_canal_diameter_mm: 0.3,
  enable_central_canal: true,

  // === White Matter ===
  white_matter_thickness_mm: 2.5,
  enable_tract_columns: false,
  dorsal_column_width_mm: 3.0,
  lateral_column_width_mm: 4.0,
  ventral_column_width_mm: 3.5,

  // === Anterior Fissure ===
  enable_anterior_fissure: false,
  anterior_fissure_depth_mm: 3.0,
  anterior_fissure_width_mm: 0.5,

  // === Guidance Channels ===
  num_guidance_channels: 50,
  channel_diameter_um: 150.0,
  channel_pattern: 'radial',
  channel_region: 'white_matter',

  // === Meningeal Layers ===
  enable_pia_mater: false,
  pia_mater_thickness_um: 50.0,
  enable_arachnoid_mater: false,
  arachnoid_thickness_um: 100.0,
  enable_dura_mater: false,
  dura_mater_thickness_mm: 0.5,
  subarachnoid_space_mm: 2.0,

  // === Root Entry/Exit Zones ===
  enable_root_entry_zones: false,
  dorsal_root_diameter_mm: 1.0,
  ventral_root_diameter_mm: 0.8,
  root_spacing_mm: 10.0,
  root_angle_deg: 45.0,

  // === Vascular Features ===
  enable_anterior_spinal_artery: false,
  anterior_spinal_artery_diameter_mm: 0.8,
  enable_posterior_spinal_arteries: false,
  posterior_spinal_artery_diameter_mm: 0.4,

  // === Porosity ===
  scaffold_porosity: 0.6,
  pore_size_um: 100.0,

  // === Randomization ===
  position_noise: 0.0,
  random_seed: 42,
};

export const DEFAULT_BLADDER: BladderParams = {
  // === Basic Geometry ===
  resolution: 20,
  dome_diameter_mm: 70.0,
  wall_thickness_empty_mm: 3.3,
  dome_curvature: 0.6,
  dome_height_mm: 50.0,

  // === Layer Configuration (detailed) ===
  urothelium_thickness_um: 200.0,
  enable_urothelium: true,
  lamina_propria_thickness_um: 700.0,
  enable_lamina_propria: true,
  detrusor_thickness_mm: 2.2,
  enable_detrusor_muscle: true,
  detrusor_layer_count: 3,
  serosa_thickness_um: 250.0,
  enable_serosa: true,
  layer_count: 3,

  // === Rugae (Mucosal Folds) ===
  rugae_height_um: 750.0,
  rugae_spacing_mm: 2.5,
  enable_urothelial_folds: false,
  fold_count: 12,
  fold_variance: 0.1,

  // === Compliance and Mechanics ===
  compliance_ml_per_cmH2O: 25.0,
  max_capacity_ml: 500.0,

  // === Trigone Region ===
  trigone_marker: false,
  trigone_width_mm: 25.0,
  trigone_height_mm: 30.0,

  // === Ureteral/Urethral Openings ===
  enable_ureteral_openings: false,
  ureteral_opening_diameter_mm: 3.0,
  ureteral_spacing_mm: 25.0,
  urethral_opening_diameter_mm: 6.0,
  enable_urethral_opening: false,

  // === Vascular Network ===
  enable_suburothelial_capillaries: false,
  capillary_diameter_um: 8.0,
  capillary_spacing_mm: 0.5,

  // === Muscle Bundle Features ===
  muscle_bundle_diameter_mm: 0.3,
  muscle_bundle_spacing_mm: 0.5,
  enable_muscle_bundles: false,

  // === Nerve Innervation Markers ===
  enable_nerve_markers: false,
  nerve_density_per_cm2: 50.0,

  // === Porosity ===
  scaffold_porosity: 0.6,
  pore_size_um: 100.0,
  enable_pore_network: false,

  // === Randomization ===
  position_noise_mm: 0.0,
  random_seed: 42,
};

export const DEFAULT_TRACHEA: TracheaParams = {
  // === Basic Geometry ===
  resolution: 20,
  total_length_mm: 110.0,
  outer_diameter_mm: 20.0,
  inner_diameter_mm: 16.0,

  // === Cartilage Ring Configuration ===
  num_cartilage_rings: 18,
  ring_height_mm: 4.0,
  ring_width_mm: 3.5,
  ring_thickness_mm: 1.5,
  ring_gap_deg: 80.0,
  posterior_opening_degrees: 80.0,
  interring_spacing_mm: 2.0,

  // === Cartilage Properties ===
  cartilage_type: 'hyaline',
  enable_c_shaped_cartilage: true,
  cartilage_porosity: 0.3,

  // === Mucosal Layer ===
  mucosa_thickness_um: 350.0,
  enable_mucosal_layer: false,
  enable_ciliated_epithelium_markers: false,
  cilia_density: 200.0,
  enable_cilia_markers: false,
  enable_goblet_cell_markers: false,
  epithelium_thickness_um: 60.0,
  lamina_propria_thickness_um: 290.0,

  // === Submucosa ===
  submucosa_thickness_um: 500.0,
  enable_submucosa: false,

  // === Submucosal Glands ===
  enable_submucosal_glands: false,
  gland_diameter_um: 300.0,
  gland_diameter_mm: 0.3,
  gland_density_per_mm2: 1.0,

  // === Trachealis Muscle ===
  enable_trachealis_muscle: false,
  trachealis_thickness_um: 800.0,
  trachealis_thickness_mm: 0.8,
  trachealis_width_deg: 80.0,
  trachealis_fiber_orientation_deg: 0.0,

  // === Perichondrium ===
  perichondrium_thickness_um: 80.0,
  enable_perichondrium: false,

  // === Mucosal Folds ===
  mucosal_fold_depth_mm: 0.2,
  mucosal_fold_count: 12,
  enable_mucosal_folds: false,

  // === Vascular Channels ===
  vascular_channel_diameter_um: 100.0,
  vascular_channel_diameter_mm: 0.1,
  vascular_spacing_um: 200.0,
  vascular_channel_spacing_mm: 3.0,
  enable_vascular_channels: false,

  // === Annular Ligaments ===
  enable_annular_ligaments: false,
  ligament_thickness_mm: 0.5,
  annular_ligament_thickness_mm: 0.5,

  // === Carina (for bifurcation) ===
  enable_carina: false,
  carina_angle_deg: 70.0,
  left_bronchus_angle_deg: 45.0,
  right_bronchus_angle_deg: 25.0,
  bronchus_diameter_mm: 12.0,
  bronchus_length_mm: 15.0,

  // === Porosity ===
  scaffold_porosity: 0.5,
  pore_size_um: 100.0,

  // === Randomization ===
  position_noise_mm: 0.0,
  ring_variance_pct: 0.0,
  random_seed: 42,
};

export const DEFAULT_TUBULAR_CONDUIT: TubularConduitParams = {
  resolution: 32,
  outer_diameter_mm: 5.0,
  wall_thickness_mm: 0.5,
  length_mm: 20.0,
  inner_texture: InnerTexture.SMOOTH,
  groove_count: 8,
  groove_depth_mm: 0.15,
};

export const DEFAULT_VASCULAR_PERFUSION_DISH: VascularPerfusionDishParams = {
  resolution: 12,
  // Network structure
  inlets: 4,
  levels: 2,
  splits: 2,
  spread: 0.35,
  ratio: 0.79,
  cone_angle: 60.0,
  curvature: 0.3,
  bottom_height: 0.06,
  // Variation
  radius_variation: 0.25,
  flip_chance: 0.30,
  z_variation: 0.35,
  angle_variation: 0.40,
  collision_buffer: 0.08,
  // Behavior
  even_spread: true,
  deterministic: false,
  tips_down: false,
  // Geometry
  outer_radius_mm: 4.875,
  height_mm: 2.0,
  inlet_radius_mm: 0.35,
  rim_height_mm: 0.2,
  flip_upside_down: false,
  seed: 42,
};
