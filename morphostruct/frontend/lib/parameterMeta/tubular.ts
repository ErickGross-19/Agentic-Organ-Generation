/**
 * Parameter metadata for tubular scaffold types
 * Includes: blood_vessel, nerve_conduit, spinal_cord, bladder, trachea
 */

import { ScaffoldType } from '../types/scaffolds';
import { ParamMeta } from './types';

export const BLOOD_VESSEL_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  inner_diameter_mm: { type: 'number', label: 'Inner Diameter', min: 0.5, max: 30, step: 0.5, unit: 'mm', description: 'Lumen diameter (small artery ~3mm)' },
  wall_thickness_mm: { type: 'number', label: 'Wall Thickness', min: 0.1, max: 5, step: 0.1, unit: 'mm', description: 'Total wall thickness' },
  length_mm: { type: 'number', label: 'Length', min: 5, max: 200, step: 5, unit: 'mm', description: 'Vessel segment length' },

  // === Layer Thicknesses (anatomically accurate) ===
  intima_thickness_um: { type: 'number', label: 'Intima Thickness', min: 10, max: 200, step: 5, unit: 'um', description: '~50 um for healthy artery', advanced: true },
  media_thickness_um: { type: 'number', label: 'Media Thickness', min: 50, max: 1000, step: 25, unit: 'um', description: 'Major structural layer', advanced: true },
  adventitia_thickness_um: { type: 'number', label: 'Adventitia Thickness', min: 25, max: 500, step: 25, unit: 'um', description: 'Outer connective tissue', advanced: true },
  layer_ratios: {
    type: 'array',
    label: 'Layer Ratios',
    itemCount: 3,
    itemLabels: ['Intima', 'Media', 'Adventitia'],
    min: 0,
    max: 1,
    step: 0.05,
    description: 'Thickness ratio for each layer (should sum to 1)',
    advanced: true,
  },

  // === Elastic Laminae (Media Layer) ===
  elastic_lamina_thickness_um: { type: 'number', label: 'Elastic Lamina Thickness', min: 0.5, max: 5, step: 0.1, unit: 'um', description: 'Internal elastic lamina', advanced: true },
  external_elastic_lamina_thickness_um: { type: 'number', label: 'External Elastic Lamina Thickness', min: 0.5, max: 5, step: 0.1, unit: 'um', description: 'External elastic lamina', advanced: true },
  num_elastic_laminae: { type: 'number', label: 'Number of Elastic Laminae', min: 1, max: 50, step: 1, description: 'Elastic sheets in media', advanced: true },
  enable_elastic_laminae: { type: 'boolean', label: 'Enable Elastic Laminae', description: 'Generate laminae structures', advanced: true },
  fenestration_count: { type: 'number', label: 'Fenestration Count', min: 20, max: 100, step: 5, description: 'Number of fenestrations per lamina (40-70)', advanced: true },
  fenestration_diameter_um: { type: 'number', label: 'Fenestration Diameter', min: 1, max: 5, step: 0.1, unit: 'um', description: 'Fenestration hole diameter (2-3um)', advanced: true },

  // === Smooth Muscle Cells (Media) ===
  smc_length_um: { type: 'number', label: 'SMC Length', min: 50, max: 200, step: 10, unit: 'um', description: 'Smooth muscle cell length. Typical range 100-200µm', advanced: true },
  smc_orientation_angle_deg: { type: 'number', label: 'SMC Orientation Angle', min: -90, max: 90, step: 5, unit: 'deg', description: 'Helical angle (0=circumferential)', advanced: true },
  enable_smc_alignment: { type: 'boolean', label: 'Enable SMC Alignment', description: 'Generate aligned SMC structures', advanced: true },
  smc_density_per_mm2: { type: 'number', label: 'SMC Density', min: 100, max: 1000, step: 50, unit: '/mm²', description: 'Smooth muscle cell markers per mm²', advanced: true },

  // === Endothelial Features (Intima) ===
  enable_endothelial_texture: { type: 'boolean', label: 'Enable Endothelial Texture', description: 'Inner surface texture', advanced: true },
  endothelial_cell_size_um: { type: 'number', label: 'Endothelial Cell Size', min: 10, max: 50, step: 5, unit: 'um', description: 'Typical EC size ~30um', advanced: true },
  endothelial_cell_density_per_mm2: { type: 'number', label: 'Endothelial Cell Density', min: 1000, max: 5000, step: 250, unit: '/mm²', description: 'Endothelial cells per mm² (2000-3000)', advanced: true },
  endothelial_bump_height_um: { type: 'number', label: 'Endothelial Bump Height', min: 1, max: 10, step: 0.5, unit: 'um', description: 'Height of cobblestone bumps', advanced: true },

  // === Adventitial Features ===
  enable_vasa_vasorum: { type: 'boolean', label: 'Enable Vasa Vasorum', description: 'Small vessels in adventitia', advanced: true },
  vasa_vasorum_diameter_um: { type: 'number', label: 'Vasa Vasorum Diameter', min: 20, max: 100, step: 10, unit: 'um', description: 'Microvessels feeding wall', advanced: true },
  vasa_vasorum_spacing_um: { type: 'number', label: 'Vasa Vasorum Spacing', min: 200, max: 1000, step: 50, unit: 'um', description: 'Distance between vasa vasorum', advanced: true },
  vasa_vasorum_density_per_mm2: { type: 'number', label: 'Vasa Vasorum Density', min: 10, max: 100, step: 5, unit: '/mm²', description: 'Density (20-40/mm²)', advanced: true },

  // === Bifurcation ===
  enable_bifurcation: { type: 'boolean', label: 'Enable Bifurcation', description: 'Enable Y-bifurcation', advanced: true },
  bifurcation_angle_deg: { type: 'number', label: 'Bifurcation Angle', min: 20, max: 90, step: 2, unit: 'deg', description: 'Y-bifurcation angle (36 deg typical)', advanced: true },
  daughter_diameter_ratio: { type: 'number', label: 'Daughter Diameter Ratio', min: 0.5, max: 1, step: 0.01, description: "Murray's law: r_daughter/r_parent (0.79)", advanced: true },

  // === Porosity and Permeability ===
  scaffold_porosity: { type: 'number', label: 'Scaffold Porosity', min: 0.1, max: 0.9, step: 0.05, description: 'Target porosity (0-1)', advanced: true },
  pore_size_um: { type: 'number', label: 'Pore Size', min: 25, max: 300, step: 25, unit: 'um', description: 'For cell infiltration', advanced: true },
  enable_radial_pores: { type: 'boolean', label: 'Enable Radial Pores', description: 'Transmural pores', advanced: true },
  enable_pore_network: { type: 'boolean', label: 'Enable Pore Network', description: 'Interconnected pore system', advanced: true },

  // === Vessel Geometry Variations ===
  vessel_taper: { type: 'number', label: 'Vessel Taper', min: 0, max: 0.3, step: 0.01, description: 'Diameter reduction along length (0-0.3)', advanced: true },
  taper_ratio: { type: 'number', label: 'Taper Ratio', min: 0.5, max: 1.5, step: 0.05, description: 'End diameter / start diameter (1.0 = no taper)', advanced: true },
  vessel_tortuosity: { type: 'number', label: 'Vessel Tortuosity', min: 0, max: 1, step: 0.05, description: 'Waviness/curvature (0-1)', advanced: true },
  tortuosity_index: { type: 'number', label: 'Tortuosity Index', min: 1, max: 2, step: 0.05, description: 'Actual path length / straight distance (1.0 = straight)', advanced: true },
  tortuosity_amplitude_ratio: { type: 'number', label: 'Tortuosity Amplitude', min: 0.05, max: 0.3, step: 0.01, description: 'Wave amplitude relative to diameter', advanced: true },
  tortuosity_wavelength_mm: { type: 'number', label: 'Tortuosity Wavelength', min: 1, max: 20, step: 0.5, unit: 'mm', description: 'Wave period for tortuosity', advanced: true },

  // === Mechanical Properties (for FEA export) ===
  target_compliance_percent_per_100mmHg: { type: 'number', label: 'Target Compliance', min: 1, max: 15, step: 0.5, unit: '%/100mmHg', description: 'Compliance target', advanced: true },
  burst_pressure_mmHg: { type: 'number', label: 'Burst Pressure', min: 500, max: 5000, step: 100, unit: 'mmHg', description: 'Design specification', advanced: true },

  // === Randomization & Resolution ===
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.3, step: 0.05, description: 'Random variation in structural features (0-0.3)', advanced: true },
  random_seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 64, step: 2, description: 'Cylinder segments', advanced: true },
};

export const NERVE_CONDUIT_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  outer_diameter_mm: { type: 'number', label: 'Outer Diameter', min: 1, max: 10, step: 0.25, unit: 'mm', description: 'Conduit outer diameter' },
  inner_diameter_mm: { type: 'number', label: 'Inner Diameter', min: 0.5, max: 9, step: 0.25, unit: 'mm', description: 'Lumen diameter (derived from wall)', advanced: true },
  wall_thickness_mm: { type: 'number', label: 'Wall Thickness', min: 0.1, max: 2, step: 0.1, unit: 'mm', description: 'Conduit wall thickness' },
  length_mm: { type: 'number', label: 'Length', min: 5, max: 100, step: 5, unit: 'mm', description: 'Total conduit length' },

  // === Channel Structure ===
  num_channels: { type: 'number', label: 'Number of Channels', min: 1, max: 100, step: 1, description: 'Number of guidance channels' },
  channel_diameter_um: { type: 'number', label: 'Channel Diameter', min: 50, max: 500, step: 10, unit: 'um', description: 'Individual channel diameter (100-300um for axons)' },
  channel_spacing_um: { type: 'number', label: 'Channel Spacing', min: 25, max: 300, step: 25, unit: 'um', description: 'Minimum spacing between channels', advanced: true },

  // === Fascicle Architecture ===
  fascicle_diameter_um: { type: 'number', label: 'Fascicle Diameter', min: 100, max: 1000, step: 50, unit: 'um', description: 'Fascicle bundle diameter', advanced: true },
  fascicle_spacing_um: { type: 'number', label: 'Fascicle Spacing', min: 50, max: 300, step: 25, unit: 'um', description: 'Distance between fascicles', advanced: true },
  num_fascicles: { type: 'number', label: 'Number of Fascicles', min: 1, max: 12, step: 1, description: 'Number of fascicle compartments', advanced: true },
  enable_fascicle_chambers: { type: 'boolean', label: 'Enable Fascicle Chambers', description: 'Create fascicle compartments', advanced: true },

  // === Perineurium Layer ===
  perineurium_thickness_um: { type: 'number', label: 'Perineurium Thickness', min: 5, max: 50, step: 1, unit: 'um', description: 'Perineurial sheath (~8-20um)', advanced: true },
  enable_perineurium: { type: 'boolean', label: 'Enable Perineurium', description: 'Add perineurium around fascicles', advanced: true },

  // === Epineurium Layer ===
  epineurium_thickness_um: { type: 'number', label: 'Epineurium Thickness', min: 25, max: 300, step: 25, unit: 'um', description: 'Outer connective tissue', advanced: true },
  enable_epineurium: { type: 'boolean', label: 'Enable Epineurium', description: 'Add epineurial sheath', advanced: true },

  // === Schwann Cell Guidance Features ===
  ridge_width_um: { type: 'number', label: 'Ridge Width', min: 10, max: 100, step: 5, unit: 'um', description: 'Microridge width for Schwann cells', advanced: true },
  groove_width_um: { type: 'number', label: 'Groove Width', min: 10, max: 100, step: 5, unit: 'um', description: 'Microgroove width', advanced: true },
  groove_depth_um: { type: 'number', label: 'Groove Depth', min: 1, max: 20, step: 1, unit: 'um', description: 'Groove depth', advanced: true },
  enable_microgrooves: { type: 'boolean', label: 'Enable Microgrooves', description: 'Surface microgrooves for cell alignment', advanced: true },

  // === Inner Surface Features ===
  enable_guidance_channels: { type: 'boolean', label: 'Enable Guidance Channels', description: 'Internal guidance channels' },
  guidance_channel_diameter_um: { type: 'number', label: 'Guidance Channel Diameter', min: 20, max: 150, step: 10, unit: 'um', description: 'Smaller guidance microchannels', advanced: true },
  guidance_channel_pattern: { type: 'enum', label: 'Guidance Channel Pattern', options: [{ value: 'linear', label: 'Linear' }, { value: 'spiral', label: 'Spiral' }], description: 'Channel arrangement pattern', advanced: true },

  // === Wall Porosity ===
  wall_porosity: { type: 'number', label: 'Wall Porosity', min: 0.3, max: 0.95, step: 0.05, description: 'Target wall porosity for nutrient diffusion', advanced: true },
  pore_size_um: { type: 'number', label: 'Pore Size', min: 10, max: 150, step: 10, unit: 'um', description: 'Wall pore size', advanced: true },
  inner_surface_porosity: { type: 'number', label: 'Inner Surface Porosity', min: 0.1, max: 0.7, step: 0.05, description: 'Inner surface porosity', advanced: true },

  // === Microgrooves ===
  num_microgrooves: { type: 'number', label: 'Number of Microgrooves', min: 0, max: 100, step: 4, description: 'Auto-calculated if 0', advanced: true },
  groove_spacing_um: { type: 'number', label: 'Groove Spacing', min: 20, max: 100, step: 5, unit: 'um', description: 'Distance between microgrooves', advanced: true },

  // === Growth Factor Delivery ===
  enable_growth_factor_reservoirs: { type: 'boolean', label: 'Enable Growth Factor Reservoirs', description: 'Drug delivery reservoirs', advanced: true },
  reservoir_diameter_um: { type: 'number', label: 'Reservoir Diameter', min: 100, max: 700, step: 50, unit: 'um', description: 'Reservoir size', advanced: true },
  reservoir_count: { type: 'number', label: 'Reservoir Count', min: 1, max: 12, step: 1, description: 'Number of reservoirs', advanced: true },
  reservoir_rings: { type: 'number', label: 'Reservoir Rings', min: 0, max: 10, step: 1, description: 'Number of circumferential reservoir rings', advanced: true },
  reservoir_spacing_mm: { type: 'number', label: 'Reservoir Spacing', min: 1, max: 10, step: 0.5, unit: 'mm', description: 'Axial spacing between reservoirs', advanced: true },

  // === Suture Features ===
  enable_biodegradable_suture_holes: { type: 'boolean', label: 'Enable Suture Holes', description: 'Pre-made suture holes', advanced: true },
  suture_hole_diameter_mm: { type: 'number', label: 'Suture Hole Diameter', min: 0.2, max: 1, step: 0.1, unit: 'mm', description: 'Suture hole size', advanced: true },
  suture_hole_count: { type: 'number', label: 'Suture Hole Count', min: 2, max: 8, step: 1, description: 'Number per end', advanced: true },
  suture_hole_distance_from_end_mm: { type: 'number', label: 'Suture Hole Distance from End', min: 1, max: 5, step: 0.25, unit: 'mm', description: 'Distance from end (2-3mm)', advanced: true },

  // === Geometry Variations ===
  taper_ratio: { type: 'number', label: 'Taper Ratio', min: 0.5, max: 1.5, step: 0.05, description: 'End diameter / start diameter (1.0 = no taper)', advanced: true },
  enable_flared_ends: { type: 'boolean', label: 'Enable Flared Ends', description: 'Flared ends for easier anastomosis', advanced: true },
  flare_angle_deg: { type: 'number', label: 'Flare Angle', min: 5, max: 45, step: 5, unit: 'deg', description: 'Flare angle', advanced: true },
  flare_length_mm: { type: 'number', label: 'Flare Length', min: 0.5, max: 3, step: 0.25, unit: 'mm', description: 'Length of flared region', advanced: true },

  // === Randomization & Resolution ===
  position_noise_um: { type: 'number', label: 'Position Noise', min: 0, max: 50, step: 5, unit: 'um', description: 'Random variation in channel positions', advanced: true },
  channel_variance: { type: 'number', label: 'Channel Variance', min: 0, max: 0.5, step: 0.05, description: 'Variation in channel diameter (0-0.5)', advanced: true },
  random_seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 6, max: 32, step: 2, description: 'Cylinder segments', advanced: true },
};

export const SPINAL_CORD_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  total_diameter_mm: { type: 'number', label: 'Total Diameter', min: 5, max: 20, step: 0.5, unit: 'mm', description: 'Overall cord diameter (cervical ~13mm, thoracic ~10mm)' },
  length_mm: { type: 'number', label: 'Length', min: 10, max: 200, step: 10, unit: 'mm', description: 'Segment length' },

  // === Gray Matter Configuration ===
  gray_matter_volume_ratio: { type: 'number', label: 'Gray Matter Volume Ratio', min: 0.1, max: 0.5, step: 0.05, description: 'Gray matter as fraction of total (~25-35%). Literature: 16-25% of cord cross-section' },
  gray_matter_pattern: { type: 'enum', label: 'Gray Matter Pattern', options: [{ value: 'butterfly', label: 'Butterfly' }, { value: 'h_shape', label: 'H-Shape' }, { value: 'simplified', label: 'Simplified' }], description: 'Gray matter cross-section shape' },

  // === Gray Matter Horns ===
  dorsal_horn_width_mm: { type: 'number', label: 'Dorsal Horn Width', min: 0.5, max: 5, step: 0.25, unit: 'mm', description: 'Sensory horn width', advanced: true },
  dorsal_horn_height_mm: { type: 'number', label: 'Dorsal Horn Height', min: 0.5, max: 6, step: 0.25, unit: 'mm', description: 'Dorsal horn vertical extent', advanced: true },
  ventral_horn_width_mm: { type: 'number', label: 'Ventral Horn Width', min: 0.5, max: 6, step: 0.25, unit: 'mm', description: 'Motor horn width (larger)', advanced: true },
  ventral_horn_height_mm: { type: 'number', label: 'Ventral Horn Height', min: 0.5, max: 5, step: 0.25, unit: 'mm', description: 'Ventral horn vertical extent', advanced: true },
  lateral_horn_present: { type: 'boolean', label: 'Lateral Horn Present', description: 'T1-L2 sympathetic (autonomic)', advanced: true },
  lateral_horn_size_mm: { type: 'number', label: 'Lateral Horn Size', min: 0.2, max: 2, step: 0.1, unit: 'mm', description: 'Lateral horn dimension', advanced: true },

  // === Central Canal ===
  central_canal_diameter_mm: { type: 'number', label: 'Central Canal Diameter', min: 0.05, max: 1, step: 0.05, unit: 'mm', description: 'Ependymal-lined canal (~0.1-0.5mm)', advanced: true },
  enable_central_canal: { type: 'boolean', label: 'Enable Central Canal', description: 'Include central canal' },

  // === White Matter ===
  white_matter_thickness_mm: { type: 'number', label: 'White Matter Thickness', min: 1, max: 5, step: 0.25, unit: 'mm', description: 'Surrounding white matter thickness', advanced: true },
  enable_tract_columns: { type: 'boolean', label: 'Enable Tract Columns', description: 'Distinct dorsal/lateral/ventral columns', advanced: true },
  dorsal_column_width_mm: { type: 'number', label: 'Dorsal Column Width', min: 1, max: 6, step: 0.5, unit: 'mm', description: 'Posterior funiculus', advanced: true },
  lateral_column_width_mm: { type: 'number', label: 'Lateral Column Width', min: 1, max: 8, step: 0.5, unit: 'mm', description: 'Lateral funiculus', advanced: true },
  ventral_column_width_mm: { type: 'number', label: 'Ventral Column Width', min: 1, max: 7, step: 0.5, unit: 'mm', description: 'Anterior funiculus', advanced: true },

  // === Anterior Median Fissure ===
  enable_anterior_fissure: { type: 'boolean', label: 'Enable Anterior Fissure', description: 'V-shaped groove on ventral surface', advanced: true },
  anterior_fissure_depth_mm: { type: 'number', label: 'Anterior Fissure Depth', min: 1, max: 5, step: 0.25, unit: 'mm', description: 'Depth into ventral white matter', advanced: true },
  anterior_fissure_width_mm: { type: 'number', label: 'Anterior Fissure Width', min: 0.5, max: 2, step: 0.1, unit: 'mm', description: 'Width at surface', advanced: true },

  // === Guidance Channels (for regeneration) ===
  num_guidance_channels: { type: 'number', label: 'Number of Guidance Channels', min: 0, max: 100, step: 5, description: 'Channels for axon guidance' },
  channel_diameter_um: { type: 'number', label: 'Channel Diameter', min: 50, max: 500, step: 25, unit: 'um', description: 'Individual channel diameter' },
  channel_pattern: { type: 'enum', label: 'Channel Pattern', options: [{ value: 'radial', label: 'Radial' }, { value: 'grid', label: 'Grid' }, { value: 'random', label: 'Random' }], description: 'Channel arrangement pattern', advanced: true },
  channel_region: { type: 'enum', label: 'Channel Region', options: [{ value: 'white_matter', label: 'White Matter' }, { value: 'gray_matter', label: 'Gray Matter' }, { value: 'both', label: 'Both' }], description: 'Where to place channels', advanced: true },

  // === Meningeal Layers ===
  enable_pia_mater: { type: 'boolean', label: 'Enable Pia Mater', description: 'Innermost, adherent to cord', advanced: true },
  pia_mater_thickness_um: { type: 'number', label: 'Pia Mater Thickness', min: 10, max: 200, step: 10, unit: 'um', description: '~20-100um', advanced: true },
  enable_arachnoid_mater: { type: 'boolean', label: 'Enable Arachnoid Mater', description: 'Middle, delicate layer', advanced: true },
  arachnoid_thickness_um: { type: 'number', label: 'Arachnoid Thickness', min: 25, max: 300, step: 25, unit: 'um', description: 'Arachnoid layer thickness', advanced: true },
  enable_dura_mater: { type: 'boolean', label: 'Enable Dura Mater', description: 'Outermost, tough layer', advanced: true },
  dura_mater_thickness_mm: { type: 'number', label: 'Dura Mater Thickness', min: 0.1, max: 1, step: 0.1, unit: 'mm', description: '~0.3-0.5mm', advanced: true },
  subarachnoid_space_mm: { type: 'number', label: 'Subarachnoid Space', min: 0.5, max: 5, step: 0.25, unit: 'mm', description: 'CSF space', advanced: true },

  // === Root Entry/Exit Zones ===
  enable_root_entry_zones: { type: 'boolean', label: 'Enable Root Entry Zones', description: 'Dorsal root entry points', advanced: true },
  dorsal_root_diameter_mm: { type: 'number', label: 'Dorsal Root Diameter', min: 0.3, max: 2, step: 0.1, unit: 'mm', description: 'Dorsal root diameter', advanced: true },
  ventral_root_diameter_mm: { type: 'number', label: 'Ventral Root Diameter', min: 0.2, max: 1.5, step: 0.1, unit: 'mm', description: 'Ventral root diameter', advanced: true },
  root_spacing_mm: { type: 'number', label: 'Root Spacing', min: 5, max: 30, step: 1, unit: 'mm', description: 'Segmental spacing', advanced: true },
  root_angle_deg: { type: 'number', label: 'Root Angle', min: 20, max: 70, step: 5, unit: 'deg', description: 'Angle of root emergence', advanced: true },

  // === Vascular Features ===
  enable_anterior_spinal_artery: { type: 'boolean', label: 'Enable Anterior Spinal Artery', description: 'Anterior spinal artery', advanced: true },
  anterior_spinal_artery_diameter_mm: { type: 'number', label: 'Anterior Spinal Artery Diameter', min: 0.3, max: 1.5, step: 0.1, unit: 'mm', description: 'ASA diameter', advanced: true },
  enable_posterior_spinal_arteries: { type: 'boolean', label: 'Enable Posterior Spinal Arteries', description: 'Posterior spinal arteries', advanced: true },
  posterior_spinal_artery_diameter_mm: { type: 'number', label: 'Posterior Spinal Artery Diameter', min: 0.1, max: 1, step: 0.1, unit: 'mm', description: 'PSA diameter', advanced: true },

  // === Porosity & Resolution ===
  scaffold_porosity: { type: 'number', label: 'Scaffold Porosity', min: 0.3, max: 0.9, step: 0.05, description: 'Target porosity', advanced: true },
  pore_size_um: { type: 'number', label: 'Pore Size', min: 25, max: 300, step: 25, unit: 'um', description: 'For cell infiltration', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.5, step: 0.05, description: 'Variation in structures', advanced: true },
  random_seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 12, max: 64, step: 2, description: 'Cylinder segments', advanced: true },
};

export const BLADDER_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  dome_diameter_mm: { type: 'number', label: 'Dome Diameter', min: 30, max: 150, step: 5, unit: 'mm', description: 'Bladder diameter when moderately full (~50-100mm)' },
  wall_thickness_empty_mm: { type: 'number', label: 'Wall Thickness (Empty)', min: 1, max: 8, step: 0.5, unit: 'mm', description: 'Wall thickness when empty (~3-5mm)' },
  dome_curvature: { type: 'number', label: 'Dome Curvature', min: 0.3, max: 1, step: 0.05, description: '0.5 = hemisphere, 1.0 = full sphere' },
  dome_height_mm: { type: 'number', label: 'Dome Height', min: 15, max: 150, step: 5, unit: 'mm', description: 'Auto-calculated if not set', advanced: true },

  // === Layer Configuration (detailed) ===
  urothelium_thickness_um: { type: 'number', label: 'Urothelium Thickness', min: 50, max: 400, step: 25, unit: 'um', description: '~100-200 um, 3-7 cell layers', advanced: true },
  enable_urothelium: { type: 'boolean', label: 'Enable Urothelium', description: 'Innermost epithelial layer', advanced: true },
  lamina_propria_thickness_um: { type: 'number', label: 'Lamina Propria Thickness', min: 200, max: 1500, step: 100, unit: 'um', description: '~500-1000 um submucosa', advanced: true },
  enable_lamina_propria: { type: 'boolean', label: 'Enable Lamina Propria', description: 'Submucosa layer', advanced: true },
  detrusor_thickness_mm: { type: 'number', label: 'Detrusor Thickness', min: 0.5, max: 5, step: 0.25, unit: 'mm', description: 'Major structural muscle layer', advanced: true },
  enable_detrusor_muscle: { type: 'boolean', label: 'Enable Detrusor Muscle', description: 'Smooth muscle layer', advanced: true },
  detrusor_layer_count: { type: 'number', label: 'Detrusor Layer Count', min: 1, max: 5, step: 1, description: 'Inner/middle/outer muscle layers', advanced: true },
  serosa_thickness_um: { type: 'number', label: 'Serosa Thickness', min: 50, max: 500, step: 50, unit: 'um', description: '~100-300 um outer covering', advanced: true },
  enable_serosa: { type: 'boolean', label: 'Enable Serosa', description: 'Outer covering layer', advanced: true },
  layer_count: { type: 'number', label: 'Layer Count', min: 2, max: 4, step: 1, description: 'Simple layer count (overridden by detailed layers)' },

  // === Rugae (Mucosal Folds) ===
  rugae_height_um: { type: 'number', label: 'Rugae Height', min: 200, max: 1500, step: 100, unit: 'um', description: 'Fold height when contracted (~500-1000 um)', advanced: true },
  rugae_spacing_mm: { type: 'number', label: 'Rugae Spacing', min: 1, max: 5, step: 0.5, unit: 'mm', description: 'Distance between folds', advanced: true },
  enable_urothelial_folds: { type: 'boolean', label: 'Enable Urothelial Folds', description: 'Generate rugae structures', advanced: true },
  fold_count: { type: 'number', label: 'Fold Count', min: 4, max: 24, step: 2, description: 'Number of radial folds', advanced: true },
  fold_variance: { type: 'number', label: 'Fold Variance', min: 0, max: 0.5, step: 0.05, description: 'Random variation in fold depth', advanced: true },

  // === Compliance and Mechanics ===
  compliance_ml_per_cmH2O: { type: 'number', label: 'Compliance', min: 5, max: 60, step: 5, unit: 'ml/cmH2O', description: 'Bladder compliance (design spec)', advanced: true },
  max_capacity_ml: { type: 'number', label: 'Max Capacity', min: 100, max: 1000, step: 50, unit: 'ml', description: 'Target capacity', advanced: true },

  // === Trigone Region ===
  trigone_marker: { type: 'boolean', label: 'Trigone Marker', description: 'Mark triangular smooth muscle region at base', advanced: true },
  trigone_width_mm: { type: 'number', label: 'Trigone Width', min: 10, max: 50, step: 5, unit: 'mm', description: 'Width of trigone', advanced: true },
  trigone_height_mm: { type: 'number', label: 'Trigone Height', min: 15, max: 60, step: 5, unit: 'mm', description: 'Vertical extent of trigone', advanced: true },

  // === Ureteral/Urethral Openings ===
  enable_ureteral_openings: { type: 'boolean', label: 'Enable Ureteral Openings', description: 'Two ureteral orifices', advanced: true },
  ureteral_opening_diameter_mm: { type: 'number', label: 'Ureteral Opening Diameter', min: 1, max: 6, step: 0.5, unit: 'mm', description: '~2-4 mm', advanced: true },
  ureteral_spacing_mm: { type: 'number', label: 'Ureteral Spacing', min: 10, max: 50, step: 5, unit: 'mm', description: 'Distance between ureters', advanced: true },
  urethral_opening_diameter_mm: { type: 'number', label: 'Urethral Opening Diameter', min: 3, max: 12, step: 1, unit: 'mm', description: 'Internal urethral orifice (~5-8mm)', advanced: true },
  enable_urethral_opening: { type: 'boolean', label: 'Enable Urethral Opening', description: 'Include urethral orifice', advanced: true },

  // === Vascular Network ===
  enable_suburothelial_capillaries: { type: 'boolean', label: 'Enable Suburothelial Capillaries', description: 'Capillary network in lamina propria', advanced: true },
  capillary_diameter_um: { type: 'number', label: 'Capillary Diameter', min: 3, max: 15, step: 1, unit: 'um', description: '~5-10 um', advanced: true },
  capillary_spacing_mm: { type: 'number', label: 'Capillary Spacing', min: 0.2, max: 1, step: 0.1, unit: 'mm', description: 'Spacing in lamina propria', advanced: true },

  // === Muscle Bundle Features ===
  muscle_bundle_diameter_mm: { type: 'number', label: 'Muscle Bundle Diameter', min: 0.1, max: 0.8, step: 0.05, unit: 'mm', description: 'Detrusor muscle bundle size', advanced: true },
  muscle_bundle_spacing_mm: { type: 'number', label: 'Muscle Bundle Spacing', min: 0.2, max: 1, step: 0.1, unit: 'mm', description: 'Spacing between bundles', advanced: true },
  enable_muscle_bundles: { type: 'boolean', label: 'Enable Muscle Bundles', description: 'Generate discrete bundles', advanced: true },

  // === Nerve Innervation Markers ===
  enable_nerve_markers: { type: 'boolean', label: 'Enable Nerve Markers', description: 'Mark innervation points in bladder wall', advanced: true },
  nerve_density_per_cm2: { type: 'number', label: 'Nerve Density', min: 10, max: 150, step: 10, unit: '/cm²', description: 'Nerve ending density', advanced: true },

  // === Porosity ===
  scaffold_porosity: { type: 'number', label: 'Scaffold Porosity', min: 0.3, max: 0.9, step: 0.05, description: 'Target porosity', advanced: true },
  pore_size_um: { type: 'number', label: 'Pore Size', min: 25, max: 300, step: 25, unit: 'um', description: 'For cell infiltration', advanced: true },
  enable_pore_network: { type: 'boolean', label: 'Enable Pore Network', description: 'Interconnected pore system', advanced: true },

  // === Randomization ===
  position_noise_mm: { type: 'number', label: 'Position Noise', min: 0, max: 1, step: 0.05, unit: 'mm', description: 'Random position variation', advanced: true },
  random_seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducibility', advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 64, step: 4, description: 'Mesh resolution for geometry generation', advanced: true },
};

export const TRACHEA_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  total_length_mm: { type: 'number', label: 'Total Length', min: 20, max: 200, step: 10, unit: 'mm', description: 'Adult trachea ~10-12 cm' },
  outer_diameter_mm: { type: 'number', label: 'Outer Diameter', min: 10, max: 35, step: 1, unit: 'mm', description: '~15-25 mm outer' },
  inner_diameter_mm: { type: 'number', label: 'Inner Diameter', min: 8, max: 30, step: 1, unit: 'mm', description: '~12-18 mm lumen' },

  // === Cartilage Ring Configuration ===
  num_cartilage_rings: { type: 'number', label: 'Number of Cartilage Rings', min: 1, max: 30, step: 1, description: 'Typically 16-20 rings' },
  ring_height_mm: { type: 'number', label: 'Ring Height', min: 2, max: 8, step: 0.5, unit: 'mm', description: 'Vertical extent of each ring (~3-5 mm)' },
  ring_width_mm: { type: 'number', label: 'Ring Width', min: 2, max: 8, step: 0.5, unit: 'mm', description: 'Superior-inferior extent (vertical height)', advanced: true },
  ring_thickness_mm: { type: 'number', label: 'Ring Thickness', min: 0.5, max: 3, step: 0.25, unit: 'mm', description: 'Radial thickness (~1-2 mm)' },
  posterior_opening_degrees: { type: 'number', label: 'Posterior Opening', min: 30, max: 150, step: 5, unit: 'deg', description: 'Gap for trachealis muscle (~60-100 deg)' },
  ring_gap_deg: { type: 'number', label: 'Ring Gap', min: 30, max: 150, step: 5, unit: 'deg', description: 'Alias for posterior_opening_degrees', advanced: true },
  interring_spacing_mm: { type: 'number', label: 'Interring Spacing', min: 0.5, max: 5, step: 0.25, unit: 'mm', description: 'Annular ligament space (~1-3 mm)' },

  // === Cartilage Properties ===
  cartilage_type: { type: 'enum', label: 'Cartilage Type', options: [{ value: 'hyaline', label: 'Hyaline' }, { value: 'elastic', label: 'Elastic' }, { value: 'fibrocartilage', label: 'Fibrocartilage' }], description: 'Type of cartilage', advanced: true },
  enable_c_shaped_cartilage: { type: 'boolean', label: 'C-Shaped Cartilage', description: 'Use anatomically correct C-shaped vs full rings' },
  cartilage_porosity: { type: 'number', label: 'Cartilage Porosity', min: 0.6, max: 0.95, step: 0.05, description: 'For scaffold porosity', advanced: true },

  // === Mucosal Layer ===
  mucosa_thickness_um: { type: 'number', label: 'Mucosa Thickness', min: 100, max: 800, step: 50, unit: 'um', description: '~300-500 um (epithelium + lamina propria)', advanced: true },
  enable_mucosal_layer: { type: 'boolean', label: 'Enable Mucosal Layer', description: 'Add mucosal layer', advanced: true },
  enable_cilia_markers: { type: 'boolean', label: 'Enable Cilia Markers', description: 'Texture markers for mucociliary apparatus', advanced: true },
  cilia_density: { type: 'number', label: 'Cilia Density', min: 0, max: 1, step: 0.1, description: 'Marker density (0-1)', advanced: true },
  enable_ciliated_epithelium_markers: { type: 'boolean', label: 'Enable Ciliated Epithelium Markers', description: 'Texture markers', advanced: true },
  enable_goblet_cell_markers: { type: 'boolean', label: 'Enable Goblet Cell Markers', description: 'Mucus cell markers', advanced: true },
  epithelium_thickness_um: { type: 'number', label: 'Epithelium Thickness', min: 30, max: 100, step: 10, unit: 'µm', description: 'Ciliated epithelial layer thickness', advanced: true },

  // === Submucosa ===
  submucosa_thickness_um: { type: 'number', label: 'Submucosa Thickness', min: 100, max: 1000, step: 50, unit: 'um', description: '~300-700 um', advanced: true },
  lamina_propria_thickness_um: { type: 'number', label: 'Lamina Propria Thickness', min: 100, max: 500, step: 25, unit: 'µm', description: 'Submucosal connective tissue layer', advanced: true },
  enable_submucosa: { type: 'boolean', label: 'Enable Submucosa', description: 'Add submucosa layer', advanced: true },

  // === Submucosal Glands ===
  enable_submucosal_glands: { type: 'boolean', label: 'Enable Submucosal Glands', description: 'Add submucosal glands', advanced: true },
  gland_diameter_um: { type: 'number', label: 'Gland Diameter (µm)', min: 100, max: 500, step: 25, unit: 'µm', description: '~200-400 µm', advanced: true },
  gland_diameter_mm: { type: 'number', label: 'Gland Diameter', min: 0.1, max: 0.8, step: 0.05, unit: 'mm', description: '~0.2-0.5 mm', advanced: true },
  gland_density_per_mm2: { type: 'number', label: 'Gland Density', min: 0.5, max: 5, step: 0.25, unit: '/mm2', description: 'Glands per mm2', advanced: true },

  // === Trachealis Muscle ===
  enable_trachealis_muscle: { type: 'boolean', label: 'Enable Trachealis Muscle', description: 'Posterior smooth muscle', advanced: true },
  trachealis_thickness_um: { type: 'number', label: 'Trachealis Thickness (µm)', min: 300, max: 1500, step: 100, unit: 'µm', description: '~500-1500 µm (0.5-1.5 mm)', advanced: true },
  trachealis_thickness_mm: { type: 'number', label: 'Trachealis Thickness', min: 0.3, max: 2, step: 0.1, unit: 'mm', description: '~0.5-1.5 mm', advanced: true },
  trachealis_fiber_orientation_deg: { type: 'number', label: 'Fiber Orientation', min: -30, max: 90, step: 5, unit: 'deg', description: 'Smooth muscle fiber angle', advanced: true },
  trachealis_width_deg: { type: 'number', label: 'Trachealis Width', min: 60, max: 150, step: 5, unit: 'deg', description: 'Angular width of trachealis muscle', advanced: true },

  // === Perichondrium ===
  perichondrium_thickness_um: { type: 'number', label: 'Perichondrium Thickness', min: 30, max: 150, step: 10, unit: 'um', description: '~50-100 um', advanced: true },
  enable_perichondrium: { type: 'boolean', label: 'Enable Perichondrium', description: 'Add perichondrium layer', advanced: true },

  // === Mucosal Folds ===
  mucosal_fold_depth_mm: { type: 'number', label: 'Mucosal Fold Depth', min: 0.05, max: 0.5, step: 0.05, unit: 'mm', description: '~0.1-0.3 mm', advanced: true },
  mucosal_fold_count: { type: 'number', label: 'Mucosal Fold Count', min: 4, max: 24, step: 2, description: 'Longitudinal folds', advanced: true },
  enable_mucosal_folds: { type: 'boolean', label: 'Enable Mucosal Folds', description: 'Add mucosal folds', advanced: true },

  // === Vascular Channels ===
  vascular_channel_diameter_um: { type: 'number', label: 'Vascular Channel Diameter (µm)', min: 50, max: 300, step: 25, unit: 'µm', description: 'For scaffold perfusion (~50-200 µm)', advanced: true },
  vascular_channel_diameter_mm: { type: 'number', label: 'Vascular Channel Diameter', min: 0.1, max: 0.5, step: 0.05, unit: 'mm', description: 'For scaffold perfusion', advanced: true },
  vascular_channel_spacing_mm: { type: 'number', label: 'Vascular Channel Spacing', min: 1, max: 8, step: 0.5, unit: 'mm', description: 'Spacing between channels', advanced: true },
  vascular_spacing_um: { type: 'number', label: 'Vascular Spacing', min: 100, max: 500, step: 25, unit: 'µm', description: 'Spacing for vascular network', advanced: true },
  enable_vascular_channels: { type: 'boolean', label: 'Enable Vascular Channels', description: 'Add vascular channels', advanced: true },

  // === Annular Ligaments ===
  enable_annular_ligaments: { type: 'boolean', label: 'Enable Annular Ligaments', description: 'Fibrous tissue between rings', advanced: true },
  annular_ligament_thickness_mm: { type: 'number', label: 'Annular Ligament Thickness', min: 0.1, max: 1.0, step: 0.1, unit: 'mm', description: 'Thickness of annular ligaments between rings', advanced: true },

  // === Carina (for bifurcation) ===
  enable_carina: { type: 'boolean', label: 'Enable Carina', description: 'Bifurcation at distal end', advanced: true },
  carina_angle_deg: { type: 'number', label: 'Carina Angle', min: 40, max: 90, step: 5, unit: 'deg', description: 'Angle between main bronchi', advanced: true },
  bronchus_diameter_mm: { type: 'number', label: 'Bronchus Diameter', min: 6, max: 20, step: 1, unit: 'mm', description: 'Main bronchus diameter', advanced: true },
  bronchus_length_mm: { type: 'number', label: 'Bronchus Length', min: 5, max: 30, step: 2.5, unit: 'mm', description: 'Length of bronchi segments', advanced: true },
  left_bronchus_angle_deg: { type: 'number', label: 'Left Bronchus Angle', min: 30, max: 60, step: 2, unit: 'deg', description: 'More horizontal (~45 deg from vertical)', advanced: true },
  right_bronchus_angle_deg: { type: 'number', label: 'Right Bronchus Angle', min: 15, max: 40, step: 2, unit: 'deg', description: 'Steeper (~25 deg from vertical)', advanced: true },

  // === Porosity & Resolution ===
  scaffold_porosity: { type: 'number', label: 'Scaffold Porosity', min: 0.2, max: 0.8, step: 0.05, description: 'Target porosity', advanced: true },
  pore_size_um: { type: 'number', label: 'Pore Size', min: 25, max: 300, step: 25, unit: 'um', description: 'For cell infiltration', advanced: true },
  position_noise_mm: { type: 'number', label: 'Position Noise', min: 0, max: 0.5, step: 0.05, unit: 'mm', description: 'Random variation', advanced: true },
  ring_variance_pct: { type: 'number', label: 'Ring Variance', min: 0, max: 20, step: 1, unit: '%', description: 'Variation in ring dimensions', advanced: true },
  random_seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 12, max: 64, step: 2, description: 'Cylinder segments', advanced: true },
};

export const TUBULAR_CONDUIT_META: Record<string, ParamMeta> = {
  outer_diameter_mm: { type: 'number', label: 'Outer Diameter', min: 1, max: 20, step: 0.5, unit: 'mm', description: 'Outer diameter of tubular conduit' },
  wall_thickness_mm: { type: 'number', label: 'Wall Thickness', min: 0.3, max: 5, step: 0.1, unit: 'mm', description: 'Wall thickness of conduit' },
  length_mm: { type: 'number', label: 'Length', min: 1, max: 100, step: 1, unit: 'mm', description: 'Total conduit length' },
  inner_texture: {
    type: 'enum',
    label: 'Inner Texture',
    options: [
      { value: 'smooth', label: 'Smooth' },
      { value: 'grooved', label: 'Grooved' },
      { value: 'porous', label: 'Porous' },
    ],
    description: 'Inner surface texture type',
  },
  groove_count: { type: 'number', label: 'Groove Count', min: 2, max: 32, step: 1, description: 'Number of longitudinal grooves (for grooved texture)' },
  groove_depth_mm: { type: 'number', label: 'Groove Depth', min: 0.05, max: 0.5, step: 0.05, unit: 'mm', description: 'Depth of grooves (for grooved texture)' },
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 64, step: 2, description: 'Cylinder segments', advanced: true },
};

export const VASCULAR_PERFUSION_DISH_META: Record<string, ParamMeta> = {
  // === Network Structure ===
  inlets: { type: 'number', label: 'Inlets', min: 1, max: 25, step: 1, description: 'Number of inlet branches' },
  levels: { type: 'number', label: 'Branching Levels', min: 0, max: 8, step: 1, description: 'Depth of branching hierarchy (0 = straight)' },
  splits: { type: 'number', label: 'Splits per Branch', min: 1, max: 6, step: 1, description: 'Number of child branches per split' },
  spread: { type: 'number', label: 'Spread', min: 0.1, max: 0.8, step: 0.05, description: 'Horizontal spread factor' },
  ratio: { type: 'number', label: 'Radius Ratio', min: 0.5, max: 0.95, step: 0.01, description: "Child/parent radius ratio (Murray's law = 0.79)" },
  cone_angle: { type: 'number', label: 'Cone Angle', min: 10, max: 180, step: 5, unit: '°', description: 'Branching cone angle' },
  curvature: { type: 'number', label: 'Curvature', min: 0, max: 1, step: 0.05, description: 'Branch curve intensity' },
  bottom_height: { type: 'number', label: 'Bottom Height', min: 0.02, max: 1, step: 0.02, description: 'Height from bottom where vessels terminate' },

  // === Variation/Randomness ===
  radius_variation: { type: 'number', label: 'Radius Variation', min: 0, max: 1, step: 0.05, description: 'Random radius variation (0-100%)', advanced: true },
  flip_chance: { type: 'number', label: 'Flip Chance', min: 0, max: 0.5, step: 0.05, description: 'Probability of flipping branch direction', advanced: true },
  z_variation: { type: 'number', label: 'Z-axis Variation', min: 0, max: 0.5, step: 0.05, description: 'Random Z-axis variation', advanced: true },
  angle_variation: { type: 'number', label: 'Angle Variation', min: 0, max: 0.5, step: 0.05, description: 'Random angle variation', advanced: true },
  collision_buffer: { type: 'number', label: 'Collision Buffer', min: 0, max: 0.3, step: 0.01, description: 'Buffer distance for collision detection' },

  // === Behavior Flags ===
  even_spread: { type: 'boolean', label: 'Even Spread', description: 'Use evenly distributed inlet positioning (vs directional cone)', advanced: true },
  deterministic: { type: 'boolean', label: 'Deterministic', description: 'Use deterministic straight grid-aligned channels', advanced: true },
  tips_down: { type: 'boolean', label: 'Tips Down', description: 'Terminal branches curve to point downward' },

  // === Geometry Dimensions ===
  outer_radius_mm: { type: 'number', label: 'Outer Radius', min: 1, max: 20, step: 0.5, unit: 'mm', description: 'Outer cylinder radius' },
  height_mm: { type: 'number', label: 'Height', min: 0.5, max: 10, step: 0.1, unit: 'mm', description: 'Total scaffold height' },
  inlet_radius_mm: { type: 'number', label: 'Inlet Radius', min: 0.1, max: 2, step: 0.05, unit: 'mm', description: 'Inlet channel radius' },
  rim_height_mm: { type: 'number', label: 'Rim Height', min: 0, max: 3, step: 0.1, unit: 'mm', description: 'Height of rim around dish edge' },
  flip_upside_down: { type: 'boolean', label: 'Flip Upside Down', description: 'Flip scaffold 180° for export' },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 99999, step: 1, description: 'Seed for reproducible randomization', advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 6, max: 32, step: 2, description: 'Mesh resolution (affects quality and performance)', advanced: true },
};

// Combined export for tubular scaffold types
export const TUBULAR_PARAMETER_META: Partial<Record<ScaffoldType, Record<string, ParamMeta>>> = {
  [ScaffoldType.BLOOD_VESSEL]: BLOOD_VESSEL_META,
  [ScaffoldType.NERVE_CONDUIT]: NERVE_CONDUIT_META,
  [ScaffoldType.SPINAL_CORD]: SPINAL_CORD_META,
  [ScaffoldType.BLADDER]: BLADDER_META,
  [ScaffoldType.TRACHEA]: TRACHEA_META,
  [ScaffoldType.TUBULAR_CONDUIT]: TUBULAR_CONDUIT_META,
  [ScaffoldType.VASCULAR_PERFUSION_DISH]: VASCULAR_PERFUSION_DISH_META,
};
