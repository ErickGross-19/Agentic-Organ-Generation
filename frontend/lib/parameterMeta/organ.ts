/**
 * Parameter metadata for organ scaffold types
 * Includes: hepatic_lobule, cardiac_patch, kidney_tubule, lung_alveoli, pancreatic_islet, liver_sinusoid
 */

import { ScaffoldType } from '../types/scaffolds';
import { ParamMeta, boundingBoxMeta } from './types';

export const HEPATIC_LOBULE_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  num_lobules: { type: 'number', label: 'Number of Lobules', min: 1, max: 19, step: 1, description: 'Number of lobules in honeycomb arrangement (1-19)' },
  lobule_radius: { type: 'number', label: 'Lobule Radius', min: 0.5, max: 3, step: 0.1, unit: 'mm', description: 'Outer radius of each hexagonal lobule' },
  lobule_height: { type: 'number', label: 'Lobule Height', min: 1, max: 10, step: 0.5, unit: 'mm', description: 'Height of lobule prism' },
  wall_thickness: { type: 'number', label: 'Wall Thickness', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Thickness of hexagonal walls' },
  resolution: { type: 'number', label: 'Resolution', min: 4, max: 32, step: 2, description: 'Mesh resolution for cylinders', advanced: true },

  // === Central Vein ===
  central_vein_radius: { type: 'number', label: 'Central Vein Radius', min: 0.05, max: 0.5, step: 0.025, unit: 'mm', description: 'Radius of central vein' },
  cv_entrance_length: { type: 'number', label: 'CV Entrance Length', min: 0.1, max: 0.5, step: 0.05, unit: 'mm', description: 'Length of central vein entrance channels', advanced: true },
  cv_entrance_radius: { type: 'number', label: 'CV Entrance Radius', min: 0.02, max: 0.1, step: 0.01, unit: 'mm', description: 'Radius of entrance channels', advanced: true },
  cv_entrance_count: { type: 'number', label: 'CV Entrance Count', min: 1, max: 10, step: 1, description: 'Number of entrance channels per corner direction', advanced: true },
  cv_entrance_z_randomness: { type: 'number', label: 'CV Entrance Z Randomness', min: 0, max: 1, step: 0.1, description: '0 = evenly spaced, 1 = fully random', advanced: true },
  cv_entrance_min_spacing: { type: 'number', label: 'CV Entrance Min Spacing', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Minimum spacing between entrances', advanced: true },

  // === Portal Triad ===
  portal_vein_radius: { type: 'number', label: 'Portal Vein Radius', min: 0.05, max: 0.3, step: 0.01, unit: 'mm', description: 'Radius of portal veins at corners' },
  hepatic_artery_radius: { type: 'number', label: 'Hepatic Artery Radius', min: 0.02, max: 0.15, step: 0.01, unit: 'mm', description: 'Radius of hepatic arteries', advanced: true },
  bile_duct_radius: { type: 'number', label: 'Bile Duct Radius', min: 0.02, max: 0.15, step: 0.01, unit: 'mm', description: 'Radius of bile ducts', advanced: true },
  show_bile_ducts: { type: 'boolean', label: 'Show Bile Ducts', description: 'Whether to include bile ducts', advanced: true },
  triad_wall_distance: { type: 'number', label: 'Triad Wall Distance', min: 0, max: 0.2, step: 0.025, unit: 'mm', description: 'Distance of triad from corner', advanced: true },
  entrance_length: { type: 'number', label: 'Portal Entrance Length', min: 0.1, max: 0.5, step: 0.05, unit: 'mm', description: 'Length of portal entrance channels', advanced: true },
  entrance_radius: { type: 'number', label: 'Portal Entrance Radius', min: 0.02, max: 0.1, step: 0.01, unit: 'mm', description: 'Radius of portal entrance channels', advanced: true },
  entrance_count: { type: 'number', label: 'Portal Entrance Count', min: 1, max: 10, step: 1, description: 'Number of portal entrances per triad', advanced: true },
  entrance_z_randomness: { type: 'number', label: 'Portal Entrance Z Randomness', min: 0, max: 1, step: 0.1, description: '0 = evenly spaced, 1 = fully random', advanced: true },
  entrance_min_spacing: { type: 'number', label: 'Portal Entrance Min Spacing', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Minimum spacing between entrances', advanced: true },

  // === Sinusoids ===
  show_sinusoids: { type: 'boolean', label: 'Show Sinusoids', description: 'Whether to generate sinusoidal channels' },
  sinusoid_radius: { type: 'number', label: 'Sinusoid Radius', min: 0.01, max: 0.1, step: 0.005, unit: 'mm', description: 'Radius of sinusoid channels', advanced: true },
  sinusoid_count: { type: 'number', label: 'Sinusoid Count', min: 1, max: 12, step: 1, description: 'Number of sinusoids per corner', advanced: true },
  sinusoid_levels: { type: 'number', label: 'Sinusoid Levels', min: 1, max: 6, step: 1, description: 'Number of Z-levels for sinusoids', advanced: true },

  // === Organic Variation ===
  corner_noise: { type: 'number', label: 'Corner Noise', min: 0, max: 1, step: 0.1, description: 'Random perturbation of corner positions (0-1)', advanced: true },
  angle_noise: { type: 'number', label: 'Angle Noise', min: 0, max: 1, step: 0.1, description: 'Random angular perturbation (0-1)', advanced: true },
  stretch_noise: { type: 'number', label: 'Stretch Noise', min: 0, max: 1, step: 0.1, description: 'Random stretch deformation (0-1)', advanced: true },
  size_variance: { type: 'number', label: 'Size Variance', min: 0, max: 1, step: 0.1, description: 'Variation in lobule sizes (0-1)', advanced: true },
  edge_curve: { type: 'number', label: 'Edge Curve', min: 0, max: 1, step: 0.1, description: 'Curvature of hexagon edges (0-1)', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Random seed for reproducibility', advanced: true },

  // === Collectors ===
  show_hepatic_collector: { type: 'boolean', label: 'Show Hepatic Collector', description: 'Show hepatic vein collector network', advanced: true },
  show_portal_collector: { type: 'boolean', label: 'Show Portal Collector', description: 'Show portal vein collector network', advanced: true },
  hepatic_collector_height: { type: 'number', label: 'Hepatic Collector Height', min: 0.5, max: 5, step: 0.25, unit: 'mm', description: 'Height of hepatic collector above lobules', advanced: true },
  portal_collector_height: { type: 'number', label: 'Portal Collector Height', min: 0.5, max: 5, step: 0.25, unit: 'mm', description: 'Height of portal collector below lobules', advanced: true },
};

export const CARDIAC_PATCH_META: Record<string, ParamMeta> = {
  // === Cardiomyocyte Dimensions ===
  cardiomyocyte_length: { type: 'number', label: 'Cardiomyocyte Length', min: 80, max: 150, step: 5, unit: 'um', description: '80-150 um typical', advanced: true },
  cardiomyocyte_diameter: { type: 'number', label: 'Cardiomyocyte Diameter', min: 10, max: 35, step: 1, unit: 'um', description: '10-35 um typical', advanced: true },
  sarcomere_length: { type: 'number', label: 'Sarcomere Length', min: 1.8, max: 2.2, step: 0.1, unit: 'um', description: '1.8-2.2 um at rest', advanced: true },

  // === Fiber Architecture ===
  fiber_diameter: { type: 'number', label: 'Fiber Diameter', min: 50, max: 200, step: 10, unit: 'um', description: '50-100 um for fiber bundles' },
  fiber_spacing: { type: 'number', label: 'Fiber Spacing', min: 100, max: 500, step: 25, unit: 'um', description: 'Based on cardiomyocyte + ECM' },
  fiber_alignment_degree: { type: 'number', label: 'Fiber Alignment Degree', min: 0, max: 1, step: 0.1, description: '0-1 (1 = perfectly aligned)', advanced: true },

  // === Layer Configuration ===
  epicardial_helix_angle: { type: 'number', label: 'Epicardial Helix Angle', min: -90, max: 90, step: 5, unit: 'deg', description: '-60 to -70° typical (left-handed helix), default: -45°' },
  endocardial_helix_angle: { type: 'number', label: 'Endocardial Helix Angle', min: -90, max: 90, step: 5, unit: 'deg', description: '+60 to +80° typical (right-handed helix), default: 45°', advanced: true },
  layer_count: { type: 'number', label: 'Layer Count', min: 1, max: 10, step: 1, description: 'Number of fiber layers' },
  enable_helix_gradient: { type: 'boolean', label: 'Enable Helix Gradient', description: 'Gradient from endo to epicardial', advanced: true },

  // === Patch Dimensions ===
  patch_length: { type: 'number', label: 'Patch Length', min: 1, max: 30, step: 1, unit: 'mm', description: 'Patch length in mm' },
  patch_width: { type: 'number', label: 'Patch Width', min: 1, max: 30, step: 1, unit: 'mm', description: 'Patch width in mm' },
  patch_thickness: { type: 'number', label: 'Patch Thickness', min: 0.5, max: 5, step: 0.1, unit: 'mm', description: 'Native: 10-15mm, scaffold: 0.5-3mm' },

  // === Porosity ===
  porosity: { type: 'number', label: 'Porosity', min: 0.6, max: 0.9, step: 0.05, description: 'Target void fraction (70-85% typical)', advanced: true },
  pore_size: { type: 'number', label: 'Pore Size', min: 20, max: 100, step: 5, unit: 'um', description: '20-100 um optimal for cardiac cells', advanced: true },
  enable_interconnected_pores: { type: 'boolean', label: 'Enable Interconnected Pores', description: 'Interconnected pore network', advanced: true },
  pore_interconnectivity: { type: 'number', label: 'Pore Interconnectivity', min: 0.5, max: 1, step: 0.05, description: '0-1 connectivity ratio', advanced: true },

  // === Vascularization Features ===
  enable_capillary_channels: { type: 'boolean', label: 'Enable Capillary Channels', description: 'Channels for vascularization' },
  capillary_diameter: { type: 'number', label: 'Capillary Diameter', min: 5, max: 15, step: 1, unit: 'um', description: '5-10 um typical', advanced: true },
  capillary_density: { type: 'number', label: 'Capillary Density', min: 2000, max: 4000, step: 250, unit: '/mm2', description: '2500-3500/mm2 typical', advanced: true },
  capillary_spacing: { type: 'number', label: 'Capillary Spacing', min: 10, max: 50, step: 5, unit: 'um', description: 'Based on oxygen diffusion ~200um max', advanced: true },

  // === Mechanical Features ===
  enable_cross_fibers: { type: 'boolean', label: 'Enable Cross Fibers', description: 'Perpendicular fiber reinforcement', advanced: true },
  cross_fiber_ratio: { type: 'number', label: 'Cross Fiber Ratio', min: 0.1, max: 0.5, step: 0.05, description: 'Ratio of cross to main fibers', advanced: true },
  enable_fiber_crimping: { type: 'boolean', label: 'Enable Fiber Crimping', description: 'Wavy fiber pattern', advanced: true },
  crimp_amplitude: { type: 'number', label: 'Crimp Amplitude', min: 2, max: 10, step: 1, unit: 'um', description: 'Crimp wave amplitude', advanced: true },
  crimp_wavelength: { type: 'number', label: 'Crimp Wavelength', min: 25, max: 100, step: 5, unit: 'um', description: 'Crimp wave wavelength', advanced: true },

  // === Electrical Guidance ===
  enable_conduction_channels: { type: 'boolean', label: 'Enable Conduction Channels', description: 'Channels for electrical propagation', advanced: true },
  conduction_channel_diameter: { type: 'number', label: 'Conduction Channel Diameter', min: 50, max: 200, step: 10, unit: 'um', description: 'Conduction channel size', advanced: true },
  conduction_channel_spacing: { type: 'number', label: 'Conduction Channel Spacing', min: 250, max: 1000, step: 50, unit: 'um', description: 'Spacing between channels', advanced: true },

  // === Stochastic Variation ===
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 1, step: 0.1, description: 'Random position jitter (0-1)', advanced: true },
  diameter_variance: { type: 'number', label: 'Diameter Variance', min: 0, max: 1, step: 0.1, description: 'Random diameter variation (0-1)', advanced: true },
  angle_noise: { type: 'number', label: 'Angle Noise', min: 0, max: 1, step: 0.1, description: 'Random angle variation (0-1)', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Random seed for reproducibility', advanced: true },

  // === Resolution ===
  resolution: { type: 'number', label: 'Resolution', min: 4, max: 16, step: 2, description: 'Cylinder segments', advanced: true },
};

export const KIDNEY_TUBULE_META: Record<string, ParamMeta> = {
  // === Tubule Geometry ===
  tubule_outer_diameter: { type: 'number', label: 'Tubule Outer Diameter', min: 200, max: 800, step: 25, unit: 'um', description: 'Scaled for scaffold (native ~50-60 um)' },
  tubule_inner_diameter: { type: 'number', label: 'Tubule Inner Diameter', min: 100, max: 600, step: 25, unit: 'um', description: 'Lumen diameter' },
  tubule_length: { type: 'number', label: 'Tubule Length', min: 3, max: 30, step: 1, unit: 'mm', description: 'Native PCT ~14mm' },

  // === Epithelial Features ===
  epithelial_cell_height: { type: 'number', label: 'Epithelial Cell Height', min: 10, max: 30, step: 2, unit: 'um', description: '15-25 um native', advanced: true },
  microvilli_height: { type: 'number', label: 'Microvilli Height', min: 1, max: 5, step: 0.5, unit: 'um', description: 'Brush border, 1-3 um native', advanced: true },
  enable_brush_border_texture: { type: 'boolean', label: 'Enable Brush Border Texture', description: 'Surface texture for microvilli', advanced: true },
  brush_border_density: { type: 'number', label: 'Brush Border Density', min: 0.5, max: 1, step: 0.1, description: '0-1 coverage', advanced: true },

  // === Basement Membrane ===
  enable_basement_membrane: { type: 'boolean', label: 'Enable Basement Membrane', description: 'Include basement membrane layer' },
  basement_membrane_thickness: { type: 'number', label: 'Basement Membrane Thickness', min: 0.1, max: 1, step: 0.1, unit: 'um', description: '0.3-0.5 um native', advanced: true },

  // === Convolution Parameters ===
  convolution_amplitude: { type: 'number', label: 'Convolution Amplitude', min: 100, max: 500, step: 25, unit: 'um', description: 'Reduced from native for scaffold' },
  convolution_frequency: { type: 'number', label: 'Convolution Frequency', min: 0.5, max: 5, step: 0.5, description: 'Oscillations per mm' },
  convolution_phase_xy: { type: 'number', label: 'Convolution Phase XY', min: 0, max: 6.28, step: 0.5, description: 'Phase offset between XY planes', advanced: true },
  enable_3d_convolution: { type: 'boolean', label: 'Enable 3D Convolution', description: 'Helical vs planar convolution' },

  // === Scaffold Structure ===
  scaffold_porosity: { type: 'number', label: 'Scaffold Porosity', min: 0.5, max: 0.9, step: 0.05, description: 'Target void fraction', advanced: true },
  wall_porosity: { type: 'number', label: 'Wall Porosity', min: 0.1, max: 0.5, step: 0.05, description: 'Porosity within tubule wall', advanced: true },
  pore_size: { type: 'number', label: 'Pore Size', min: 5, max: 20, step: 1, unit: 'um', description: 'For nutrient diffusion', advanced: true },

  // === Peritubular Capillaries ===
  enable_peritubular_capillaries: { type: 'boolean', label: 'Enable Peritubular Capillaries', description: 'Capillary network around tubule', advanced: true },
  capillary_diameter: { type: 'number', label: 'Capillary Diameter', min: 5, max: 15, step: 1, unit: 'um', description: '5-10 um native', advanced: true },
  capillary_spacing: { type: 'number', label: 'Capillary Spacing', min: 25, max: 100, step: 5, unit: 'um', description: 'Spacing between capillaries', advanced: true },
  capillary_wrap_angle: { type: 'number', label: 'Capillary Wrap Angle', min: 90, max: 360, step: 30, unit: 'deg', description: 'Degrees around tubule', advanced: true },

  // === Segment Types ===
  tubule_segment_type: {
    type: 'enum',
    label: 'Tubule Segment Type',
    options: [
      { value: 'proximal', label: 'Proximal' },
      { value: 'distal', label: 'Distal' },
      { value: 'collecting', label: 'Collecting' },
      { value: 'loop_of_henle', label: 'Loop of Henle' },
    ],
    description: 'Type of kidney tubule segment',
    advanced: true,
  },
  enable_segment_transitions: { type: 'boolean', label: 'Enable Segment Transitions', description: 'Smooth transitions between segment types', advanced: true },
  transition_length: { type: 'number', label: 'Transition Length', min: 0.25, max: 1, step: 0.25, unit: 'mm', description: 'Length of transition zones', advanced: true },

  // === Tubule Network ===
  enable_branching: { type: 'boolean', label: 'Enable Branching', description: 'Create branching tubule network', advanced: true },
  branch_count: { type: 'number', label: 'Branch Count', min: 0, max: 5, step: 1, description: 'Number of branches', advanced: true },
  branch_angle: { type: 'number', label: 'Branch Angle', min: 20, max: 60, step: 5, unit: 'deg', description: 'Angle of branches', advanced: true },
  branch_diameter_ratio: { type: 'number', label: 'Branch Diameter Ratio', min: 0.5, max: 0.9, step: 0.1, description: 'Child/parent diameter ratio', advanced: true },

  // === Cell Attachment Features ===
  enable_cell_attachment_sites: { type: 'boolean', label: 'Enable Cell Attachment Sites', description: 'Surface features for cell attachment', advanced: true },
  attachment_site_diameter: { type: 'number', label: 'Attachment Site Diameter', min: 2, max: 10, step: 1, unit: 'um', description: 'Diameter of attachment sites', advanced: true },
  attachment_site_spacing: { type: 'number', label: 'Attachment Site Spacing', min: 10, max: 40, step: 5, unit: 'um', description: 'Spacing between sites', advanced: true },
  attachment_site_depth: { type: 'number', label: 'Attachment Site Depth', min: 1, max: 5, step: 0.5, unit: 'um', description: 'Depth of attachment sites', advanced: true },

  // === Stochastic Variation ===
  diameter_variance: { type: 'number', label: 'Diameter Variance', min: 0, max: 1, step: 0.1, description: 'Random diameter variation (0-1)', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 1, step: 0.1, description: 'Random position jitter (0-1)', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Random seed for reproducibility', advanced: true },

  // === Resolution ===
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 24, step: 2, description: 'Segments around cylinder', advanced: true },
};

export const LUNG_ALVEOLI_META: Record<string, ParamMeta> = {
  // === Alveolar Geometry ===
  alveolar_diameter: { type: 'number', label: 'Alveolar Diameter', min: 150, max: 350, step: 25, unit: 'um', description: '200-300 um typical' },
  alveolar_wall_thickness: { type: 'number', label: 'Alveolar Wall Thickness', min: 0.2, max: 2, step: 0.2, unit: 'um', description: '0.2-0.6 um native, scaled for scaffold', advanced: true },
  alveolar_depth: { type: 'number', label: 'Alveolar Depth', min: 100, max: 250, step: 25, unit: 'um', description: 'Depth of alveolar sac', advanced: true },
  alveoli_per_duct: { type: 'number', label: 'Alveoli per Duct', min: 3, max: 12, step: 1, description: 'Number of alveoli around each alveolar duct', advanced: true },

  // === Airway Tree ===
  total_branching_generations: { type: 'number', label: 'Total Branching Generations', min: 10, max: 23, step: 1, description: 'Native has 23 generations (reference only)', advanced: true },
  scaffold_generations: { type: 'number', label: 'Scaffold Generations', min: 1, max: 5, step: 1, description: 'Generations to actually model (max 5 for performance)' },
  terminal_bronchiole_diameter: { type: 'number', label: 'Terminal Bronchiole Diameter', min: 500, max: 1200, step: 50, unit: 'um', description: '600-1000 um native', advanced: true },
  airway_diameter: { type: 'number', label: 'Airway Diameter', min: 0.5, max: 3, step: 0.1, unit: 'mm', description: 'Starting airway diameter' },
  branch_angle: { type: 'number', label: 'Branch Angle', min: 25, max: 50, step: 5, unit: 'deg', description: '30-45 degrees typical' },
  branch_length_ratio: { type: 'number', label: 'Branch Length Ratio', min: 0.6, max: 0.95, step: 0.05, description: 'Child length / parent length', advanced: true },

  // === Murrays Law Branching ===
  branching_ratio: { type: 'number', label: 'Branching Ratio (Murray)', min: 0.7, max: 0.9, step: 0.01, description: "Murray's law: r_child = r_parent * 0.79", advanced: true },
  enable_asymmetric_branching: { type: 'boolean', label: 'Enable Asymmetric Branching', description: 'Different angles for daughter branches', advanced: true },
  asymmetry_factor: { type: 'number', label: 'Asymmetry Factor', min: 0, max: 0.3, step: 0.05, description: '0-1 difference between branches', advanced: true },

  // === Porosity and Structure ===
  porosity: { type: 'number', label: 'Porosity', min: 0.75, max: 0.95, step: 0.05, description: 'High porosity for gas exchange', advanced: true },
  pore_interconnectivity: { type: 'number', label: 'Pore Interconnectivity', min: 0.8, max: 1, step: 0.05, description: 'Connection between alveoli', advanced: true },
  enable_pores_of_kohn: { type: 'boolean', label: 'Enable Pores of Kohn', description: 'Inter-alveolar pores', advanced: true },
  pore_of_kohn_diameter: { type: 'number', label: 'Pore of Kohn Diameter', min: 6, max: 15, step: 1, unit: 'um', description: '8-12 um native', advanced: true },
  pores_per_alveolus: { type: 'number', label: 'Pores per Alveolus', min: 1, max: 6, step: 1, description: 'Average pores of Kohn per alveolus', advanced: true },

  // === Blood-Air Barrier ===
  enable_blood_air_barrier: { type: 'boolean', label: 'Enable Blood-Air Barrier', description: 'Explicit barrier layer', advanced: true },
  type_1_cell_coverage: { type: 'number', label: 'Type I Cell Coverage', min: 0.9, max: 0.98, step: 0.01, description: 'Type I pneumocyte coverage', advanced: true },
  type_2_cell_coverage: { type: 'number', label: 'Type II Cell Coverage', min: 0.02, max: 0.1, step: 0.01, description: 'Type II pneumocyte (surfactant), 5-7% of alveolar surface area, default: 0.06', advanced: true },
  type_2_cell_diameter: { type: 'number', label: 'Type II Cell Diameter', min: 8, max: 12, step: 0.5, unit: 'um', description: 'Type II pneumocyte diameter (cuboidal cells ~9-10 um)', advanced: true },

  // === Capillary Network ===
  enable_capillary_network: { type: 'boolean', label: 'Enable Capillary Network', description: 'Capillary network for gas exchange', advanced: true },
  capillary_diameter: { type: 'number', label: 'Capillary Diameter', min: 5, max: 12, step: 1, unit: 'um', description: '5-10 um typical', advanced: true },
  capillary_density: { type: 'number', label: 'Capillary Density', min: 2500, max: 3500, step: 100, unit: '/mm2', description: 'Very high in alveoli', advanced: true },
  capillary_spacing: { type: 'number', label: 'Capillary Spacing', min: 10, max: 25, step: 2, unit: 'um', description: 'Distance between capillaries', advanced: true },

  // === Alveolar Duct Features ===
  enable_alveolar_ducts: { type: 'boolean', label: 'Enable Alveolar Ducts', description: 'Transition zone', advanced: true },
  alveolar_duct_length: { type: 'number', label: 'Alveolar Duct Length', min: 0.5, max: 2, step: 0.1, unit: 'mm', description: '1-2 mm native', advanced: true },
  alveolar_duct_diameter: { type: 'number', label: 'Alveolar Duct Diameter', min: 300, max: 600, step: 50, unit: 'um', description: 'Diameter of alveolar duct', advanced: true },

  // === Scaffold Bounding ===
  bounding_box: boundingBoxMeta,

  // === Surfactant Layer ===
  surfactant_layer_thickness: { type: 'number', label: 'Surfactant Layer Thickness', min: 0.1, max: 0.3, step: 0.01, unit: 'um', description: '100-200 nm native hypophase', advanced: true },

  // === Stochastic Variation ===
  size_variance: { type: 'number', label: 'Size Variance', min: 0, max: 1, step: 0.1, description: 'Variation in alveolar size (0-1)', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 1, step: 0.1, description: 'Variation in positions (0-1)', advanced: true },
  angle_noise: { type: 'number', label: 'Angle Noise', min: 0, max: 1, step: 0.1, description: 'Variation in branch angles (0-1)', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Random seed for reproducibility', advanced: true },

  // === Resolution ===
  resolution: { type: 'number', label: 'Resolution', min: 6, max: 16, step: 2, description: 'Sphere/cylinder segments', advanced: true },
};

export const PANCREATIC_ISLET_META: Record<string, ParamMeta> = {
  // === Islet Geometry ===
  islet_diameter: { type: 'number', label: 'Islet Diameter', min: 50, max: 500, step: 25, unit: 'um', description: '50-500 um, average ~150' },
  islet_count: { type: 'number', label: 'Islet Count', min: 1, max: 20, step: 1, description: 'Number of islets in scaffold' },
  islet_spacing: { type: 'number', label: 'Islet Spacing', min: 150, max: 600, step: 50, unit: 'um', description: 'Distance between islet centers' },

  // === Cell Type Fractions ===
  beta_cell_fraction: { type: 'number', label: 'Beta Cell Fraction', min: 0.4, max: 0.8, step: 0.05, description: 'Insulin-producing (50-80% native)', advanced: true },
  alpha_cell_fraction: { type: 'number', label: 'Alpha Cell Fraction', min: 0.15, max: 0.40, step: 0.01, description: 'Glucagon-producing. Human: 35%, Rodent: 15-20%', advanced: true },
  delta_cell_fraction: { type: 'number', label: 'Delta Cell Fraction', min: 0.03, max: 0.15, step: 0.01, description: 'Somatostatin-producing (3-10% native)', advanced: true },
  pp_cell_fraction: { type: 'number', label: 'PP Cell Fraction', min: 0.01, max: 0.05, step: 0.005, description: 'Pancreatic polypeptide (1-2% native)', advanced: true },
  enable_cell_distribution_markers: { type: 'boolean', label: 'Enable Cell Distribution Markers', description: 'Visual markers for cell zones', advanced: true },

  // === Encapsulation Capsule ===
  enable_capsule: { type: 'boolean', label: 'Enable Capsule', description: 'Immunoprotection capsule' },
  capsule_thickness: { type: 'number', label: 'Capsule Thickness', min: 50, max: 200, step: 10, unit: 'um', description: 'Immunoprotection layer', advanced: true },
  capsule_diameter: { type: 'number', label: 'Capsule Diameter', min: 300, max: 1000, step: 50, unit: 'um', description: 'Outer diameter including islets', advanced: true },
  capsule_porosity: { type: 'number', label: 'Capsule Porosity', min: 0.2, max: 0.6, step: 0.05, description: 'Allow nutrients, block immune cells', advanced: true },
  capsule_pore_size: { type: 'number', label: 'Capsule Pore Size', min: 5.0, max: 30.0, step: 1.0, unit: 'um', description: '10-15 μm for immunoisolation', advanced: true },

  // === Islet Shell Structure ===
  shell_thickness: { type: 'number', label: 'Shell Thickness', min: 20, max: 100, step: 10, unit: 'um', description: 'Outer shell of islet' },
  core_porosity: { type: 'number', label: 'Core Porosity', min: 0.4, max: 0.8, step: 0.05, description: 'Inner core porosity', advanced: true },
  shell_porosity: { type: 'number', label: 'Shell Porosity', min: 0.3, max: 0.7, step: 0.05, description: 'Outer shell porosity', advanced: true },
  enable_core_shell_architecture: { type: 'boolean', label: 'Enable Core-Shell Architecture', description: 'Core-shell structure', advanced: true },

  // === Porosity and Diffusion ===
  pore_size: { type: 'number', label: 'Pore Size', min: 10, max: 50, step: 5, unit: 'um', description: 'For nutrient diffusion' },
  pore_density: { type: 'number', label: 'Pore Density', min: 0.05, max: 0.2, step: 0.01, description: 'Pores per surface area', advanced: true },
  oxygen_diffusion_coefficient: { type: 'number', label: 'Oxygen Diffusion Coefficient', min: 1e-9, max: 5e-9, step: 0.5e-9, unit: 'm2/s', description: 'Reference value', advanced: true },
  glucose_diffusion_coefficient: { type: 'number', label: 'Glucose Diffusion Coefficient', min: 3e-10, max: 1e-9, step: 1e-10, unit: 'm2/s', description: 'Reference value', advanced: true },

  // === Vascularization ===
  enable_vascular_channels: { type: 'boolean', label: 'Enable Vascular Channels', description: 'Channels for nutrient delivery', advanced: true },
  vascular_channel_diameter: { type: 'number', label: 'Vascular Channel Diameter', min: 10, max: 40, step: 5, unit: 'um', description: 'Diameter of vascular channels', advanced: true },
  vascular_channel_count: { type: 'number', label: 'Vascular Channel Count', min: 2, max: 8, step: 1, description: 'Channels per islet', advanced: true },
  vascular_channel_pattern: {
    type: 'enum',
    label: 'Vascular Channel Pattern',
    options: [
      { value: 'radial', label: 'Radial' },
      { value: 'parallel', label: 'Parallel' },
    ],
    description: 'Pattern of vascular channels',
    advanced: true,
  },

  // === Viability Support ===
  islet_viability: { type: 'number', label: 'Islet Viability', min: 0.7, max: 1, step: 0.05, description: 'Target viability (fraction of live cells)', advanced: true },
  max_diffusion_distance: { type: 'number', label: 'Max Diffusion Distance', min: 100, max: 200, step: 10, unit: 'um', description: 'Oxygen diffusion limit', advanced: true },

  // === ECM Components ===
  enable_ecm_coating: { type: 'boolean', label: 'Enable ECM Coating', description: 'Collagen/laminin layer', advanced: true },
  ecm_thickness: { type: 'number', label: 'ECM Thickness', min: 2, max: 10, step: 1, unit: 'um', description: 'ECM layer thickness', advanced: true },
  ecm_type: {
    type: 'enum',
    label: 'ECM Type',
    options: [
      { value: 'collagen', label: 'Collagen' },
      { value: 'laminin', label: 'Laminin' },
      { value: 'matrigel', label: 'Matrigel' },
    ],
    description: 'Type of ECM coating',
    advanced: true,
  },

  // === Multi-Islet Configuration ===
  cluster_pattern: {
    type: 'enum',
    label: 'Cluster Pattern',
    options: [
      { value: 'hexagonal', label: 'Hexagonal' },
      { value: 'random', label: 'Random' },
      { value: 'linear', label: 'Linear' },
    ],
    description: 'Arrangement pattern of islets',
    advanced: true,
  },
  enable_inter_islet_connections: { type: 'boolean', label: 'Enable Inter-Islet Connections', description: 'Connections between islets', advanced: true },
  connection_diameter: { type: 'number', label: 'Connection Diameter', min: 15, max: 50, step: 5, unit: 'um', description: 'Diameter of inter-islet connections', advanced: true },

  // === Stochastic Variation ===
  size_variance: { type: 'number', label: 'Size Variance', min: 0, max: 1, step: 0.1, description: 'Variation in islet size (0-1)', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 1, step: 0.1, description: 'Variation in position (0-1)', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Random seed for reproducibility', advanced: true },

  // === Resolution ===
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 20, step: 2, description: 'Sphere segments', advanced: true },
};

export const LIVER_SINUSOID_META: Record<string, ParamMeta> = {
  // === Sinusoid Geometry ===
  sinusoid_diameter: { type: 'number', label: 'Sinusoid Diameter', min: 7, max: 20, step: 1, unit: 'um', description: '7-15 um native' },
  sinusoid_length: { type: 'number', label: 'Sinusoid Length', min: 150, max: 600, step: 25, unit: 'um', description: '200-500 um native' },
  sinusoid_wall_thickness: { type: 'number', label: 'Sinusoid Wall Thickness', min: 0.2, max: 1, step: 0.1, unit: 'um', description: 'Endothelial cell layer', advanced: true },

  // === Fenestration Parameters ===
  fenestration_diameter: { type: 'number', label: 'Fenestration Diameter', min: 50, max: 250, step: 25, unit: 'nm', description: '50-150 nm native, scaled for scaffold' },
  fenestration_density: { type: 'number', label: 'Fenestration Density', min: 5, max: 15, step: 1, unit: '/um2', description: '9-13 per um2 native', advanced: true },
  fenestration_pattern: {
    type: 'enum',
    label: 'Fenestration Pattern',
    options: [
      { value: 'clustered', label: 'Clustered (Sieve Plates)' },
      { value: 'random', label: 'Random' },
    ],
    description: 'Distribution pattern of fenestrations',
    advanced: true,
  },
  sieve_plate_count: { type: 'number', label: 'Sieve Plate Count', min: 4, max: 16, step: 2, description: 'Number of sieve plate regions', advanced: true },
  fenestration_porosity: { type: 'number', label: 'Fenestration Porosity', min: 0.03, max: 0.12, step: 0.01, description: '~6-8% of endothelial surface', advanced: true },

  // === Space of Disse ===
  enable_space_of_disse: { type: 'boolean', label: 'Enable Space of Disse', description: 'Perisinusoidal space' },
  space_of_disse_thickness: { type: 'number', label: 'Space of Disse Thickness', min: 0.2, max: 1, step: 0.1, unit: 'um', description: '0.2-0.5 um native', advanced: true },
  space_of_disse_porosity: { type: 'number', label: 'Space of Disse Porosity', min: 0.6, max: 0.95, step: 0.05, description: 'High porosity for protein exchange', advanced: true },

  // === Hepatocyte Zone ===
  enable_hepatocyte_zone: { type: 'boolean', label: 'Enable Hepatocyte Zone', description: 'Surrounding hepatocyte markers', advanced: true },
  hepatocyte_diameter: { type: 'number', label: 'Hepatocyte Diameter', min: 20, max: 35, step: 2, unit: 'um', description: '20-30 um native', advanced: true },
  hepatocyte_spacing: { type: 'number', label: 'Hepatocyte Spacing', min: 3, max: 10, step: 1, unit: 'um', description: 'Gap between hepatocytes', advanced: true },
  hepatocytes_per_sinusoid: { type: 'number', label: 'Hepatocytes per Sinusoid', min: 6, max: 20, step: 2, description: 'Around circumference', advanced: true },

  // === Cell Type Zones ===
  enable_kupffer_cell_zones: { type: 'boolean', label: 'Enable Kupffer Cell Zones', description: 'Macrophage attachment sites', advanced: true },
  kupffer_cell_density: { type: 'number', label: 'Kupffer Cell Density', min: 0.1, max: 0.25, step: 0.05, description: '~15% of sinusoid wall', advanced: true },
  enable_stellate_cell_zones: { type: 'boolean', label: 'Enable Stellate Cell Zones', description: 'HSC in space of Disse', advanced: true },
  stellate_cell_spacing: { type: 'number', label: 'Stellate Cell Spacing', min: 30, max: 80, step: 10, unit: 'um', description: 'Spacing of stellate cells', advanced: true },

  // === Sinusoid Network ===
  sinusoid_count: { type: 'number', label: 'Sinusoid Count', min: 1, max: 10, step: 1, description: 'Number of parallel sinusoids' },
  sinusoid_spacing: { type: 'number', label: 'Sinusoid Spacing', min: 30, max: 100, step: 10, unit: 'um', description: 'Distance between sinusoid centers', advanced: true },
  network_pattern: {
    type: 'enum',
    label: 'Network Pattern',
    options: [
      { value: 'parallel', label: 'Parallel' },
      { value: 'branching', label: 'Branching' },
      { value: 'anastomosing', label: 'Anastomosing' },
    ],
    description: 'Sinusoid network configuration',
    advanced: true,
  },
  enable_central_vein_connection: { type: 'boolean', label: 'Enable Central Vein Connection', description: 'Connect to central vein', advanced: true },
  central_vein_diameter: { type: 'number', label: 'Central Vein Diameter', min: 20, max: 50, step: 5, unit: 'um', description: 'Central vein diameter', advanced: true },

  // === Scaffold Structure ===
  scaffold_length: { type: 'number', label: 'Scaffold Length', min: 1, max: 5, step: 0.25, unit: 'mm', description: 'Total scaffold length', advanced: true },
  enable_scaffold_shell: { type: 'boolean', label: 'Enable Scaffold Shell', description: 'Outer structural shell', advanced: true },
  shell_thickness: { type: 'number', label: 'Shell Thickness', min: 25, max: 100, step: 5, unit: 'um', description: 'Outer shell thickness', advanced: true },
  shell_porosity: { type: 'number', label: 'Shell Porosity', min: 0.3, max: 0.7, step: 0.05, description: 'Shell porosity', advanced: true },

  // === Bile Canaliculi ===
  enable_bile_canaliculi: { type: 'boolean', label: 'Enable Bile Canaliculi', description: 'Add bile canaliculus channels', advanced: true },
  canaliculus_diameter: { type: 'number', label: 'Canaliculus Diameter', min: 0.5, max: 3, step: 0.25, unit: 'um', description: '0.5-2.0 um native', advanced: true },
  canaliculus_spacing: { type: 'number', label: 'Canaliculus Spacing', min: 15, max: 40, step: 5, unit: 'um', description: 'Spacing between canaliculi', advanced: true },

  // === ECM Components ===
  enable_ecm_fibers: { type: 'boolean', label: 'Enable ECM Fibers', description: 'Collagen fibers in space of Disse', advanced: true },
  ecm_fiber_diameter: { type: 'number', label: 'ECM Fiber Diameter', min: 0.05, max: 0.2, step: 0.01, unit: 'um', description: '100-150 nm native reticular fibers', advanced: true },
  ecm_fiber_density: { type: 'number', label: 'ECM Fiber Density', min: 0.1, max: 0.4, step: 0.05, description: 'Fraction of space of Disse', advanced: true },

  // === Stochastic Variation ===
  diameter_variance: { type: 'number', label: 'Diameter Variance', min: 0, max: 1, step: 0.1, description: 'Variation in sinusoid diameter (0-1)', advanced: true },
  fenestration_variance: { type: 'number', label: 'Fenestration Variance', min: 0, max: 1, step: 0.1, description: 'Variation in fenestration size (0-1)', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 1, step: 0.1, description: 'Position jitter (0-1)', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Random seed for reproducibility', advanced: true },

  // === Resolution ===
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 20, step: 2, description: 'Segments around cylinder', advanced: true },
};

// Combined export for organ scaffold types
export const ORGAN_PARAMETER_META: Partial<Record<ScaffoldType, Record<string, ParamMeta>>> = {
  [ScaffoldType.HEPATIC_LOBULE]: HEPATIC_LOBULE_META,
  [ScaffoldType.CARDIAC_PATCH]: CARDIAC_PATCH_META,
  [ScaffoldType.KIDNEY_TUBULE]: KIDNEY_TUBULE_META,
  [ScaffoldType.LUNG_ALVEOLI]: LUNG_ALVEOLI_META,
  [ScaffoldType.PANCREATIC_ISLET]: PANCREATIC_ISLET_META,
  [ScaffoldType.LIVER_SINUSOID]: LIVER_SINUSOID_META,
};
