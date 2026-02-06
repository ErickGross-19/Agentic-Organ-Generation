/**
 * Parameter metadata for skeletal scaffold types
 * Includes: trabecular_bone, osteochondral, articular_cartilage, meniscus, tendon_ligament, intervertebral_disc, haversian_bone
 */

import { ScaffoldType } from '../types/scaffolds';
import { ParamMeta, boundingBoxMeta } from './types';

export const TRABECULAR_BONE_META: Record<string, ParamMeta> = {
  // Basic geometry
  bounding_box: boundingBoxMeta,
  resolution: { type: 'number', label: 'Resolution', min: 4, max: 16, step: 1, description: 'Mesh resolution for curved surfaces', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducible randomization' },

  // Pore characteristics
  pore_size_um: { type: 'number', label: 'Mean Pore Size', min: 100, max: 800, step: 25, unit: 'um', description: 'Mean pore diameter (400 um typical)' },
  pore_size_min_um: { type: 'number', label: 'Minimum Pore Size', min: 50, max: 400, step: 25, unit: 'um', description: 'Minimum pore diameter', advanced: true },
  pore_size_max_um: { type: 'number', label: 'Maximum Pore Size', min: 200, max: 1000, step: 50, unit: 'um', description: 'Maximum pore diameter', advanced: true },
  pore_size_variance: { type: 'number', label: 'Pore Size Variance', min: 0, max: 0.5, step: 0.05, description: 'Random variation in pore size (0-1)', advanced: true },

  // Trabecular architecture
  trabecular_thickness_um: { type: 'number', label: 'Trabecular Thickness', min: 100, max: 400, step: 25, unit: 'um', description: 'Mean trabecular thickness (100-300 um typical)' },
  trabecular_spacing_um: { type: 'number', label: 'Trabecular Spacing', min: 200, max: 1200, step: 50, unit: 'um', description: 'Mean trabecular separation (300-1000 um typical)', advanced: true },
  strut_thickness_um: { type: 'number', label: 'Strut Thickness', min: 100, max: 400, step: 25, unit: 'um', description: 'Legacy alias for trabecular_thickness', advanced: true },
  strut_diameter_um: { type: 'number', label: 'Strut Diameter', min: 100, max: 500, step: 25, unit: 'um', description: 'Alternative strut specification', advanced: true },

  // Porosity and volume metrics
  porosity: { type: 'number', label: 'Porosity', min: 0.4, max: 0.95, step: 0.05, description: 'Target porosity/void fraction (50-90% typical)' },
  bone_volume_fraction: { type: 'number', label: 'Bone Volume Fraction', min: 0.05, max: 0.6, step: 0.05, description: 'BV/TV ratio (1 - porosity)', advanced: true },
  connectivity_density: { type: 'number', label: 'Connectivity Density', min: 1, max: 20, step: 1, description: 'Connections per mm3 (1-15 typical)', advanced: true },

  // Structural anisotropy
  anisotropy_ratio: { type: 'number', label: 'Anisotropy Ratio', min: 1.0, max: 3.0, step: 0.1, description: 'Primary direction stretch (1.0 = isotropic)' },
  degree_anisotropy: { type: 'number', label: 'Degree of Anisotropy', min: 1.0, max: 3.0, step: 0.1, description: 'MIL-based anisotropy (1.0 = isotropic)', advanced: true },
  primary_orientation_deg: { type: 'number', label: 'Primary Orientation', min: 0, max: 180, step: 15, unit: 'deg', description: 'Primary trabecular alignment direction', advanced: true },
  fabric_tensor_eigenratio: { type: 'number', label: 'Fabric Tensor Eigenratio', min: 1.0, max: 3.0, step: 0.1, description: 'Ratio of principal fabric eigenvalues', advanced: true },

  // Randomization and organic appearance
  randomization_factor: { type: 'number', label: 'Randomization Factor', min: 0, max: 0.5, step: 0.05, description: 'Overall structural randomness (0-1)', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.5, step: 0.05, description: 'Node position jitter relative to spacing', advanced: true },
  thickness_variance: { type: 'number', label: 'Thickness Variance', min: 0, max: 0.3, step: 0.05, description: 'Variation in trabecular thickness', advanced: true },
  connectivity_variance: { type: 'number', label: 'Connectivity Variance', min: 0, max: 0.4, step: 0.05, description: 'Variation in local connectivity', advanced: true },

  // Rod vs plate morphology
  structure_model_index: { type: 'number', label: 'Structure Model Index', min: 0, max: 3, step: 0.25, description: '0 = plates, 3 = rods, ~1.5 typical', advanced: true },
  enable_plate_like_regions: { type: 'boolean', label: 'Enable Plate-like Regions', description: 'Include some plate-like structures', advanced: true },
  plate_fraction: { type: 'number', label: 'Plate Fraction', min: 0, max: 1, step: 0.1, description: 'Fraction of plate-like vs rod-like structures', advanced: true },

  // Surface characteristics
  surface_roughness: { type: 'number', label: 'Surface Roughness', min: 0, max: 0.3, step: 0.05, description: 'Surface roughness factor', advanced: true },
  enable_resorption_pits: { type: 'boolean', label: 'Enable Resorption Pits', description: 'Simulate osteoclast resorption lacunae', advanced: true },
  resorption_pit_density: { type: 'number', label: 'Resorption Pit Density', min: 0, max: 0.2, step: 0.01, description: 'Pits per mm2 of surface', advanced: true },

  // Gradient options
  enable_density_gradient: { type: 'boolean', label: 'Enable Density Gradient', description: 'Variable density across scaffold', advanced: true },
  gradient_direction: { type: 'enum', label: 'Gradient Direction', options: [{ value: 'x', label: 'X' }, { value: 'y', label: 'Y' }, { value: 'z', label: 'Z' }], description: 'Axis for gradient', advanced: true },
  gradient_start_porosity: { type: 'number', label: 'Gradient Start Porosity', min: 0.4, max: 0.95, step: 0.05, description: 'Porosity at gradient start', advanced: true },
  gradient_end_porosity: { type: 'number', label: 'Gradient End Porosity', min: 0.4, max: 0.95, step: 0.05, description: 'Porosity at gradient end', advanced: true },
};

export const HAVERSIAN_BONE_META: Record<string, ParamMeta> = {
  // Basic geometry
  bounding_box: boundingBoxMeta,
  resolution: { type: 'number', label: 'Resolution', min: 4, max: 24, step: 2, description: 'Mesh resolution for curved surfaces', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducible randomization' },

  // Haversian canal (central vascular channel)
  haversian_canal_diameter_um: { type: 'number', label: 'Haversian Canal Diameter', min: 30, max: 120, step: 5, unit: 'um', description: 'Central canal diameter (40-90 um typical)' },
  canal_diameter_um: { type: 'number', label: 'Canal Diameter', min: 30, max: 120, step: 5, unit: 'um', description: 'Legacy alias for Haversian canal', advanced: true },
  canal_wall_thickness_um: { type: 'number', label: 'Canal Wall Thickness', min: 1, max: 20, step: 1, unit: 'um', description: 'Canal lining thickness', advanced: true },
  enable_haversian_vessels: { type: 'boolean', label: 'Enable Haversian Vessels', description: 'Include vessel structures', advanced: true },

  // Osteon (Haversian system) properties
  osteon_diameter_um: { type: 'number', label: 'Osteon Diameter', min: 100, max: 400, step: 25, unit: 'um', description: 'Full osteon diameter (150-350 um typical)' },
  canal_spacing_um: { type: 'number', label: 'Canal Spacing', min: 200, max: 600, step: 25, unit: 'um', description: 'Spacing between canals' },
  osteon_spacing_um: { type: 'number', label: 'Osteon Spacing', min: 150, max: 500, step: 25, unit: 'um', description: 'Center-to-center osteon spacing', advanced: true },
  osteon_pattern: { type: 'enum', label: 'Osteon Pattern', options: [{ value: 'hexagonal', label: 'Hexagonal' }, { value: 'random', label: 'Random' }, { value: 'organized', label: 'Organized' }], description: 'Spatial arrangement of osteons' },

  // Volkmann canals (transverse/oblique connecting canals)
  enable_volkmann_canals: { type: 'boolean', label: 'Enable Volkmann Canals', description: 'Include transverse canals' },
  volkmann_canal_diameter_um: { type: 'number', label: 'Volkmann Canal Diameter', min: 20, max: 150, step: 5, unit: 'um', description: 'Transverse canal diameter (20-100 um typical)', advanced: true },
  volkmann_canal_density: { type: 'number', label: 'Volkmann Canal Density', min: 0, max: 1, step: 0.1, description: 'Density of Volkmann canals (0-1)', advanced: true },
  volkmann_angle_deg: { type: 'number', label: 'Volkmann Angle', min: 30, max: 90, step: 10, unit: 'deg', description: 'Angle from longitudinal axis', advanced: true },

  // Concentric lamellae
  num_concentric_lamellae: { type: 'number', label: 'Lamellae Count', min: 2, max: 25, step: 1, description: 'Lamellae per osteon (4-20 typical)', advanced: true },
  lamella_thickness_um: { type: 'number', label: 'Lamella Thickness', min: 2, max: 10, step: 0.5, unit: 'um', description: 'Individual lamella thickness (3-7 um)', advanced: true },
  interlamellar_spacing_um: { type: 'number', label: 'Interlamellar Spacing', min: 0.5, max: 5, step: 0.5, unit: 'um', description: 'Gap between lamellae', advanced: true },
  enable_lamellar_detail: { type: 'boolean', label: 'Enable Lamellar Detail', description: 'Show lamellar structure', advanced: true },

  // Lacunae (osteocyte spaces)
  enable_lacunae: { type: 'boolean', label: 'Enable Lacunae', description: 'Osteocyte lacunae', advanced: true },
  lacuna_density: { type: 'number', label: 'Lacuna Density', min: 5000, max: 30000, step: 1000, description: 'Lacunae per mm3 (10000-25000 typical)', advanced: true },
  lacuna_length_um: { type: 'number', label: 'Lacuna Length', min: 10, max: 30, step: 1, unit: 'um', description: 'Lacuna long axis (15-25 um)', advanced: true },
  lacuna_width_um: { type: 'number', label: 'Lacuna Width', min: 3, max: 15, step: 1, unit: 'um', description: 'Lacuna short axis (5-10 um)', advanced: true },

  // Canaliculi
  enable_canaliculi: { type: 'boolean', label: 'Enable Canaliculi', description: 'Fine connecting channels (high resolution)', advanced: true },
  canaliculus_diameter_um: { type: 'number', label: 'Canaliculus Diameter', min: 0.2, max: 1.5, step: 0.1, unit: 'um', description: 'Very small channels', advanced: true },
  canaliculi_per_lacuna: { type: 'number', label: 'Canaliculi per Lacuna', min: 20, max: 150, step: 10, description: 'Channels per lacuna (40-100 typical)', advanced: true },

  // Cement line
  enable_cement_line: { type: 'boolean', label: 'Enable Cement Line', description: 'Boundary between osteons', advanced: true },
  cement_line_thickness_um: { type: 'number', label: 'Cement Line Thickness', min: 1, max: 10, step: 0.5, unit: 'um', description: 'Cement line thickness (1-5 um)', advanced: true },

  // Interstitial lamellae
  enable_interstitial_lamellae: { type: 'boolean', label: 'Enable Interstitial Lamellae', description: 'Old bone fragments', advanced: true },
  interstitial_fraction: { type: 'number', label: 'Interstitial Fraction', min: 0, max: 0.5, step: 0.05, description: 'Fraction of area as interstitial', advanced: true },

  // Cortical bone properties
  cortical_thickness: { type: 'number', label: 'Cortical Thickness', min: 1, max: 15, step: 0.5, unit: 'mm', description: 'Overall cortical thickness' },
  cortical_porosity: { type: 'number', label: 'Cortical Porosity', min: 0.02, max: 0.2, step: 0.01, description: 'Total porosity (5-15%)', advanced: true },
  bone_mineral_density: { type: 'number', label: 'Bone Mineral Density', min: 1.0, max: 2.5, step: 0.1, description: 'g/cm3 indicator', advanced: true },

  // Periosteal and endosteal surfaces
  enable_periosteal_surface: { type: 'boolean', label: 'Enable Periosteal Surface', description: 'Outer bone surface', advanced: true },
  enable_endosteal_surface: { type: 'boolean', label: 'Enable Endosteal Surface', description: 'Inner bone surface', advanced: true },
  periosteal_layer_thickness_um: { type: 'number', label: 'Periosteal Layer Thickness', min: 20, max: 100, step: 10, unit: 'um', description: 'Periosteum-adjacent layer', advanced: true },
  endosteal_layer_thickness_um: { type: 'number', label: 'Endosteal Layer Thickness', min: 10, max: 80, step: 5, unit: 'um', description: 'Endosteum-adjacent layer', advanced: true },

  // Remodeling features
  enable_resorption_spaces: { type: 'boolean', label: 'Enable Resorption Spaces', description: 'Active remodeling cavities', advanced: true },
  resorption_space_diameter_um: { type: 'number', label: 'Resorption Space Diameter', min: 100, max: 400, step: 25, unit: 'um', description: 'Cutting cone diameter', advanced: true },
  resorption_density: { type: 'number', label: 'Resorption Density', min: 0, max: 0.1, step: 0.01, description: 'Fraction of bone under remodeling', advanced: true },

  // Mechanical orientation
  primary_orientation_deg: { type: 'number', label: 'Primary Orientation', min: 0, max: 180, step: 15, unit: 'deg', description: 'Main osteon alignment', advanced: true },
  orientation_variance_deg: { type: 'number', label: 'Orientation Variance', min: 0, max: 30, step: 5, unit: 'deg', description: 'Random variance in alignment', advanced: true },

  // Surface and randomization
  surface_roughness: { type: 'number', label: 'Surface Roughness', min: 0, max: 0.2, step: 0.01, description: 'Surface texture', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.3, step: 0.05, description: 'Random jitter in osteon positions', advanced: true },
  diameter_variance: { type: 'number', label: 'Diameter Variance', min: 0, max: 0.3, step: 0.05, description: 'Variation in osteon/canal diameters', advanced: true },
  lamella_count_variance: { type: 'number', label: 'Lamella Count Variance', min: 0, max: 5, step: 1, description: '+/- variation in lamella count', advanced: true },
};

export const OSTEOCHONDRAL_META: Record<string, ParamMeta> = {
  // Basic geometry
  diameter: { type: 'number', label: 'Diameter', min: 3, max: 25, step: 0.5, unit: 'mm', description: 'Scaffold diameter' },
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 32, step: 2, description: 'Mesh resolution', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducible randomization', advanced: true },

  // Zone depths
  cartilage_depth: { type: 'number', label: 'Cartilage Depth', min: 1, max: 6, step: 0.25, unit: 'mm', description: 'Total articular cartilage thickness (2-4 mm typical)' },
  calcified_cartilage_depth: { type: 'number', label: 'Calcified Cartilage Depth', min: 0.05, max: 0.5, step: 0.05, unit: 'mm', description: 'Calcified cartilage zone thickness (0.1-0.3 mm)', advanced: true },
  subchondral_plate_depth: { type: 'number', label: 'Subchondral Plate Depth', min: 0.1, max: 1.0, step: 0.1, unit: 'mm', description: 'Subchondral bone plate thickness (0.2-0.5 mm)', advanced: true },
  bone_depth: { type: 'number', label: 'Bone Depth', min: 1, max: 10, step: 0.5, unit: 'mm', description: 'Trabecular bone depth' },
  transition_width: { type: 'number', label: 'Transition Width', min: 0.2, max: 3, step: 0.2, unit: 'mm', description: 'Gradient transition zone width' },

  // Cartilage zone ratios
  superficial_zone_ratio: { type: 'number', label: 'Superficial Zone Ratio', min: 0.05, max: 0.3, step: 0.01, description: '10-20% of cartilage', advanced: true },
  middle_zone_ratio: { type: 'number', label: 'Middle Zone Ratio', min: 0.3, max: 0.7, step: 0.05, description: '40-60% of cartilage', advanced: true },
  deep_zone_ratio: { type: 'number', label: 'Deep Zone Ratio', min: 0.2, max: 0.5, step: 0.05, description: '30-40% of cartilage', advanced: true },

  // Porosity per zone
  cartilage_porosity: { type: 'number', label: 'Cartilage Porosity', min: 0.7, max: 0.95, step: 0.05, description: 'Uncalcified cartilage porosity' },
  calcified_cartilage_porosity: { type: 'number', label: 'Calcified Cartilage Porosity', min: 0.4, max: 0.8, step: 0.05, description: 'Lower porosity in calcified zone', advanced: true },
  subchondral_plate_porosity: { type: 'number', label: 'Subchondral Plate Porosity', min: 0.05, max: 0.3, step: 0.05, description: 'Dense cortical-like bone', advanced: true },
  bone_porosity: { type: 'number', label: 'Bone Porosity', min: 0.5, max: 0.9, step: 0.05, description: 'Trabecular bone porosity' },

  // Pore size gradients
  superficial_pore_size: { type: 'number', label: 'Superficial Pore Size', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Small pores in superficial zone', advanced: true },
  middle_pore_size: { type: 'number', label: 'Middle Pore Size', min: 0.1, max: 0.5, step: 0.025, unit: 'mm', description: 'Medium pores in middle zone', advanced: true },
  deep_pore_size: { type: 'number', label: 'Deep Pore Size', min: 0.2, max: 0.6, step: 0.025, unit: 'mm', description: 'Larger pores in deep zone', advanced: true },
  bone_pore_size: { type: 'number', label: 'Bone Pore Size', min: 0.3, max: 0.8, step: 0.05, unit: 'mm', description: 'Trabecular bone pore size', advanced: true },

  // Gradient and transition properties
  gradient_type: { type: 'enum', label: 'Gradient Type', options: [{ value: 'linear', label: 'Linear' }, { value: 'exponential', label: 'Exponential' }, { value: 'sigmoid', label: 'Sigmoid' }], description: 'Porosity transition profile' },
  gradient_sharpness: { type: 'number', label: 'Gradient Sharpness', min: 0.5, max: 3.0, step: 0.25, description: 'Controls transition steepness (1.0 = normal)', advanced: true },

  // Tidemark and cement line features
  enable_tidemark: { type: 'boolean', label: 'Enable Tidemark', description: 'Interface between calcified/uncalcified cartilage', advanced: true },
  tidemark_thickness: { type: 'number', label: 'Tidemark Thickness', min: 0.005, max: 0.05, step: 0.005, unit: 'mm', description: 'Tidemark thickness (~10 um)', advanced: true },
  enable_cement_line: { type: 'boolean', label: 'Enable Cement Line', description: 'Interface between calcified cartilage/bone', advanced: true },
  cement_line_thickness: { type: 'number', label: 'Cement Line Thickness', min: 0.001, max: 0.02, step: 0.001, unit: 'mm', description: 'Cement line (~5 um)', advanced: true },

  // Collagen fiber orientation
  superficial_fiber_angle_deg: { type: 'number', label: 'Superficial Fiber Angle', min: -15, max: 15, step: 5, unit: 'deg', description: 'Tangential (parallel to surface)', advanced: true },
  deep_fiber_angle_deg: { type: 'number', label: 'Deep Fiber Angle', min: 75, max: 105, step: 5, unit: 'deg', description: 'Perpendicular to surface', advanced: true },

  // Channel features
  enable_vascular_channels: { type: 'boolean', label: 'Enable Vascular Channels', description: 'Vascular channels in bone zone' },
  vascular_channel_diameter: { type: 'number', label: 'Vascular Channel Diameter', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Blood vessel channel diameter', advanced: true },
  vascular_channel_spacing: { type: 'number', label: 'Vascular Channel Spacing', min: 0.3, max: 2.0, step: 0.1, unit: 'mm', description: 'Spacing between vascular channels', advanced: true },

  // Subchondral bone properties
  subchondral_trabecular_thickness: { type: 'number', label: 'Subchondral Trabecular Thickness', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Thinner trabeculae near surface', advanced: true },
  subchondral_connectivity: { type: 'number', label: 'Subchondral Connectivity', min: 0.4, max: 1.0, step: 0.1, description: 'Higher connectivity near plate', advanced: true },

  // Mechanical property indicators
  enable_stiffness_gradient: { type: 'boolean', label: 'Enable Stiffness Gradient', description: 'Visual indicator of stiffness zones', advanced: true },
  cartilage_modulus_indicator: { type: 'number', label: 'Cartilage Modulus Indicator', min: 0.1, max: 5.0, step: 0.1, description: 'Relative cartilage stiffness', advanced: true },
  bone_modulus_indicator: { type: 'number', label: 'Bone Modulus Indicator', min: 5, max: 50, step: 5, description: 'Relative bone stiffness', advanced: true },

  // Randomization
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.3, step: 0.05, description: 'Random variation in pore positions', advanced: true },
  pore_size_variance: { type: 'number', label: 'Pore Size Variance', min: 0, max: 0.3, step: 0.05, description: 'Random variation in pore sizes', advanced: true },
};

export const ARTICULAR_CARTILAGE_META: Record<string, ParamMeta> = {
  // Basic geometry
  diameter: { type: 'number', label: 'Diameter', min: 3, max: 25, step: 0.5, unit: 'mm', description: 'Scaffold diameter' },
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 32, step: 2, description: 'Mesh resolution', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducible randomization', advanced: true },

  // Overall thickness
  total_thickness: { type: 'number', label: 'Total Thickness', min: 0.5, max: 6, step: 0.25, unit: 'mm', description: 'Total thickness (2-3 mm typical for knee)' },
  total_thickness_um: { type: 'number', label: 'Total Thickness (um)', min: 500, max: 6000, step: 100, unit: 'um', description: 'Total thickness in micrometers', advanced: true },

  // Zone thickness ratios
  superficial_thickness_ratio: { type: 'number', label: 'Superficial Zone Ratio', min: 0.05, max: 0.3, step: 0.01, description: '10-20% of total' },
  middle_thickness_ratio: { type: 'number', label: 'Middle Zone Ratio', min: 0.3, max: 0.7, step: 0.05, description: '40-60% of total' },
  deep_thickness_ratio: { type: 'number', label: 'Deep Zone Ratio', min: 0.2, max: 0.5, step: 0.05, description: '30-40% of total' },
  zone_ratios: { type: 'array', label: 'Zone Ratios', itemCount: 3, itemLabels: ['Superficial', 'Middle', 'Deep'], min: 0.05, max: 0.8, step: 0.05, description: 'Legacy: must sum to 1.0', advanced: true },

  // Collagen fiber orientation per zone
  superficial_fiber_orientation_deg: { type: 'number', label: 'Superficial Fiber Orientation', min: -15, max: 15, step: 5, unit: 'deg', description: 'Tangential (parallel to surface)', advanced: true },
  middle_fiber_orientation_deg: { type: 'number', label: 'Middle Fiber Orientation', min: 30, max: 60, step: 5, unit: 'deg', description: 'Oblique orientation', advanced: true },
  deep_fiber_orientation_deg: { type: 'number', label: 'Deep Fiber Orientation', min: 75, max: 105, step: 5, unit: 'deg', description: 'Perpendicular to surface', advanced: true },
  fiber_orientation_variance_deg: { type: 'number', label: 'Fiber Orientation Variance', min: 0, max: 30, step: 5, unit: 'deg', description: 'Random variance in orientation', advanced: true },

  // Pore characteristics per zone
  superficial_pore_size: { type: 'number', label: 'Superficial Pore Size', min: 0.05, max: 0.2, step: 0.01, unit: 'mm', description: 'Smaller pores for dense superficial zone' },
  middle_pore_size: { type: 'number', label: 'Middle Pore Size', min: 0.1, max: 0.4, step: 0.025, unit: 'mm', description: 'Medium pores' },
  deep_pore_size: { type: 'number', label: 'Deep Pore Size', min: 0.15, max: 0.5, step: 0.025, unit: 'mm', description: 'Larger pores for columnar structure' },
  pore_gradient: { type: 'array', label: 'Pore Gradient (mm)', itemCount: 3, itemLabels: ['Superficial', 'Middle', 'Deep'], min: 0.05, max: 0.5, step: 0.05, description: 'Legacy pore size array', advanced: true },

  // Porosity per zone
  superficial_porosity: { type: 'number', label: 'Superficial Porosity', min: 0.5, max: 0.85, step: 0.05, description: 'Lower porosity in superficial zone', advanced: true },
  middle_porosity: { type: 'number', label: 'Middle Porosity', min: 0.6, max: 0.9, step: 0.05, description: 'Moderate porosity', advanced: true },
  deep_porosity: { type: 'number', label: 'Deep Porosity', min: 0.7, max: 0.95, step: 0.05, description: 'Higher porosity for nutrient diffusion', advanced: true },

  // Cell density per zone
  superficial_cell_density: { type: 'number', label: 'Superficial Cell Density', min: 5000, max: 20000, step: 1000, description: 'Cells per mm3 (highest density)', advanced: true },
  middle_cell_density: { type: 'number', label: 'Middle Cell Density', min: 2000, max: 10000, step: 500, description: 'Cells per mm3 (moderate)', advanced: true },
  deep_cell_density: { type: 'number', label: 'Deep Cell Density', min: 1000, max: 8000, step: 500, description: 'Cells per mm3 (lowest density)', advanced: true },

  // Proteoglycan content indicators
  superficial_proteoglycan: { type: 'number', label: 'Superficial Proteoglycan', min: 0.1, max: 0.5, step: 0.05, description: 'Lowest content (relative units)', advanced: true },
  middle_proteoglycan: { type: 'number', label: 'Middle Proteoglycan', min: 0.4, max: 0.9, step: 0.05, description: 'Moderate content', advanced: true },
  deep_proteoglycan: { type: 'number', label: 'Deep Proteoglycan', min: 0.7, max: 1.0, step: 0.05, description: 'Highest content', advanced: true },

  // Collagen fiber properties
  collagen_fiber_diameter_um: { type: 'number', label: 'Collagen Fiber Diameter', min: 20, max: 100, step: 5, unit: 'um', description: 'Fiber bundle diameter', advanced: true },
  collagen_fiber_spacing_um: { type: 'number', label: 'Collagen Fiber Spacing', min: 50, max: 200, step: 10, unit: 'um', description: 'Spacing between fibers', advanced: true },
  enable_fiber_bundles: { type: 'boolean', label: 'Enable Fiber Bundles', description: 'Show fiber bundle structures', advanced: true },

  // Channel features for nutrient diffusion
  enable_vertical_channels: { type: 'boolean', label: 'Enable Vertical Channels', description: 'Channels in deep zone' },
  vertical_channel_diameter: { type: 'number', label: 'Vertical Channel Diameter', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Channel diameter', advanced: true },
  vertical_channel_spacing: { type: 'number', label: 'Vertical Channel Spacing', min: 0.2, max: 1.0, step: 0.1, unit: 'mm', description: 'Spacing between channels', advanced: true },
  enable_horizontal_channels: { type: 'boolean', label: 'Enable Horizontal Channels', description: 'Tangential channels in superficial zone', advanced: true },

  // Surface characteristics
  surface_roughness: { type: 'number', label: 'Surface Roughness', min: 0, max: 0.15, step: 0.01, description: 'Articular surface roughness', advanced: true },
  enable_surface_texture: { type: 'boolean', label: 'Enable Surface Texture', description: 'Textured articular surface', advanced: true },

  // Tidemark (calcified cartilage interface)
  enable_tidemark_layer: { type: 'boolean', label: 'Enable Tidemark Layer', description: 'Include calcified cartilage zone', advanced: true },
  tidemark_thickness: { type: 'number', label: 'Tidemark Thickness', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Calcified zone thickness', advanced: true },
  tidemark_porosity: { type: 'number', label: 'Tidemark Porosity', min: 0.4, max: 0.75, step: 0.05, description: 'Lower porosity for calcified zone', advanced: true },

  // Randomization and organic appearance
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.3, step: 0.05, description: 'Random jitter in pore positions', advanced: true },
  pore_size_variance: { type: 'number', label: 'Pore Size Variance', min: 0, max: 0.4, step: 0.05, description: 'Random variation in pore sizes', advanced: true },
  zone_boundary_blur: { type: 'number', label: 'Zone Boundary Blur', min: 0, max: 0.3, step: 0.05, description: 'Soft boundaries between zones', advanced: true },
};

export const MENISCUS_META: Record<string, ParamMeta> = {
  // Basic geometry
  outer_radius: { type: 'number', label: 'Outer Radius', min: 15, max: 30, step: 0.5, unit: 'mm', description: 'Outer edge radius (45mm diameter / 2 typical)' },
  inner_radius: { type: 'number', label: 'Inner Radius', min: 5, max: 20, step: 0.5, unit: 'mm', description: 'Inner edge radius (20mm diameter / 2 typical)' },
  outer_diameter: { type: 'number', label: 'Outer Diameter', min: 30, max: 60, step: 1, unit: 'mm', description: 'Alternative specification', advanced: true },
  inner_diameter: { type: 'number', label: 'Inner Diameter', min: 10, max: 35, step: 1, unit: 'mm', description: 'Alternative specification', advanced: true },
  thickness: { type: 'number', label: 'Thickness', min: 3, max: 12, step: 0.5, unit: 'mm', description: 'Maximum thickness at periphery' },
  height: { type: 'number', label: 'Height', min: 3, max: 12, step: 0.5, unit: 'mm', description: 'Legacy alias for thickness', advanced: true },
  arc_degrees: { type: 'number', label: 'Arc Extent', min: 180, max: 360, step: 15, unit: 'deg', description: 'C-shape arc extent (300 typical)' },
  resolution: { type: 'number', label: 'Resolution', min: 16, max: 64, step: 4, description: 'Mesh resolution', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducible randomization', advanced: true },

  // Wedge geometry
  wedge_angle_deg: { type: 'number', label: 'Wedge Angle', min: 10, max: 45, step: 5, unit: 'deg', description: 'Wedge slope angle' },
  inner_edge_height_ratio: { type: 'number', label: 'Inner Edge Height Ratio', min: 0.1, max: 0.5, step: 0.05, description: 'Inner edge height as fraction of max height', advanced: true },

  // Radial zones
  zone_count: { type: 'number', label: 'Zone Count', min: 2, max: 5, step: 1, description: 'Number of radial zones' },
  outer_zone_thickness_ratio: { type: 'number', label: 'Outer Zone Ratio', min: 0.2, max: 0.5, step: 0.05, description: 'Vascular/red zone', advanced: true },
  middle_zone_thickness_ratio: { type: 'number', label: 'Middle Zone Ratio', min: 0.2, max: 0.5, step: 0.05, description: 'Red-white zone', advanced: true },
  inner_zone_thickness_ratio: { type: 'number', label: 'Inner Zone Ratio', min: 0.2, max: 0.5, step: 0.05, description: 'Avascular/white zone', advanced: true },

  // Circumferential fiber properties
  circumferential_bundle_diameter_um: { type: 'number', label: 'Circumferential Bundle Diameter', min: 50, max: 200, step: 10, unit: 'um', description: 'Fiber bundle diameter', advanced: true },
  circumferential_bundle_spacing_um: { type: 'number', label: 'Circumferential Bundle Spacing', min: 100, max: 400, step: 25, unit: 'um', description: 'Spacing between bundles', advanced: true },
  circumferential_fiber_density: { type: 'number', label: 'Circumferential Fiber Density', min: 0.4, max: 1.0, step: 0.1, description: 'Density in outer zone', advanced: true },

  // Radial (tie) fiber properties
  radial_bundle_diameter_um: { type: 'number', label: 'Radial Bundle Diameter', min: 10, max: 50, step: 5, unit: 'um', description: 'Tie fiber diameter', advanced: true },
  radial_bundle_spacing_um: { type: 'number', label: 'Radial Bundle Spacing', min: 50, max: 200, step: 10, unit: 'um', description: 'Spacing between radial fibers', advanced: true },
  radial_fiber_density: { type: 'number', label: 'Radial Fiber Density', min: 0.3, max: 1.0, step: 0.1, description: 'Tie fiber density', advanced: true },
  enable_radial_tie_fibers: { type: 'boolean', label: 'Enable Radial Tie Fibers', description: 'Include tie fibers', advanced: true },

  // General fiber properties
  fiber_diameter: { type: 'number', label: 'Fiber Diameter', min: 0.05, max: 0.5, step: 0.025, unit: 'mm', description: 'Default fiber diameter' },
  fiber_orientation_variance_deg: { type: 'number', label: 'Fiber Orientation Variance', min: 0, max: 30, step: 5, unit: 'deg', description: 'Random variance in fiber angle', advanced: true },

  // Porosity per zone
  outer_zone_porosity: { type: 'number', label: 'Outer Zone Porosity', min: 0.4, max: 0.8, step: 0.05, description: 'Vascular zone (higher for blood supply)', advanced: true },
  middle_zone_porosity: { type: 'number', label: 'Middle Zone Porosity', min: 0.35, max: 0.7, step: 0.05, description: 'Transition zone', advanced: true },
  inner_zone_porosity: { type: 'number', label: 'Inner Zone Porosity', min: 0.3, max: 0.6, step: 0.05, description: 'Avascular zone (denser)', advanced: true },
  porosity: { type: 'number', label: 'Overall Porosity', min: 0.4, max: 0.75, step: 0.05, description: 'Overall target porosity', advanced: true },

  // Vascular features
  enable_vascular_channels: { type: 'boolean', label: 'Enable Vascular Channels', description: 'Blood vessel channels in outer zone' },
  vascular_channel_diameter: { type: 'number', label: 'Vascular Channel Diameter', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Vessel diameter', advanced: true },
  vascular_channel_spacing: { type: 'number', label: 'Vascular Channel Spacing', min: 0.3, max: 1.5, step: 0.1, unit: 'mm', description: 'Spacing between vessels', advanced: true },
  vascular_penetration_depth: { type: 'number', label: 'Vascular Penetration Depth', min: 0.2, max: 0.5, step: 0.05, description: 'Fraction of radial depth', advanced: true },

  // Horn attachments
  enable_anterior_horn: { type: 'boolean', label: 'Enable Anterior Horn', description: 'Anterior horn attachment region', advanced: true },
  enable_posterior_horn: { type: 'boolean', label: 'Enable Posterior Horn', description: 'Posterior horn attachment region', advanced: true },
  horn_insertion_width: { type: 'number', label: 'Horn Insertion Width', min: 2, max: 10, step: 0.5, unit: 'mm', description: 'Width of horn insertion', advanced: true },
  horn_insertion_depth: { type: 'number', label: 'Horn Insertion Depth', min: 1, max: 6, step: 0.5, unit: 'mm', description: 'Depth of horn attachment', advanced: true },

  // Surface properties
  surface_roughness: { type: 'number', label: 'Surface Roughness', min: 0, max: 0.15, step: 0.01, description: 'Surface texture', advanced: true },
  enable_femoral_surface: { type: 'boolean', label: 'Enable Femoral Surface', description: 'Concave upper surface', advanced: true },
  enable_tibial_surface: { type: 'boolean', label: 'Enable Tibial Surface', description: 'Flatter lower surface', advanced: true },
  femoral_curvature_radius: { type: 'number', label: 'Femoral Curvature Radius', min: 15, max: 50, step: 5, unit: 'mm', description: 'Curvature matching femoral condyle', advanced: true },

  // Collagen organization
  enable_lamellar_structure: { type: 'boolean', label: 'Enable Lamellar Structure', description: 'Layered fiber organization', advanced: true },
  lamella_count: { type: 'number', label: 'Lamella Count', min: 3, max: 12, step: 1, description: 'Number of fiber layers', advanced: true },
  interlaminar_spacing_um: { type: 'number', label: 'Interlaminar Spacing', min: 20, max: 100, step: 10, unit: 'um', description: 'Space between layers', advanced: true },

  // Cell lacunae
  enable_cell_lacunae: { type: 'boolean', label: 'Enable Cell Lacunae', description: 'Chondrocyte spaces', advanced: true },
  lacuna_density: { type: 'number', label: 'Lacuna Density', min: 100, max: 1000, step: 50, description: 'Lacunae per mm3', advanced: true },
  lacuna_diameter_um: { type: 'number', label: 'Lacuna Diameter', min: 8, max: 25, step: 1, unit: 'um', description: 'Lacuna size', advanced: true },

  // Randomization
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.25, step: 0.05, description: 'Random position jitter', advanced: true },
  thickness_variance: { type: 'number', label: 'Thickness Variance', min: 0, max: 0.25, step: 0.05, description: 'Variation in local thickness', advanced: true },
};

export const TENDON_LIGAMENT_META: Record<string, ParamMeta> = {
  // Basic geometry
  length: { type: 'number', label: 'Length', min: 5, max: 100, step: 5, unit: 'mm', description: 'Total length' },
  width: { type: 'number', label: 'Width', min: 1, max: 25, step: 0.5, unit: 'mm', description: 'Total width' },
  thickness: { type: 'number', label: 'Thickness', min: 0.5, max: 10, step: 0.25, unit: 'mm', description: 'Total thickness' },
  resolution: { type: 'number', label: 'Resolution', min: 6, max: 24, step: 2, description: 'Mesh resolution', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducible randomization', advanced: true },

  // Fascicle organization
  bundle_count: { type: 'number', label: 'Bundle Count', min: 1, max: 12, step: 1, description: 'Number of primary fascicles' },
  fascicle_diameter: { type: 'number', label: 'Fascicle Diameter', min: 0.5, max: 3, step: 0.25, unit: 'mm', description: 'Fascicle diameter', advanced: true },
  fascicle_spacing: { type: 'number', label: 'Fascicle Spacing', min: 0.1, max: 1.0, step: 0.1, unit: 'mm', description: 'Gap between fascicles', advanced: true },
  enable_fascicle_boundaries: { type: 'boolean', label: 'Enable Fascicle Boundaries', description: 'Show endotenon boundaries', advanced: true },

  // Fiber properties
  fiber_diameter: { type: 'number', label: 'Fiber Diameter', min: 0.05, max: 0.4, step: 0.01, unit: 'mm', description: 'Fiber diameter' },
  fiber_diameter_um: { type: 'number', label: 'Fiber Diameter (um)', min: 50, max: 400, step: 25, unit: 'um', description: 'Fiber diameter in micrometers', advanced: true },
  fiber_spacing: { type: 'number', label: 'Fiber Spacing', min: 0.1, max: 1.0, step: 0.05, unit: 'mm', description: 'Spacing between fiber centers' },
  fiber_spacing_um: { type: 'number', label: 'Fiber Spacing (um)', min: 100, max: 1000, step: 50, unit: 'um', description: 'Spacing in micrometers', advanced: true },
  fibers_per_fascicle: { type: 'number', label: 'Fibers per Fascicle', min: 5, max: 50, step: 5, description: 'Number of fibers per fascicle', advanced: true },

  // Fibril properties
  fibril_diameter_um: { type: 'number', label: 'Fibril Diameter', min: 1, max: 15, step: 0.5, unit: 'um', description: 'Individual fibril diameter', advanced: true },
  fibril_spacing_um: { type: 'number', label: 'Fibril Spacing', min: 2, max: 30, step: 2, unit: 'um', description: 'Spacing between fibrils', advanced: true },
  enable_fibril_detail: { type: 'boolean', label: 'Enable Fibril Detail', description: 'Include fibril-level detail', advanced: true },

  // Crimp pattern
  crimp_amplitude: { type: 'number', label: 'Crimp Amplitude', min: 0.05, max: 1.0, step: 0.05, unit: 'mm', description: 'Wave amplitude' },
  crimp_amplitude_um: { type: 'number', label: 'Crimp Amplitude (um)', min: 50, max: 1000, step: 50, unit: 'um', description: 'Wave amplitude in micrometers', advanced: true },
  crimp_wavelength: { type: 'number', label: 'Crimp Wavelength', min: 0.05, max: 1.0, step: 0.05, unit: 'mm', description: 'Human tendon crimp pattern, typically 100-500µm' },
  crimp_wavelength_um: { type: 'number', label: 'Crimp Wavelength (um)', min: 50, max: 1000, step: 50, unit: 'um', description: 'Human tendon crimp pattern, typically 100-500µm', advanced: true },
  crimp_angle_deg: { type: 'number', label: 'Crimp Angle', min: 3, max: 20, step: 1, unit: 'deg', description: 'Crimp angle (5-15 deg typical)', advanced: true },
  crimp_variance: { type: 'number', label: 'Crimp Variance', min: 0, max: 0.3, step: 0.05, description: 'Random variation in crimp pattern', advanced: true },

  // Endotenon
  enable_endotenon: { type: 'boolean', label: 'Enable Endotenon', description: 'Connective tissue between fascicles', advanced: true },
  endotenon_thickness: { type: 'number', label: 'Endotenon Thickness', min: 0.01, max: 0.15, step: 0.01, unit: 'mm', description: 'Endotenon layer thickness', advanced: true },
  endotenon_porosity: { type: 'number', label: 'Endotenon Porosity', min: 0.3, max: 0.7, step: 0.05, description: 'Higher porosity for blood/nerve supply', advanced: true },

  // Epitenon
  enable_epitenon: { type: 'boolean', label: 'Enable Epitenon', description: 'Outer connective tissue sheath', advanced: true },
  epitenon_thickness: { type: 'number', label: 'Epitenon Thickness', min: 0.03, max: 0.25, step: 0.01, unit: 'mm', description: 'Epitenon layer thickness', advanced: true },
  epitenon_porosity: { type: 'number', label: 'Epitenon Porosity', min: 0.15, max: 0.5, step: 0.05, description: 'Moderate porosity', advanced: true },

  // Paratenon
  enable_paratenon: { type: 'boolean', label: 'Enable Paratenon', description: 'Loose areolar tissue layer', advanced: true },
  paratenon_thickness: { type: 'number', label: 'Paratenon Thickness', min: 0.1, max: 0.5, step: 0.05, unit: 'mm', description: 'Paratenon layer thickness', advanced: true },

  // Collagen alignment
  primary_fiber_angle_deg: { type: 'number', label: 'Primary Fiber Angle', min: -15, max: 15, step: 5, unit: 'deg', description: 'Main fiber direction (along length)', advanced: true },
  fiber_angle_variance_deg: { type: 'number', label: 'Fiber Angle Variance', min: 0, max: 15, step: 1, unit: 'deg', description: 'Slight angular variation', advanced: true },
  enable_cross_links: { type: 'boolean', label: 'Enable Cross Links', description: 'Cross-linking between fibers', advanced: true },
  cross_link_density: { type: 'number', label: 'Cross Link Density', min: 0, max: 0.3, step: 0.02, description: 'Cross-links per unit length', advanced: true },

  // Vascular features
  enable_vascular_channels: { type: 'boolean', label: 'Enable Vascular Channels', description: 'Blood vessel channels' },
  vascular_channel_diameter: { type: 'number', label: 'Vascular Channel Diameter', min: 0.03, max: 0.2, step: 0.01, unit: 'mm', description: 'Small vessel diameter', advanced: true },
  vascular_channel_spacing: { type: 'number', label: 'Vascular Channel Spacing', min: 0.5, max: 3.0, step: 0.25, unit: 'mm', description: 'Spacing between vessels', advanced: true },
  vascular_channel_pattern: { type: 'enum', label: 'Vascular Channel Pattern', options: [{ value: 'longitudinal', label: 'Longitudinal' }, { value: 'spiral', label: 'Spiral' }], description: 'Vessel pattern', advanced: true },

  // Insertion zones (enthesis)
  enable_enthesis_transition: { type: 'boolean', label: 'Enable Enthesis Transition', description: 'Bone insertion zone', advanced: true },
  enthesis_length: { type: 'number', label: 'Enthesis Length', min: 1, max: 8, step: 0.5, unit: 'mm', description: 'Length of transition zone', advanced: true },
  enthesis_mineralization_gradient: { type: 'number', label: 'Enthesis Mineralization Gradient', min: 0, max: 1, step: 0.1, description: 'Mineral content gradient', advanced: true },

  // Mechanical property indicators
  target_stiffness_mpa: { type: 'number', label: 'Target Stiffness', min: 200, max: 2000, step: 100, unit: 'MPa', description: 'Target elastic modulus indicator', advanced: true },
  target_ultimate_stress_mpa: { type: 'number', label: 'Target Ultimate Stress', min: 20, max: 200, step: 10, unit: 'MPa', description: 'Ultimate tensile strength indicator', advanced: true },

  // Porosity
  porosity: { type: 'number', label: 'Porosity', min: 0.05, max: 0.35, step: 0.05, description: 'Overall porosity (tendons are relatively dense)', advanced: true },
  pore_size_um: { type: 'number', label: 'Pore Size', min: 20, max: 120, step: 10, unit: 'um', description: 'Interfiber pore size', advanced: true },

  // Surface characteristics
  surface_roughness: { type: 'number', label: 'Surface Roughness', min: 0, max: 0.1, step: 0.01, description: 'Surface texture', advanced: true },
  enable_surface_texture: { type: 'boolean', label: 'Enable Surface Texture', description: 'Textured exterior', advanced: true },

  // Randomization
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.15, step: 0.01, description: 'Random position jitter', advanced: true },
  diameter_variance: { type: 'number', label: 'Diameter Variance', min: 0, max: 0.25, step: 0.05, description: 'Variation in fiber diameter', advanced: true },
  spacing_variance: { type: 'number', label: 'Spacing Variance', min: 0, max: 0.25, step: 0.05, description: 'Variation in fiber spacing', advanced: true },
};

export const INTERVERTEBRAL_DISC_META: Record<string, ParamMeta> = {
  // Basic geometry
  disc_diameter: { type: 'number', label: 'Disc Diameter', min: 25, max: 60, step: 2.5, unit: 'mm', description: 'Total disc diameter (30-50 mm typical)' },
  disc_height: { type: 'number', label: 'Disc Height', min: 4, max: 18, step: 0.5, unit: 'mm', description: 'Total disc height (5-15 mm typical)' },
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 32, step: 2, description: 'Mesh resolution', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducible randomization', advanced: true },

  // Nucleus pulposus (NP)
  np_diameter: { type: 'number', label: 'NP Diameter', min: 8, max: 30, step: 1, unit: 'mm', description: 'NP diameter (40% of disc diameter typical)' },
  nucleus_percentage: { type: 'number', label: 'Nucleus Percentage', min: 0.25, max: 0.55, step: 0.05, description: 'NP as fraction of disc area', advanced: true },
  np_height_ratio: { type: 'number', label: 'NP Height Ratio', min: 0.6, max: 0.95, step: 0.05, description: 'NP height relative to disc height', advanced: true },
  np_porosity: { type: 'number', label: 'NP Porosity', min: 0.8, max: 0.98, step: 0.02, description: 'Very high porosity (gel-like)' },
  np_water_content: { type: 'number', label: 'NP Water Content', min: 0.6, max: 0.95, step: 0.05, description: 'Water content indicator', advanced: true },
  np_proteoglycan_content: { type: 'number', label: 'NP Proteoglycan Content', min: 0.5, max: 1.0, step: 0.1, description: 'Relative proteoglycan density', advanced: true },
  np_pore_size: { type: 'number', label: 'NP Pore Size', min: 0.1, max: 0.6, step: 0.05, unit: 'mm', description: 'Pore size in NP region', advanced: true },

  // Annulus fibrosus (AF)
  af_ring_count: { type: 'number', label: 'AF Ring Count', min: 2, max: 8, step: 1, description: 'Simplified ring count (legacy)' },
  num_lamellae: { type: 'number', label: 'Number of Lamellae', min: 10, max: 30, step: 1, description: 'Actual concentric lamellae count (15-25 typical)', advanced: true },
  annulus_percentage: { type: 'number', label: 'Annulus Percentage', min: 0.45, max: 0.75, step: 0.05, description: 'AF as fraction of disc area', advanced: true },
  af_porosity: { type: 'number', label: 'AF Porosity', min: 0.45, max: 0.8, step: 0.05, description: 'Lower porosity than NP', advanced: true },
  af_water_content: { type: 'number', label: 'AF Water Content', min: 0.4, max: 0.75, step: 0.05, description: 'Water content indicator', advanced: true },

  // Fiber properties
  fiber_diameter: { type: 'number', label: 'Fiber Diameter', min: 0.1, max: 0.4, step: 0.025, unit: 'mm', description: 'Collagen fiber bundle diameter' },
  fiber_diameter_um: { type: 'number', label: 'Fiber Diameter (um)', min: 100, max: 400, step: 25, unit: 'um', description: 'Fiber diameter in micrometers', advanced: true },
  fiber_angle: { type: 'number', label: 'Fiber Angle', min: 20, max: 45, step: 5, unit: 'deg', description: 'Mean fiber angle (legacy)', advanced: true },
  af_layer_angle: { type: 'number', label: 'AF Layer Angle', min: 20, max: 45, step: 5, unit: 'deg', description: 'Base alternating angle' },
  inner_af_fiber_angle_deg: { type: 'number', label: 'Inner AF Fiber Angle', min: 35, max: 55, step: 5, unit: 'deg', description: 'Inner AF fiber angle', advanced: true },
  outer_af_fiber_angle_deg: { type: 'number', label: 'Outer AF Fiber Angle', min: 55, max: 75, step: 5, unit: 'deg', description: 'Outer AF fiber angle', advanced: true },
  fiber_angle_variance_deg: { type: 'number', label: 'Fiber Angle Variance', min: 0, max: 15, step: 1, unit: 'deg', description: 'Random variance in fiber angle', advanced: true },

  // Lamella properties
  lamella_thickness: { type: 'number', label: 'Lamella Thickness', min: 0.05, max: 0.3, step: 0.025, unit: 'mm', description: 'Individual lamella thickness', advanced: true },
  interlaminar_spacing: { type: 'number', label: 'Interlaminar Spacing', min: 0.01, max: 0.1, step: 0.01, unit: 'mm', description: 'Gap between lamellae', advanced: true },
  enable_interlaminar_connections: { type: 'boolean', label: 'Enable Interlaminar Connections', description: 'Cross-links between lamellae', advanced: true },

  // Mechanical property indicators
  np_stiffness_kpa: { type: 'number', label: 'NP Stiffness', min: 0.5, max: 10, step: 0.5, unit: 'kPa', description: 'NP aggregate modulus (~1-5 kPa)', advanced: true },
  af_stiffness_kpa: { type: 'number', label: 'AF Stiffness', min: 30, max: 300, step: 10, unit: 'kPa', description: 'AF tensile modulus (~50-200 kPa)', advanced: true },
  stiffness_gradient: { type: 'boolean', label: 'Stiffness Gradient', description: 'Gradual stiffness change from NP to AF', advanced: true },

  // Cartilaginous endplates
  enable_endplates: { type: 'boolean', label: 'Enable Endplates', description: 'Include cartilaginous endplates' },
  endplate_thickness: { type: 'number', label: 'Endplate Thickness', min: 0.3, max: 1.5, step: 0.1, unit: 'mm', description: 'Endplate thickness (0.5-1.0 mm typical)', advanced: true },
  endplate_porosity: { type: 'number', label: 'Endplate Porosity', min: 0.35, max: 0.7, step: 0.05, description: 'Endplate porosity', advanced: true },
  endplate_pore_size: { type: 'number', label: 'Endplate Pore Size', min: 0.05, max: 0.25, step: 0.025, unit: 'mm', description: 'Pore size in endplates', advanced: true },

  // Nutrient pathway features
  enable_nutrient_channels: { type: 'boolean', label: 'Enable Nutrient Channels', description: 'Diffusion channels', advanced: true },
  nutrient_channel_diameter: { type: 'number', label: 'Nutrient Channel Diameter', min: 0.02, max: 0.15, step: 0.01, unit: 'mm', description: 'Small channel diameter', advanced: true },
  nutrient_channel_density: { type: 'number', label: 'Nutrient Channel Density', min: 0.1, max: 0.6, step: 0.05, description: 'Channels per mm2', advanced: true },
  enable_endplate_pores: { type: 'boolean', label: 'Enable Endplate Pores', description: 'Pores through endplate', advanced: true },

  // Transition zone (NP-AF interface)
  transition_zone_width: { type: 'number', label: 'Transition Zone Width', min: 0.2, max: 1.5, step: 0.1, unit: 'mm', description: 'Width of NP-AF transition', advanced: true },
  transition_gradient_type: { type: 'enum', label: 'Transition Gradient Type', options: [{ value: 'linear', label: 'Linear' }, { value: 'sigmoid', label: 'Sigmoid' }, { value: 'step', label: 'Step' }], description: 'Transition profile', advanced: true },

  // Vascular features
  enable_outer_vascular: { type: 'boolean', label: 'Enable Outer Vascular', description: 'Blood vessels in outer AF', advanced: true },
  vascular_channel_diameter: { type: 'number', label: 'Vascular Channel Diameter', min: 0.05, max: 0.25, step: 0.025, unit: 'mm', description: 'Vessel diameter', advanced: true },
  vascular_penetration_depth: { type: 'number', label: 'Vascular Penetration Depth', min: 0.1, max: 0.4, step: 0.05, description: 'Fraction of AF thickness', advanced: true },

  // Notochordal remnants
  enable_notochordal_cells: { type: 'boolean', label: 'Enable Notochordal Cells', description: 'Notochordal cell region', advanced: true },
  notochordal_region_radius: { type: 'number', label: 'Notochordal Region Radius', min: 0.5, max: 5, step: 0.5, unit: 'mm', description: 'Central notochordal area', advanced: true },

  // Age-related changes
  degeneration_level: { type: 'number', label: 'Degeneration Level', min: 0, max: 1, step: 0.1, description: '0 = healthy, 1 = fully degenerated', advanced: true },
  enable_fissures: { type: 'boolean', label: 'Enable Fissures', description: 'Radial or circumferential tears', advanced: true },
  fissure_count: { type: 'number', label: 'Fissure Count', min: 0, max: 6, step: 1, description: 'Number of fissures', advanced: true },

  // Surface and randomization
  surface_roughness: { type: 'number', label: 'Surface Roughness', min: 0, max: 0.15, step: 0.01, description: 'Surface texture', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.25, step: 0.05, description: 'Random position jitter', advanced: true },
  fiber_variance: { type: 'number', label: 'Fiber Variance', min: 0, max: 0.25, step: 0.05, description: 'Variation in fiber properties', advanced: true },
  lamella_variance: { type: 'number', label: 'Lamella Variance', min: 0, max: 0.15, step: 0.01, description: 'Variation in lamella thickness', advanced: true },
};

// Combined export for skeletal scaffold types
export const SKELETAL_PARAMETER_META: Partial<Record<ScaffoldType, Record<string, ParamMeta>>> = {
  [ScaffoldType.TRABECULAR_BONE]: TRABECULAR_BONE_META,
  [ScaffoldType.HAVERSIAN_BONE]: HAVERSIAN_BONE_META,
  [ScaffoldType.OSTEOCHONDRAL]: OSTEOCHONDRAL_META,
  [ScaffoldType.ARTICULAR_CARTILAGE]: ARTICULAR_CARTILAGE_META,
  [ScaffoldType.MENISCUS]: MENISCUS_META,
  [ScaffoldType.TENDON_LIGAMENT]: TENDON_LIGAMENT_META,
  [ScaffoldType.INTERVERTEBRAL_DISC]: INTERVERTEBRAL_DISC_META,
};
