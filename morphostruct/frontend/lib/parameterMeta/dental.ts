/**
 * Parameter metadata for dental scaffold types
 * Includes: dentin_pulp, ear_auricle, nasal_septum
 */

import { ScaffoldType } from '../types/scaffolds';
import { ParamMeta } from './types';

export const DENTIN_PULP_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  tooth_height: { type: 'number', label: 'Tooth Height', min: 5, max: 25, step: 0.5, unit: 'mm', description: 'Total height (crown + root)' },
  crown_diameter: { type: 'number', label: 'Crown Diameter', min: 4, max: 15, step: 0.5, unit: 'mm', description: 'Mesiodistal width of crown' },
  crown_height: { type: 'number', label: 'Crown Height', min: 2, max: 10, step: 0.5, unit: 'mm', description: 'Height of anatomical crown' },
  root_length: { type: 'number', label: 'Root Length', min: 5, max: 20, step: 0.5, unit: 'mm', description: 'Molar root ~12mm typical' },
  root_diameter: { type: 'number', label: 'Root Diameter', min: 1, max: 6, step: 0.25, unit: 'mm', description: 'Diameter at root tip' },
  // === Dentin Tubule Parameters (Microstructure) ===
  tubule_diameter_dej: { type: 'number', label: 'Tubule Diameter at DEJ', min: 0.5, max: 1.5, step: 0.1, unit: 'um', description: '0.9-1.2 μm at DEJ', advanced: true },
  tubule_diameter_pulp: { type: 'number', label: 'Tubule Diameter at Pulp', min: 2, max: 6, step: 0.1, unit: 'um', description: 'Tubule diameter near pulp (2.5-5.0um)', advanced: true },
  tubule_density_dej: { type: 'number', label: 'Tubule Density at DEJ', min: 1e6, max: 3e6, step: 1e5, description: 'Tubules per cm2 at DEJ (~19,000/mm2)', advanced: true },
  tubule_density_pulp: { type: 'number', label: 'Tubule Density at Pulp', min: 45000, max: 90000, step: 5000, description: 'Higher near pulp (50,000-90,000/mm²)', advanced: true },
  enable_tubule_representation: { type: 'boolean', label: 'Enable Tubule Representation', description: 'Simplified tubule channels (performance intensive)', advanced: true },
  tubule_resolution: { type: 'number', label: 'Tubule Resolution', min: 2, max: 8, step: 1, description: 'Segments per tubule cylinder', advanced: true },
  // === Pulp Chamber ===
  pulp_chamber_height: { type: 'number', label: 'Pulp Chamber Height', min: 1, max: 5, step: 0.25, unit: 'mm', description: 'Typical pulp chamber height (~2.76mm)', advanced: true },
  pulp_chamber_width: { type: 'number', label: 'Pulp Chamber Width', min: 1, max: 5, step: 0.25, unit: 'mm', description: 'Buccolingual width of pulp chamber', advanced: true },
  pulp_chamber_size: { type: 'number', label: 'Pulp Chamber Size', min: 0.1, max: 0.8, step: 0.05, description: 'Relative size ratio (0-1)' },
  pulp_horn_count: { type: 'number', label: 'Pulp Horn Count', min: 1, max: 5, step: 1, description: 'Number of pulp horns (2-5 for molars)', advanced: true },
  pulp_horn_height: { type: 'number', label: 'Pulp Horn Height', min: 0.1, max: 1.5, step: 0.1, unit: 'mm', description: 'Additional height for pulp horns', advanced: true },
  root_canal_taper: { type: 'number', label: 'Root Canal Taper', min: 0.02, max: 0.12, step: 0.01, description: 'Root canal taper mm/mm (0.04-0.08 typical)', advanced: true },
  // === Dentin-Enamel Junction (DEJ) ===
  dej_scallop_size: { type: 'number', label: 'DEJ Scallop Size', min: 20, max: 120, step: 5, unit: 'um', description: 'DEJ scallop convexity size (25-100um)', advanced: true },
  dej_scallop_count: { type: 'number', label: 'DEJ Scallop Count', min: 6, max: 24, step: 1, description: 'Number of scallops around circumference', advanced: true },
  dej_width: { type: 'number', label: 'DEJ Width', min: 5, max: 15, step: 0.5, unit: 'um', description: 'DEJ interface width (~8-10um)', advanced: true },
  enable_dej_texture: { type: 'boolean', label: 'Enable DEJ Texture', description: 'Add DEJ scallop texture', advanced: true },
  // === Dentin Layers ===
  dentin_thickness: { type: 'number', label: 'Dentin Thickness', min: 1, max: 2.5, step: 0.25, unit: 'mm', description: '1.5-2.2 mm typical' },
  peritubular_dentin_ratio: { type: 'number', label: 'Peritubular Dentin Ratio', min: 0.05, max: 0.3, step: 0.01, description: 'Ratio of highly mineralized peritubular dentin', advanced: true },
  intertubular_dentin_ratio: { type: 'number', label: 'Intertubular Dentin Ratio', min: 0.7, max: 0.95, step: 0.01, description: 'Ratio of less mineralized intertubular dentin', advanced: true },
  mantle_dentin_thickness: { type: 'number', label: 'Mantle Dentin Thickness', min: 0.01, max: 0.05, step: 0.01, unit: 'mm', description: '15-30 μm native', advanced: true },
  // === Enamel Interface ===
  enamel_interface_roughness: { type: 'number', label: 'Enamel Interface Roughness', min: 0, max: 1, step: 0.1, description: 'Surface roughness at crown (0-1)', advanced: true },
  enamel_thickness_occlusal: { type: 'number', label: 'Enamel Thickness Occlusal', min: 1, max: 4, step: 0.25, unit: 'mm', description: 'Enamel thickness at occlusal surface', advanced: true },
  enamel_thickness_cervical: { type: 'number', label: 'Enamel Thickness Cervical', min: 0.1, max: 1, step: 0.1, unit: 'mm', description: 'Enamel thickness at cervical margin', advanced: true },
  enable_enamel_shell: { type: 'boolean', label: 'Enable Enamel Shell', description: 'Add enamel outer shell (crown only)', advanced: true },
  // === Root Features ===
  root_count: { type: 'number', label: 'Root Count', min: 1, max: 4, step: 1, description: 'Number of roots (1 for anterior, 2-3 for molars)' },
  root_furcation_height: { type: 'number', label: 'Root Furcation Height', min: 1, max: 6, step: 0.5, unit: 'mm', description: 'Height where roots divide (multi-rooted)', advanced: true },
  cementum_thickness: { type: 'number', label: 'Cementum Thickness', min: 0.02, max: 0.3, step: 0.02, unit: 'mm', description: 'Cementum layer on root (~50-200um)', advanced: true },
  enable_cementum: { type: 'boolean', label: 'Enable Cementum', description: 'Add cementum layer on root', advanced: true },
  // === Generation Settings ===
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 32, step: 4, description: 'Circular resolution for mesh generation', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  randomness: { type: 'number', label: 'Randomness', min: 0, max: 0.5, step: 0.05, description: 'Overall randomness intensity (0-1)', advanced: true },
  detail_level: { type: 'enum', label: 'Detail Level', options: [{ value: 'low', label: 'Low' }, { value: 'medium', label: 'Medium' }, { value: 'high', label: 'High' }], description: 'Mesh detail level', advanced: true },
};

export const EAR_AURICLE_META: Record<string, ParamMeta> = {
  // === Overall Dimensions ===
  overall_height: { type: 'number', label: 'Overall Height', min: 40, max: 80, step: 2, unit: 'mm', description: 'Total ear height (adult ~60-65mm)' },
  overall_width: { type: 'number', label: 'Overall Width', min: 20, max: 50, step: 2, unit: 'mm', description: 'Total ear width (adult ~30-35mm)' },
  overall_depth: { type: 'number', label: 'Overall Depth', min: 10, max: 30, step: 2, unit: 'mm', description: 'Ear projection from skull' },
  scale_factor: { type: 'number', label: 'Scale Factor', min: 0.5, max: 1.5, step: 0.05, description: 'Overall size multiplier' },
  // === Cartilage Framework ===
  cartilage_thickness: { type: 'number', label: 'Cartilage Thickness', min: 0.5, max: 3.0, step: 0.1, unit: 'mm', description: '1-3 mm native elastic cartilage' },
  skin_thickness: { type: 'number', label: 'Skin Thickness', min: 0.5, max: 2, step: 0.1, unit: 'mm', description: 'Overlying skin thickness', advanced: true },
  thickness: { type: 'number', label: 'Combined Thickness', min: 1, max: 5, step: 0.25, unit: 'mm', description: 'Combined cartilage+skin thickness' },
  // === Structural Elements ===
  strut_width: { type: 'number', label: 'Strut Width', min: 0.2, max: 1, step: 0.1, unit: 'mm', description: 'Lattice strut width (~500um)', advanced: true },
  strut_spacing: { type: 'number', label: 'Strut Spacing', min: 0.5, max: 3, step: 0.25, unit: 'mm', description: 'Spacing between lattice struts', advanced: true },
  pore_size: { type: 'number', label: 'Pore Size', min: 0.1, max: 0.5, step: 0.05, unit: 'mm', description: '150-250 μm optimal for chondrogenesis', advanced: true },
  pore_shape: { type: 'enum', label: 'Pore Shape', options: [{ value: 'circular', label: 'Circular' }, { value: 'hexagonal', label: 'Hexagonal' }], description: 'Pore geometry type', advanced: true },
  // === Helix (Outer Rim) ===
  helix_definition: { type: 'number', label: 'Helix Definition', min: 0, max: 1, step: 0.05, description: 'Prominence of helix rim (0-1)' },
  helix_curvature: { type: 'number', label: 'Helix Curvature', min: 0, max: 1, step: 0.05, description: 'Curvature of helix fold (0-1)', advanced: true },
  helix_width: { type: 'number', label: 'Helix Width', min: 4, max: 15, step: 1, unit: 'mm', description: 'Width of helix rim', advanced: true },
  helix_thickness_factor: { type: 'number', label: 'Helix Thickness Factor', min: 0.8, max: 2, step: 0.1, description: 'Multiplier for helix thickness', advanced: true },
  // === Antihelix (Inner Ridge) ===
  antihelix_depth: { type: 'number', label: 'Antihelix Depth', min: 0, max: 1, step: 0.05, description: 'Depth of antihelix ridge (0-1)' },
  antihelix_curvature: { type: 'number', label: 'Antihelix Curvature', min: 0, max: 1, step: 0.05, description: 'Curvature of antihelix (0-1)', advanced: true },
  antihelix_bifurcation: { type: 'boolean', label: 'Antihelix Bifurcation', description: 'Include superior/inferior crura', advanced: true },
  crura_angle: { type: 'number', label: 'Crura Angle', min: 15, max: 60, step: 5, unit: 'deg', description: 'Angle between crura', advanced: true },
  // === Concha (Bowl) ===
  concha_depth: { type: 'number', label: 'Concha Depth', min: 8, max: 25, step: 1, unit: 'mm', description: 'Depth of conchal bowl', advanced: true },
  concha_diameter: { type: 'number', label: 'Concha Diameter', min: 12, max: 30, step: 2, unit: 'mm', description: 'Diameter of concha', advanced: true },
  cymba_conchae_ratio: { type: 'number', label: 'Cymba Conchae Ratio', min: 0.2, max: 0.6, step: 0.05, description: 'Ratio of cymba to cavum', advanced: true },
  // === Tragus and Antitragus ===
  tragus_width: { type: 'number', label: 'Tragus Width', min: 4, max: 15, step: 1, unit: 'mm', description: 'Tragus projection width', advanced: true },
  tragus_height: { type: 'number', label: 'Tragus Height', min: 5, max: 15, step: 1, unit: 'mm', description: 'Tragus height', advanced: true },
  tragus_projection: { type: 'number', label: 'Tragus Projection', min: 2, max: 10, step: 0.5, unit: 'mm', description: 'Anterior projection of tragus', advanced: true },
  antitragus_size: { type: 'number', label: 'Antitragus Size', min: 3, max: 10, step: 0.5, unit: 'mm', description: 'Size of antitragus', advanced: true },
  // === Lobule (Earlobe) ===
  lobule_length: { type: 'number', label: 'Lobule Length', min: 10, max: 30, step: 2, unit: 'mm', description: 'Earlobe length (adult ~15-20mm)', advanced: true },
  lobule_width: { type: 'number', label: 'Lobule Width', min: 8, max: 25, step: 2, unit: 'mm', description: 'Earlobe width', advanced: true },
  lobule_thickness: { type: 'number', label: 'Lobule Thickness', min: 2, max: 8, step: 0.5, unit: 'mm', description: 'Soft tissue thickness (no cartilage)', advanced: true },
  lobule_attached: { type: 'boolean', label: 'Lobule Attached', description: 'Attached vs free earlobe', advanced: true },
  // === Mechanical Properties ===
  mechanical_modulus_ratio: { type: 'number', label: 'Mechanical Modulus Ratio', min: 0.3, max: 1, step: 0.05, description: 'Ratio vs native cartilage (target ~0.5-0.8)', advanced: true },
  target_porosity: { type: 'number', label: 'Target Porosity', min: 0.5, max: 0.9, step: 0.05, description: 'Target void fraction for cell infiltration', advanced: true },
  // === Surface Features ===
  enable_surface_texture: { type: 'boolean', label: 'Enable Surface Texture', description: 'Surface microstructure', advanced: true },
  texture_roughness: { type: 'number', label: 'Texture Roughness', min: 0, max: 1, step: 0.1, description: 'Surface roughness intensity (0-1)', advanced: true },
  // === Generation Settings ===
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 32, step: 4, description: 'Circular resolution for mesh generation', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  randomness: { type: 'number', label: 'Randomness', min: 0, max: 0.5, step: 0.05, description: 'Overall randomness intensity (0-1)', advanced: true },
  detail_level: { type: 'enum', label: 'Detail Level', options: [{ value: 'low', label: 'Low' }, { value: 'medium', label: 'Medium' }, { value: 'high', label: 'High' }], description: 'Mesh detail level', advanced: true },
};

export const NASAL_SEPTUM_META: Record<string, ParamMeta> = {
  // === Overall Dimensions ===
  height: { type: 'number', label: 'Height', min: 20, max: 45, step: 1, unit: 'mm', description: 'Superior-inferior height (adult ~30-35mm)' },
  length: { type: 'number', label: 'Length', min: 20, max: 45, step: 1, unit: 'mm', description: 'Anterior-posterior length (adult ~25-35mm)' },
  surface_area_target: { type: 'number', label: 'Surface Area Target', min: 5, max: 15, step: 0.5, unit: 'cm2', description: 'Target surface area (~8.18 cm2 typical)', advanced: true },
  width_superior: { type: 'number', label: 'Width Superior', min: 15, max: 35, step: 2, unit: 'mm', description: 'Width at superior edge', advanced: true },
  width_inferior: { type: 'number', label: 'Width Inferior', min: 25, max: 50, step: 2, unit: 'mm', description: 'Width at inferior edge (usually wider)', advanced: true },
  // === Thickness Profile ===
  thickness: { type: 'number', label: 'Average Thickness', min: 0.5, max: 4, step: 0.25, unit: 'mm', description: 'Average cartilage thickness' },
  thickness_min: { type: 'number', label: 'Minimum Thickness', min: 0.5, max: 2, step: 0.1, unit: 'mm', description: 'Min thickness at mid-septum region', advanced: true },
  thickness_max: { type: 'number', label: 'Maximum Thickness', min: 2, max: 5, step: 0.25, unit: 'mm', description: 'Max thickness at base/vomer junction', advanced: true },
  thickness_base: { type: 'number', label: 'Base Thickness', min: 1.5, max: 4, step: 0.25, unit: 'mm', description: 'Thickness at base of septum', advanced: true },
  thickness_mid: { type: 'number', label: 'Mid Thickness', min: 0.5, max: 2, step: 0.1, unit: 'mm', description: 'Thickness at mid-septum', advanced: true },
  thickness_anterior: { type: 'number', label: 'Anterior Thickness', min: 0.5, max: 3, step: 0.25, unit: 'mm', description: 'Thickness at anterior edge', advanced: true },
  enable_thickness_gradient: { type: 'boolean', label: 'Enable Thickness Gradient', description: 'Vary thickness across septum', advanced: true },
  // === Quadrangular Cartilage Shape ===
  anterior_height: { type: 'number', label: 'Anterior Height', min: 18, max: 40, step: 2, unit: 'mm', description: 'Anterior border height', advanced: true },
  posterior_height: { type: 'number', label: 'Posterior Height', min: 12, max: 30, step: 2, unit: 'mm', description: 'Posterior border height (perpendicular plate)', advanced: true },
  dorsal_length: { type: 'number', label: 'Dorsal Length', min: 15, max: 40, step: 2, unit: 'mm', description: 'Length along nasal dorsum', advanced: true },
  basal_length: { type: 'number', label: 'Basal Length', min: 20, max: 45, step: 2, unit: 'mm', description: 'Length at base (maxillary crest)', advanced: true },
  // === Curvature and Deviation ===
  curvature_radius: { type: 'number', label: 'Curvature Radius', min: 40, max: 200, step: 10, unit: 'mm', description: 'Primary radius of curvature' },
  curvature_secondary: { type: 'number', label: 'Secondary Curvature', min: 80, max: 300, step: 20, unit: 'mm', description: 'Secondary curvature (perpendicular)', advanced: true },
  curve_type: { type: 'enum', label: 'Curve Type', options: [{ value: 'single', label: 'Single' }, { value: 's_curve', label: 'S-Curve' }, { value: 'complex', label: 'Complex' }], description: 'Deviation type' },
  deviation_angle: { type: 'number', label: 'Deviation Angle', min: 0, max: 20, step: 1, unit: 'deg', description: 'Deviation from midline', advanced: true },
  deviation_location: { type: 'number', label: 'Deviation Location', min: 0, max: 1, step: 0.1, description: 'Where max deviation occurs (0-1)', advanced: true },
  // === Cartilage Properties ===
  cartilage_porosity: { type: 'number', label: 'Cartilage Porosity', min: 0.4, max: 0.85, step: 0.05, description: 'Scaffold porosity target', advanced: true },
  pore_size: { type: 'number', label: 'Pore Size', min: 0.1, max: 0.6, step: 0.05, unit: 'mm', description: 'Pore diameter for cell infiltration', advanced: true },
  enable_porous_structure: { type: 'boolean', label: 'Enable Porous Structure', description: 'Add pore network', advanced: true },
  // === Cell Seeding Ratios ===
  cell_ratio_adsc_chondrocyte: { type: 'number', label: 'ADSC:Chondrocyte Ratio', min: 0.1, max: 0.5, step: 0.05, description: 'ADSC to chondrocyte ratio (1:3 = 0.25)', advanced: true },
  enable_cell_guidance_channels: { type: 'boolean', label: 'Enable Cell Guidance Channels', description: 'Microchannels for cell migration', advanced: true },
  // === Three-Layer Architecture ===
  three_layer_structure: { type: 'boolean', label: 'Three-Layer Structure', description: 'Perichondrium-cartilage-perichondrium', advanced: true },
  perichondrium_thickness: { type: 'number', label: 'Perichondrium Thickness', min: 0.05, max: 0.3, step: 0.02, unit: 'mm', description: 'Outer perichondrium layer (~100um)', advanced: true },
  core_cartilage_ratio: { type: 'number', label: 'Core Cartilage Ratio', min: 0.6, max: 0.95, step: 0.05, description: 'Ratio of core cartilage to total thickness', advanced: true },
  // === Surface Features ===
  enable_mucosal_texture: { type: 'boolean', label: 'Enable Mucosal Texture', description: 'Surface texture for mucosal adhesion', advanced: true },
  mucosal_groove_depth: { type: 'number', label: 'Mucosal Groove Depth', min: 0.02, max: 0.15, step: 0.01, unit: 'mm', description: 'Groove depth for mucosa attachment', advanced: true },
  enable_vascular_channels: { type: 'boolean', label: 'Enable Vascular Channels', description: 'Channels for vascularization', advanced: true },
  vascular_channel_diameter: { type: 'number', label: 'Vascular Channel Diameter', min: 0.1, max: 0.5, step: 0.05, unit: 'mm', description: 'Channel diameter', advanced: true },
  vascular_channel_spacing: { type: 'number', label: 'Vascular Channel Spacing', min: 1, max: 5, step: 0.5, unit: 'mm', description: 'Spacing between channels', advanced: true },
  // === Edges and Margins ===
  edge_rounding: { type: 'number', label: 'Edge Rounding', min: 0.2, max: 1.5, step: 0.1, unit: 'mm', description: 'Radius of edge rounding', advanced: true },
  enable_suture_holes: { type: 'boolean', label: 'Enable Suture Holes', description: 'Holes for surgical fixation', advanced: true },
  suture_hole_diameter: { type: 'number', label: 'Suture Hole Diameter', min: 0.5, max: 2, step: 0.25, unit: 'mm', description: 'Diameter of suture holes', advanced: true },
  suture_hole_spacing: { type: 'number', label: 'Suture Hole Spacing', min: 3, max: 10, step: 0.5, unit: 'mm', description: 'Spacing between suture holes', advanced: true },
  // === Generation Settings ===
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 32, step: 4, description: 'Mesh resolution', advanced: true },
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  randomness: { type: 'number', label: 'Randomness', min: 0, max: 0.5, step: 0.05, description: 'Overall randomness intensity (0-1)', advanced: true },
  detail_level: { type: 'enum', label: 'Detail Level', options: [{ value: 'low', label: 'Low' }, { value: 'medium', label: 'Medium' }, { value: 'high', label: 'High' }], description: 'Mesh detail level', advanced: true },
};

// Combined export for dental scaffold types
export const DENTAL_PARAMETER_META: Partial<Record<ScaffoldType, Record<string, ParamMeta>>> = {
  [ScaffoldType.DENTIN_PULP]: DENTIN_PULP_META,
  [ScaffoldType.EAR_AURICLE]: EAR_AURICLE_META,
  [ScaffoldType.NASAL_SEPTUM]: NASAL_SEPTUM_META,
};
