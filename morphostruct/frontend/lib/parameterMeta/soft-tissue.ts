/**
 * Parameter metadata for soft tissue scaffold types
 * Includes: multilayer_skin, skeletal_muscle, cornea, adipose
 */

import { ScaffoldType } from '../types/scaffolds';
import { ParamMeta } from './types';

export const MULTILAYER_SKIN_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  diameter_mm: { type: 'number', label: 'Diameter', min: 5, max: 50, step: 1, unit: 'mm', description: 'Overall scaffold diameter' },

  // === Layer Thicknesses ===
  epidermis_thickness_um: { type: 'number', label: 'Epidermis Thickness', min: 30, max: 600, step: 10, unit: 'um', description: 'Varies by body site: 30µm (eyelid) to 600µm (palm/sole)' },
  keratinocyte_layers: { type: 'number', label: 'Keratinocyte Layers', min: 3, max: 8, step: 1, description: 'Number of cell layers in epidermis (4-6 typical)', advanced: true },
  dermis_thickness_mm: { type: 'number', label: 'Dermis Thickness', min: 1, max: 4, step: 0.1, unit: 'mm', description: '1-4 mm typical' },
  papillary_dermis_thickness_um: { type: 'number', label: 'Papillary Dermis Thickness', min: 100, max: 400, step: 25, unit: 'um', description: '~20% of dermis, loose connective tissue', advanced: true },
  reticular_dermis_thickness_mm: { type: 'number', label: 'Reticular Dermis Thickness', min: 0.5, max: 3.5, step: 0.1, unit: 'mm', description: '~80% of dermis, dense connective tissue', advanced: true },
  hypodermis_thickness_mm: { type: 'number', label: 'Hypodermis Thickness', min: 1, max: 50, step: 1, unit: 'mm', description: '1-50 mm depending on location' },

  // === Rete Ridges (dermal-epidermal junction) ===
  enable_rete_ridges: { type: 'boolean', label: 'Enable Rete Ridges', description: 'Dermal-epidermal junction ridges', advanced: true },
  rete_ridge_height_um: { type: 'number', label: 'Rete Ridge Height', min: 35, max: 65, step: 5, unit: 'um', description: '35-65 um typical', advanced: true },
  rete_ridge_width_um: { type: 'number', label: 'Rete Ridge Width', min: 100, max: 200, step: 10, unit: 'um', description: '100-200 um', advanced: true },
  rete_ridge_spacing_um: { type: 'number', label: 'Rete Ridge Spacing', min: 70, max: 140, step: 10, unit: 'um', description: '70-140 um center-to-center', advanced: true },
  rete_ridge_depth_variance: { type: 'number', label: 'Rete Ridge Depth Variance', min: 0, max: 0.5, step: 0.05, description: 'Random variation in depth', advanced: true },

  // === Dermal Papillae ===
  enable_dermal_papillae: { type: 'boolean', label: 'Enable Dermal Papillae', description: 'Finger-like projections into epidermis', advanced: true },
  papillae_density_per_mm2: { type: 'number', label: 'Papillae Density', min: 50, max: 200, step: 10, unit: '/mm2', description: '50-200 per mm²', advanced: true },
  papillae_height_um: { type: 'number', label: 'Papillae Height', min: 50, max: 200, step: 5, unit: 'μm', description: '50-200 μm typical height', advanced: true },
  papillae_diameter_um: { type: 'number', label: 'Papillae Diameter', min: 50, max: 150, step: 5, unit: 'μm', description: '50-150 μm typical diameter', advanced: true },

  // === Hair Follicle Parameters ===
  enable_hair_follicles: { type: 'boolean', label: 'Enable Hair Follicles', description: 'Add hair follicle channels', advanced: true },
  hair_follicle_density_per_cm2: { type: 'number', label: 'Hair Follicle Density', min: 50, max: 200, step: 10, unit: '/cm2', description: '100-150/cm2 on scalp, 50-70 elsewhere', advanced: true },
  hair_follicle_diameter_um: { type: 'number', label: 'Hair Follicle Diameter', min: 50, max: 100, step: 5, unit: 'um', description: '50-100 um', advanced: true },
  hair_follicle_depth_mm: { type: 'number', label: 'Hair Follicle Depth', min: 1, max: 5, step: 0.25, unit: 'mm', description: 'Extends into dermis/hypodermis', advanced: true },

  // === Sweat Gland Parameters ===
  enable_sweat_glands: { type: 'boolean', label: 'Enable Sweat Glands', description: 'Add eccrine sweat gland channels', advanced: true },
  sweat_gland_density_per_cm2: { type: 'number', label: 'Sweat Gland Density', min: 100, max: 200, step: 10, unit: '/cm2', description: '100-200/cm2 (eccrine)', advanced: true },
  sweat_gland_diameter_um: { type: 'number', label: 'Sweat Gland Diameter', min: 20, max: 60, step: 5, unit: 'um', description: 'Coiled tubular structure', advanced: true },
  sweat_gland_depth_mm: { type: 'number', label: 'Sweat Gland Depth', min: 0.5, max: 3, step: 0.25, unit: 'mm', description: 'In dermis', advanced: true },

  // === Sebaceous Glands ===
  enable_sebaceous_glands: { type: 'boolean', label: 'Enable Sebaceous Glands', description: 'Oil-producing glands associated with hair follicles', advanced: true },
  sebaceous_gland_density_per_cm2: { type: 'number', label: 'Sebaceous Gland Density', min: 50, max: 200, step: 10, unit: '/cm2', description: '50-200/cm² on face', advanced: true },
  sebaceous_gland_diameter_um: { type: 'number', label: 'Sebaceous Gland Diameter', min: 50, max: 150, step: 5, unit: 'μm', description: '50-150 μm lobular structure', advanced: true },

  // === Dermal Features ===
  dermal_porosity: { type: 'number', label: 'Dermal Porosity', min: 0.15, max: 0.25, step: 0.01, description: '0.15-0.25 typical', advanced: true },
  pore_diameter_um: { type: 'number', label: 'Pore Diameter', min: 150, max: 300, step: 10, unit: 'um', description: '150-300 um for cell migration', advanced: true },
  pore_gradient: {
    type: 'array',
    label: 'Pore Gradient',
    itemCount: 3,
    itemLabels: ['Epidermis', 'Dermis', 'Hypodermis'],
    min: 0,
    max: 0.6,
    step: 0.05,
    description: 'Porosity by layer',
  },
  pore_interconnectivity: { type: 'number', label: 'Pore Interconnectivity', min: 0, max: 1, step: 0.1, description: '0=isolated pores, 1=fully connected', advanced: true },

  // === Vascular Channels ===
  enable_vascular_channels: { type: 'boolean', label: 'Enable Vascular Channels', description: 'Vertical perfusion channels' },
  vascular_channel_diameter_mm: { type: 'number', label: 'Vascular Channel Diameter', min: 0.1, max: 0.3, step: 0.01, unit: 'mm', description: '100-200 um' },
  vascular_channel_count: { type: 'number', label: 'Vascular Channel Count', min: 0, max: 16, step: 1 },
  vascular_channel_spacing_mm: { type: 'number', label: 'Vascular Channel Spacing', min: 1, max: 5, step: 0.25, unit: 'mm', advanced: true },

  // === Collagen Architecture ===
  enable_collagen_orientation: { type: 'boolean', label: 'Enable Collagen Orientation', description: 'Model collagen fiber direction', advanced: true },
  papillary_collagen_diameter_um: { type: 'number', label: 'Papillary Collagen Diameter', min: 1, max: 5, step: 0.5, unit: 'um', description: 'Fine fibers', advanced: true },
  reticular_collagen_diameter_um: { type: 'number', label: 'Reticular Collagen Diameter', min: 5, max: 20, step: 1, unit: 'um', description: 'Coarse fibers', advanced: true },
  collagen_alignment: { type: 'number', label: 'Collagen Alignment', min: 0, max: 1, step: 0.1, description: '0=random, 1=fully aligned (Langer lines)', advanced: true },

  // === Surface Features ===
  enable_surface_texture: { type: 'boolean', label: 'Enable Surface Texture', description: 'Add surface roughness', advanced: true },
  surface_roughness: { type: 'number', label: 'Surface Roughness', min: 0, max: 0.5, step: 0.05, description: '0-1 scale', advanced: true },

  // === Resolution & Randomization ===
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  randomness: { type: 'number', label: 'Randomness', min: 0, max: 0.5, step: 0.05, description: 'Stochastic variation scale', advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.5, step: 0.05, description: 'Positional variation for organic appearance', advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 32, step: 2, advanced: true },
};

export const SKELETAL_MUSCLE_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  length_mm: { type: 'number', label: 'Length', min: 5, max: 100, step: 5, unit: 'mm', description: 'Scaffold length' },
  width_mm: { type: 'number', label: 'Width', min: 2, max: 50, step: 1, unit: 'mm', description: 'Scaffold width' },
  height_mm: { type: 'number', label: 'Height', min: 2, max: 30, step: 1, unit: 'mm', description: 'Scaffold height' },

  // === Muscle Fiber Parameters ===
  fiber_diameter_um: { type: 'number', label: 'Fiber Diameter', min: 10, max: 100, step: 5, unit: 'um', description: 'Varies 10-100µm by fiber type' },
  fiber_spacing_um: { type: 'number', label: 'Fiber Spacing', min: 50, max: 300, step: 10, unit: 'um', description: 'Center-to-center spacing' },
  fiber_length_mm: { type: 'number', label: 'Fiber Length', min: 5, max: 50, step: 1, unit: 'mm', description: 'Individual fiber length', advanced: true },

  // === Fascicle Parameters ===
  fascicle_diameter_mm: { type: 'number', label: 'Fascicle Diameter', min: 1, max: 10, step: 0.5, unit: 'mm', description: '1-10 mm bundles', advanced: true },
  fascicle_count: { type: 'number', label: 'Fascicle Count', min: 1, max: 16, step: 1, description: 'Number of fascicles' },
  fascicle_spacing_mm: { type: 'number', label: 'Fascicle Spacing', min: 0.5, max: 3, step: 0.25, unit: 'mm', description: 'Gap between fascicles', advanced: true },

  // === Sarcomere Parameters ===
  sarcomere_length_um: { type: 'number', label: 'Sarcomere Length', min: 2.0, max: 3.0, step: 0.1, unit: 'um', description: '2.0-2.5 um at rest', advanced: true },
  sarcomere_resolution: { type: 'number', label: 'Sarcomere Resolution', min: 2, max: 8, step: 1, description: 'Segments per sarcomere', advanced: true },

  // === Fiber Architecture ===
  architecture_type: {
    type: 'enum',
    label: 'Architecture Type',
    options: [
      { value: 'parallel', label: 'Parallel' },
      { value: 'unipennate', label: 'Unipennate' },
      { value: 'bipennate', label: 'Bipennate' },
      { value: 'multipennate', label: 'Multipennate' },
      { value: 'circular', label: 'Circular' },
      { value: 'convergent', label: 'Convergent' },
    ],
    description: 'Fiber arrangement pattern',
  },
  pennation_angle_deg: { type: 'number', label: 'Pennation Angle', min: 0, max: 45, step: 1, unit: 'deg', description: '0-30 deg typical for pennate' },
  fiber_alignment: { type: 'number', label: 'Fiber Alignment', min: 0, max: 1, step: 0.1, description: '0=random, 1=perfectly aligned', advanced: true },
  fiber_curvature: { type: 'number', label: 'Fiber Curvature', min: 0, max: 1, step: 0.1, description: '0=straight, 1=curved', advanced: true },

  // === Connective Tissue - Endomysium (around individual fibers) ===
  endomysium_thickness_um: { type: 'number', label: 'Endomysium Thickness', min: 0.1, max: 2, step: 0.1, unit: 'um', description: 'Very thin layer around each fiber', advanced: true },
  endomysium_porosity: { type: 'number', label: 'Endomysium Porosity', min: 0.1, max: 0.5, step: 0.05, description: 'Porosity of endomysium layer', advanced: true },

  // === Connective Tissue - Perimysium (around fascicles) ===
  perimysium_thickness_um: { type: 'number', label: 'Perimysium Thickness', min: 10, max: 50, step: 5, unit: 'um', description: '10-50 um', advanced: true },
  perimysium_porosity: { type: 'number', label: 'Perimysium Porosity', min: 0.1, max: 0.5, step: 0.05, description: 'Porosity of perimysium layer', advanced: true },
  enable_perimysium: { type: 'boolean', label: 'Enable Perimysium', description: 'Add perimysium around fascicles', advanced: true },

  // === Connective Tissue - Epimysium (outer layer) ===
  epimysium_thickness_um: { type: 'number', label: 'Epimysium Thickness', min: 50, max: 200, step: 10, unit: 'um', description: '50-200 um', advanced: true },
  enable_epimysium: { type: 'boolean', label: 'Enable Epimysium', description: 'Add outer epimysium layer', advanced: true },

  // === Mechanical Properties ===
  contraction_force_uN: { type: 'number', label: 'Contraction Force', min: 100, max: 1000, step: 50, unit: 'uN', description: 'uN per fiber typical', advanced: true },

  // === Vascular Network ===
  enable_vascular_channels: { type: 'boolean', label: 'Enable Vascular Channels', description: 'Add capillary network', advanced: true },
  capillary_diameter_um: { type: 'number', label: 'Capillary Diameter', min: 5, max: 12, step: 1, unit: 'um', description: '~8 um', advanced: true },
  capillary_density_per_mm2: { type: 'number', label: 'Capillary Density', min: 200, max: 500, step: 25, unit: '/mm2', description: '200-500/mm2 typical', advanced: true },
  arteriole_diameter_um: { type: 'number', label: 'Arteriole Diameter', min: 20, max: 100, step: 10, unit: 'um', description: 'Feeding arterioles', advanced: true },
  venule_diameter_um: { type: 'number', label: 'Venule Diameter', min: 30, max: 120, step: 10, unit: 'um', description: 'Draining venules', advanced: true },

  // === Neuromuscular Features ===
  enable_motor_endplates: { type: 'boolean', label: 'Enable Motor Endplates', description: 'Add neuromuscular junctions', advanced: true },
  motor_endplate_density_per_mm2: { type: 'number', label: 'Motor Endplate Density', min: 0.1, max: 1, step: 0.1, unit: '/mm2', description: 'Sparse distribution', advanced: true },

  // === Resolution & Randomization ===
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.3, step: 0.05, description: 'Positional variation', advanced: true },
  diameter_variance: { type: 'number', label: 'Diameter Variance', min: 0, max: 0.3, step: 0.05, description: 'Fiber diameter variation', advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 4, max: 16, step: 2, advanced: true },
};

export const CORNEA_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  diameter_mm: { type: 'number', label: 'Diameter', min: 10, max: 12, step: 0.1, unit: 'mm', description: '10-12 mm typical adult' },
  radius_of_curvature_mm: { type: 'number', label: 'Radius of Curvature', min: 7.5, max: 8.0, step: 0.1, unit: 'mm', description: '7.5-8.0 mm anterior surface' },

  // === Total Thickness ===
  total_thickness_um: { type: 'number', label: 'Total Thickness', min: 500, max: 560, step: 10, unit: 'um', description: '500-560 um central, thicker peripherally' },

  // === Layer Thicknesses ===
  epithelium_thickness_um: { type: 'number', label: 'Epithelium Thickness', min: 45, max: 55, step: 1, unit: 'um', description: '50-52 um, 5-6 cell layers', advanced: true },
  bowmans_layer_thickness_um: { type: 'number', label: 'Bowmans Layer Thickness', min: 8, max: 14, step: 1, unit: 'um', description: '8-14 um', advanced: true },
  stroma_thickness_um: { type: 'number', label: 'Stroma Thickness', min: 400, max: 550, step: 10, unit: 'um', description: '~90% of corneal thickness', advanced: true },
  descemet_thickness_um: { type: 'number', label: 'Descemet Thickness', min: 3, max: 15, step: 1, unit: 'um', description: '3-10 um, thickens with age', advanced: true },
  endothelium_thickness_um: { type: 'number', label: 'Endothelium Thickness', min: 4, max: 6, step: 0.5, unit: 'um', description: '4-6 um', advanced: true },

  // === Stromal Architecture ===
  num_lamellae: { type: 'number', label: 'Number of Lamellae', min: 200, max: 500, step: 25, description: '200-500 lamellae in human cornea' },
  lamella_thickness_um: { type: 'number', label: 'Lamella Thickness', min: 1.5, max: 2.5, step: 0.1, unit: 'um', description: '1.5-2.5 um each', advanced: true },
  lamella_angle_variation_deg: { type: 'number', label: 'Lamella Angle Variation', min: 60, max: 120, step: 10, unit: 'deg', description: 'Orthogonal arrangement', advanced: true },

  // === Collagen Fibril Parameters ===
  collagen_fibril_diameter_nm: { type: 'number', label: 'Collagen Fibril Diameter', min: 25, max: 35, step: 1, unit: 'nm', description: '~31 nm uniform diameter', advanced: true },
  collagen_fibril_spacing_nm: { type: 'number', label: 'Collagen Fibril Spacing', min: 35, max: 70, step: 5, unit: 'nm', description: '~62 nm center-to-center', advanced: true },

  // === Keratocyte Distribution ===
  keratocyte_density_per_mm3: { type: 'number', label: 'Keratocyte Density', min: 15000, max: 30000, step: 1000, unit: '/mm3', description: '20,000-25,000/mm3', advanced: true },
  enable_keratocyte_markers: { type: 'boolean', label: 'Enable Keratocyte Markers', description: 'Add keratocyte position markers', advanced: true },

  // === Nerve Plexus ===
  enable_nerve_plexus: { type: 'boolean', label: 'Enable Nerve Plexus', description: 'Add nerve fiber channels', advanced: true },
  subbasal_nerve_density_per_mm2: { type: 'number', label: 'Subbasal Nerve Density', min: 5000, max: 6500, step: 100, unit: '/mm2', description: '5000-6000/mm2', advanced: true },
  stromal_nerve_bundle_count: { type: 'number', label: 'Stromal Nerve Bundle Count', min: 60, max: 80, step: 5, description: '60-80 bundles', advanced: true },

  // === Optical Properties ===
  transparency_factor: { type: 'number', label: 'Transparency Factor', min: 0, max: 1, step: 0.1, description: 'Affects pore/feature density', advanced: true },
  refractive_index: { type: 'number', label: 'Refractive Index', min: 1.35, max: 1.40, step: 0.01, description: 'For reference (1.376)', advanced: true },

  // === Limbal Transition ===
  enable_limbal_zone: { type: 'boolean', label: 'Enable Limbal Zone', description: 'Add limbal transition zone', advanced: true },
  limbal_width_mm: { type: 'number', label: 'Limbal Width', min: 1, max: 2, step: 0.1, unit: 'mm', description: '1-2 mm transition zone', advanced: true },

  // === Surface Curvature ===
  asphericity_q: { type: 'number', label: 'Asphericity Q', min: -0.5, max: 0, step: 0.02, description: 'Prolate ellipsoid (Q = -0.26 typical)', advanced: true },
  posterior_radius_mm: { type: 'number', label: 'Posterior Radius', min: 6.2, max: 6.8, step: 0.1, unit: 'mm', description: '6.2-6.8 mm posterior surface', advanced: true },

  // === Resolution & Randomization ===
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  thickness_variance: { type: 'number', label: 'Thickness Variance', min: 0, max: 0.15, step: 0.01, description: '5% thickness variation', advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 12, max: 48, step: 4, description: 'High resolution for smooth dome', advanced: true },
};

export const ADIPOSE_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  diameter_mm: { type: 'number', label: 'Diameter', min: 5, max: 50, step: 1, unit: 'mm', description: 'Overall scaffold diameter' },
  height_mm: { type: 'number', label: 'Height', min: 2, max: 30, step: 1, unit: 'mm', description: 'Scaffold height' },

  // === Adipocyte Parameters ===
  adipocyte_diameter_um: { type: 'number', label: 'Adipocyte Diameter', min: 50, max: 200, step: 10, unit: 'um', description: '~90 um lean, ~120 um obese' },
  adipocyte_size_variance: { type: 'number', label: 'Adipocyte Size Variance', min: 0, max: 0.3, step: 0.05, description: 'Size variability', advanced: true },
  cell_density_per_mL: { type: 'number', label: 'Cell Density', min: 20e6, max: 60e6, step: 5e6, unit: '/mL', description: '20-60 million cells/mL', advanced: true },

  // === Lobule Structure ===
  lobule_size_mm: { type: 'number', label: 'Lobule Size', min: 1, max: 3, step: 0.25, unit: 'mm', description: '1-3 mm typical lobule diameter', advanced: true },
  enable_lobules: { type: 'boolean', label: 'Enable Lobules', description: 'Organize into lobule structure', advanced: true },
  lobules_per_cm2: { type: 'number', label: 'Lobules per cm2', min: 4, max: 8, step: 1, unit: '/cm2', description: '~4-8 lobules per cm2', advanced: true },

  // === Septum/Connective Tissue ===
  septum_thickness_um: { type: 'number', label: 'Septum Thickness', min: 30, max: 100, step: 10, unit: 'um', description: '30-100 um interlobular septa' },
  septum_porosity: { type: 'number', label: 'Septum Porosity', min: 0.2, max: 0.5, step: 0.05, description: 'Loose connective tissue', advanced: true },

  // === Vascular Network ===
  enable_vascular_channels: { type: 'boolean', label: 'Enable Vascular Channels', description: 'Add perfusion channels' },
  capillary_diameter_um: { type: 'number', label: 'Capillary Diameter', min: 5, max: 10, step: 1, unit: 'um', description: '5-10 um typical', advanced: true },
  capillary_density_per_mm2: { type: 'number', label: 'Capillary Density', min: 200, max: 600, step: 50, unit: '/mm2', description: 'High vascular density', advanced: true },
  vascular_channel_diameter_mm: { type: 'number', label: 'Vascular Channel Diameter', min: 0.2, max: 0.6, step: 0.05, unit: 'mm', description: 'Larger perfusion channels' },
  vascular_channel_spacing_mm: { type: 'number', label: 'Vascular Channel Spacing', min: 1, max: 5, step: 0.5, unit: 'mm' },

  // === Arteriole/Venule ===
  arteriole_diameter_um: { type: 'number', label: 'Arteriole Diameter', min: 20, max: 50, step: 5, unit: 'um', description: '20-50 um', advanced: true },
  venule_diameter_um: { type: 'number', label: 'Venule Diameter', min: 30, max: 60, step: 5, unit: 'um', description: '30-60 um', advanced: true },

  // === Stromal Vascular Fraction ===
  enable_svf_channels: { type: 'boolean', label: 'Enable SVF Channels', description: 'Space for SVF cells', advanced: true },
  svf_channel_diameter_um: { type: 'number', label: 'SVF Channel Diameter', min: 30, max: 100, step: 10, unit: 'um', advanced: true },

  // === Pore Architecture ===
  porosity: { type: 'number', label: 'Porosity', min: 0.6, max: 0.9, step: 0.05, description: 'High porosity for cell infiltration' },
  pore_size_um: { type: 'number', label: 'Pore Size', min: 150, max: 300, step: 25, unit: 'um', description: '150-300 um for adipogenic differentiation', advanced: true },
  pore_interconnectivity: { type: 'number', label: 'Pore Interconnectivity', min: 0.5, max: 1, step: 0.1, description: '0-1 scale', advanced: true },

  // === ECM Features ===
  enable_ecm_fibers: { type: 'boolean', label: 'Enable ECM Fibers', description: 'Add ECM fiber structure', advanced: true },
  collagen_fiber_density: { type: 'number', label: 'Collagen Fiber Density', min: 0.1, max: 0.5, step: 0.05, description: 'Lower than other tissues', advanced: true },
  elastin_fiber_density: { type: 'number', label: 'Elastin Fiber Density', min: 0.05, max: 0.2, step: 0.05, description: 'Elastin content', advanced: true },

  // === Basement Membrane ===
  enable_basement_membrane: { type: 'boolean', label: 'Enable Basement Membrane', description: '~100 nm around adipocytes', advanced: true },
  basement_membrane_thickness_um: { type: 'number', label: 'Basement Membrane Thickness', min: 0.05, max: 0.2, step: 0.01, unit: 'um', description: '~100 nm', advanced: true },

  // === Surface Features ===
  enable_surface_texture: { type: 'boolean', label: 'Enable Surface Texture', description: 'Add surface roughness', advanced: true },
  surface_roughness: { type: 'number', label: 'Surface Roughness', min: 0, max: 0.5, step: 0.05, description: '0-1 scale', advanced: true },

  // === Mechanical Properties ===
  target_stiffness_kPa: { type: 'number', label: 'Target Stiffness', min: 1, max: 4, step: 0.5, unit: 'kPa', description: '1-4 kPa for adipose tissue', advanced: true },

  // === Resolution & Randomization ===
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, advanced: true },
  position_noise: { type: 'number', label: 'Position Noise', min: 0, max: 0.5, step: 0.05, description: 'Positional variation for organic look', advanced: true },
  resolution: { type: 'number', label: 'Resolution', min: 4, max: 16, step: 2, advanced: true },
};

// Combined export for soft tissue scaffold types
export const SOFT_TISSUE_PARAMETER_META: Partial<Record<ScaffoldType, Record<string, ParamMeta>>> = {
  [ScaffoldType.MULTILAYER_SKIN]: MULTILAYER_SKIN_META,
  [ScaffoldType.SKELETAL_MUSCLE]: SKELETAL_MUSCLE_META,
  [ScaffoldType.CORNEA]: CORNEA_META,
  [ScaffoldType.ADIPOSE]: ADIPOSE_META,
};
