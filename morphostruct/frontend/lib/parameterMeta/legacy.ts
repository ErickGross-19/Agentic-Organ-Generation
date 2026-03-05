/**
 * Parameter metadata for legacy scaffold types
 * Includes: vascular_network, porous_disc, tubular_conduit, primitive
 */

import { ScaffoldType } from '../types/scaffolds';
import { ParamMeta, boundingBoxMeta } from './types';

export const VASCULAR_NETWORK_META: Record<string, ParamMeta> = {
  // === Network Structure ===
  inlets: { type: 'number', label: 'Inlets', min: 1, max: 25, step: 1, description: 'Number of inlet points' },
  levels: { type: 'number', label: 'Branching Levels', min: 0, max: 5, step: 1, description: 'Branching depth levels' },
  splits: { type: 'number', label: 'Splits per Branch', min: 2, max: 4, step: 1, description: 'Number of child branches per split' },
  spread: { type: 'number', label: 'Spread', min: 0.0, max: 1.0, step: 0.05, description: 'Horizontal spread factor' },
  ratio: { type: 'number', label: 'Radius Ratio (Murray)', min: 0.5, max: 1.0, step: 0.01, description: "Child/parent radius ratio (Murray's law = 0.79)" },
  curvature: { type: 'number', label: 'Curvature', min: 0.0, max: 1.0, step: 0.05, description: 'Branch curve intensity' },

  // === Behavior ===
  seed: { type: 'number', label: 'Random Seed', min: 0, max: 9999, step: 1, description: 'Seed for reproducibility', advanced: true },
  tips_down: { type: 'boolean', label: 'Tips Down', description: 'Terminal branches curve downward' },
  deterministic: { type: 'boolean', label: 'Deterministic Pattern', description: 'Use grid pattern instead of random' },

  // === Geometry ===
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 32, step: 2, description: 'Cylinder/sphere segments', advanced: true },
  outer_radius: { type: 'number', label: 'Outer Radius', min: 1, max: 20, step: 0.125, unit: 'mm', description: 'Outer ring radius' },
  inner_radius: { type: 'number', label: 'Inner Radius', min: 0.5, max: 15, step: 0.125, unit: 'mm', description: 'Inner scaffold radius', advanced: true },
  height: { type: 'number', label: 'Height', min: 0.5, max: 10, step: 0.1, unit: 'mm', description: 'Total height' },
  scaffold_height: { type: 'number', label: 'Scaffold Height', min: 0.5, max: 10, step: 0.1, unit: 'mm', description: 'Active scaffold height', advanced: true },
  inlet_radius: { type: 'number', label: 'Inlet Radius', min: 0.1, max: 2, step: 0.05, unit: 'mm', description: 'Radius of inlet channels' },
};

export const POROUS_DISC_META: Record<string, ParamMeta> = {
  // === Basic Geometry ===
  diameter_mm: { type: 'number', label: 'Diameter', min: 1, max: 50, step: 0.5, unit: 'mm', description: 'Disc diameter (1-50 mm)' },
  height_mm: { type: 'number', label: 'Height', min: 0.5, max: 10, step: 0.1, unit: 'mm', description: 'Disc height/thickness (0.5-10 mm)' },

  // === Pore Parameters ===
  pore_diameter_um: { type: 'number', label: 'Pore Diameter', min: 50, max: 500, step: 10, unit: 'um', description: 'Pore diameter in microns' },
  pore_spacing_um: { type: 'number', label: 'Pore Spacing', min: 100, max: 1000, step: 25, unit: 'um', description: 'Center-to-center pore spacing' },
  pore_pattern: {
    type: 'enum',
    label: 'Pore Pattern',
    options: [
      { value: 'hexagonal', label: 'Hexagonal' },
      { value: 'grid', label: 'Grid' },
    ],
    description: 'Pore arrangement pattern',
  },

  // === Resolution ===
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 64, step: 2, description: 'Number of segments for cylindrical surfaces', advanced: true },
};

// TUBULAR_CONDUIT_META moved to tubular.ts
export { TUBULAR_CONDUIT_META } from './tubular';

export const PRIMITIVE_META: Record<string, ParamMeta> = {
  // === Shape Selection ===
  shape: {
    type: 'enum',
    label: 'Shape',
    options: [
      { value: 'cylinder', label: 'Cylinder' },
      { value: 'sphere', label: 'Sphere' },
      { value: 'box', label: 'Box' },
      { value: 'cone', label: 'Cone' },
    ],
    description: 'Base primitive shape',
  },

  // === Dimensions (shape-dependent) ===
  dimensions: {
    type: 'object',
    label: 'Dimensions',
    properties: {
      // Cylinder/Cone
      radius_mm: { type: 'number', label: 'Radius', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
      height_mm: { type: 'number', label: 'Height', min: 0.5, max: 100, step: 0.5, unit: 'mm' },
      // Cone specific
      bottom_radius_mm: { type: 'number', label: 'Bottom Radius', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
      top_radius_mm: { type: 'number', label: 'Top Radius', min: 0, max: 50, step: 0.5, unit: 'mm' },
      // Box
      x_mm: { type: 'number', label: 'X Size', min: 0.5, max: 100, step: 0.5, unit: 'mm' },
      y_mm: { type: 'number', label: 'Y Size', min: 0.5, max: 100, step: 0.5, unit: 'mm' },
      z_mm: { type: 'number', label: 'Z Size', min: 0.5, max: 100, step: 0.5, unit: 'mm' },
    },
    description: 'Shape-dependent dimension parameters',
  },

  // === Modifications (CSG operations applied in sequence) ===
  modifications: {
    type: 'complex_array',
    label: 'Modifications',
    itemSchema: {
      discriminator: 'operation',
      discriminatorOptions: [
        { value: 'hole', label: 'Hole (Through-Hole)' },
        { value: 'shell', label: 'Shell (Hollow Out)' },
      ],
      variantParams: {
        hole: {
          diameter_mm: { type: 'number', label: 'Hole Diameter', min: 0.1, max: 50, step: 0.1, unit: 'mm' },
          axis: {
            type: 'enum',
            label: 'Axis',
            options: [
              { value: 'x', label: 'X Axis' },
              { value: 'y', label: 'Y Axis' },
              { value: 'z', label: 'Z Axis' },
            ],
          },
        },
        shell: {
          wall_thickness_mm: { type: 'number', label: 'Wall Thickness', min: 0.1, max: 10, step: 0.1, unit: 'mm' },
        },
      },
    },
    description: 'CSG operations to apply (hole creates through-hole, shell hollows out)',
    advanced: true,
  },

  // === Resolution ===
  resolution: { type: 'number', label: 'Resolution', min: 8, max: 64, step: 4, description: 'Mesh resolution for curved surfaces', advanced: true },
};

// Combined export for legacy scaffold types (TUBULAR_CONDUIT moved to TUBULAR_PARAMETER_META)
export const LEGACY_PARAMETER_META: Partial<Record<ScaffoldType, Record<string, ParamMeta>>> = {
  [ScaffoldType.VASCULAR_NETWORK]: VASCULAR_NETWORK_META,
  [ScaffoldType.POROUS_DISC]: POROUS_DISC_META,
  [ScaffoldType.PRIMITIVE]: PRIMITIVE_META,
};
