import { ParamMeta, NumberParamMeta } from './types';

// ============================================================================
// Primitive Dimension Schemas
// ============================================================================

// Each primitive has specific dimension parameters

// --- BASIC (4 existing) ---
export const CYLINDER_DIMS: Record<string, NumberParamMeta> = {
  radius_mm: { type: 'number', label: 'Radius', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
  height_mm: { type: 'number', label: 'Height', min: 0.5, max: 100, step: 0.5, unit: 'mm' },
};

export const SPHERE_DIMS: Record<string, NumberParamMeta> = {
  radius_mm: { type: 'number', label: 'Radius', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
};

export const BOX_DIMS: Record<string, NumberParamMeta> = {
  x_mm: { type: 'number', label: 'X Size', min: 0.5, max: 100, step: 0.5, unit: 'mm' },
  y_mm: { type: 'number', label: 'Y Size', min: 0.5, max: 100, step: 0.5, unit: 'mm' },
  z_mm: { type: 'number', label: 'Z Size', min: 0.5, max: 100, step: 0.5, unit: 'mm' },
};

export const CONE_DIMS: Record<string, NumberParamMeta> = {
  bottom_radius_mm: { type: 'number', label: 'Bottom Radius', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
  top_radius_mm: { type: 'number', label: 'Top Radius', min: 0, max: 50, step: 0.5, unit: 'mm' },
  height_mm: { type: 'number', label: 'Height', min: 0.5, max: 100, step: 0.5, unit: 'mm' },
};

// --- GEOMETRIC (8 new) ---
export const TORUS_DIMS: Record<string, NumberParamMeta> = {
  major_radius_mm: { type: 'number', label: 'Major Radius', min: 1, max: 50, step: 0.5, unit: 'mm' },
  minor_radius_mm: { type: 'number', label: 'Minor Radius', min: 0.1, max: 10, step: 0.1, unit: 'mm' },
  arc_degrees: { type: 'number', label: 'Arc', min: 0, max: 360, step: 15, unit: '°' },
};

export const CAPSULE_DIMS: Record<string, NumberParamMeta> = {
  radius_mm: { type: 'number', label: 'Radius', min: 0.5, max: 25, step: 0.5, unit: 'mm' },
  length_mm: { type: 'number', label: 'Length', min: 1, max: 100, step: 1, unit: 'mm' },
};

export const PYRAMID_DIMS: Record<string, NumberParamMeta> = {
  base_x_mm: { type: 'number', label: 'Base X', min: 1, max: 50, step: 0.5, unit: 'mm' },
  base_y_mm: { type: 'number', label: 'Base Y', min: 1, max: 50, step: 0.5, unit: 'mm' },
  height_mm: { type: 'number', label: 'Height', min: 1, max: 100, step: 0.5, unit: 'mm' },
};

export const WEDGE_DIMS: Record<string, NumberParamMeta> = {
  length_mm: { type: 'number', label: 'Length', min: 1, max: 100, step: 1, unit: 'mm' },
  width_mm: { type: 'number', label: 'Width', min: 1, max: 50, step: 0.5, unit: 'mm' },
  height_mm: { type: 'number', label: 'Height', min: 1, max: 50, step: 0.5, unit: 'mm' },
  angle_degrees: { type: 'number', label: 'Angle', min: 10, max: 80, step: 5, unit: '°' },
};

export const PRISM_DIMS: Record<string, NumberParamMeta> = {
  sides: { type: 'number', label: 'Sides', min: 3, max: 12, step: 1 },
  radius_mm: { type: 'number', label: 'Radius', min: 1, max: 50, step: 0.5, unit: 'mm' },
  height_mm: { type: 'number', label: 'Height', min: 1, max: 100, step: 1, unit: 'mm' },
};

export const TUBE_DIMS: Record<string, NumberParamMeta> = {
  outer_radius_mm: { type: 'number', label: 'Outer Radius', min: 1, max: 50, step: 0.5, unit: 'mm' },
  inner_radius_mm: { type: 'number', label: 'Inner Radius', min: 0.5, max: 45, step: 0.5, unit: 'mm' },
  length_mm: { type: 'number', label: 'Length', min: 1, max: 100, step: 1, unit: 'mm' },
};

export const ELLIPSOID_DIMS: Record<string, NumberParamMeta> = {
  radius_x_mm: { type: 'number', label: 'Radius X', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
  radius_y_mm: { type: 'number', label: 'Radius Y', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
  radius_z_mm: { type: 'number', label: 'Radius Z', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
};

export const HEMISPHERE_DIMS: Record<string, NumberParamMeta> = {
  radius_mm: { type: 'number', label: 'Radius', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
};

// --- ARCHITECTURAL (7 new) ---
export const FILLET_DIMS: Record<string, NumberParamMeta> = {
  radius_mm: { type: 'number', label: 'Radius', min: 0.5, max: 20, step: 0.5, unit: 'mm' },
  length_mm: { type: 'number', label: 'Length', min: 1, max: 100, step: 1, unit: 'mm' },
};

export const CHAMFER_DIMS: Record<string, NumberParamMeta> = {
  size_mm: { type: 'number', label: 'Size', min: 0.5, max: 20, step: 0.5, unit: 'mm' },
  length_mm: { type: 'number', label: 'Length', min: 1, max: 100, step: 1, unit: 'mm' },
};

export const SLOT_DIMS: Record<string, NumberParamMeta> = {
  width_mm: { type: 'number', label: 'Width', min: 1, max: 30, step: 0.5, unit: 'mm' },
  length_mm: { type: 'number', label: 'Length', min: 2, max: 100, step: 1, unit: 'mm' },
  depth_mm: { type: 'number', label: 'Depth', min: 0.5, max: 50, step: 0.5, unit: 'mm' },
};

export const COUNTERBORE_DIMS: Record<string, NumberParamMeta> = {
  hole_diameter_mm: { type: 'number', label: 'Hole Diameter', min: 1, max: 20, step: 0.5, unit: 'mm' },
  bore_diameter_mm: { type: 'number', label: 'Bore Diameter', min: 2, max: 40, step: 0.5, unit: 'mm' },
  bore_depth_mm: { type: 'number', label: 'Bore Depth', min: 1, max: 30, step: 0.5, unit: 'mm' },
  total_depth_mm: { type: 'number', label: 'Total Depth', min: 2, max: 50, step: 0.5, unit: 'mm' },
};

export const COUNTERSINK_DIMS: Record<string, NumberParamMeta> = {
  hole_diameter_mm: { type: 'number', label: 'Hole Diameter', min: 1, max: 20, step: 0.5, unit: 'mm' },
  sink_diameter_mm: { type: 'number', label: 'Sink Diameter', min: 2, max: 40, step: 0.5, unit: 'mm' },
  sink_angle_degrees: { type: 'number', label: 'Sink Angle', min: 60, max: 120, step: 10, unit: '°' },
  total_depth_mm: { type: 'number', label: 'Total Depth', min: 2, max: 50, step: 0.5, unit: 'mm' },
};

export const BOSS_DIMS: Record<string, NumberParamMeta> = {
  diameter_mm: { type: 'number', label: 'Diameter', min: 2, max: 40, step: 0.5, unit: 'mm' },
  height_mm: { type: 'number', label: 'Height', min: 1, max: 50, step: 0.5, unit: 'mm' },
  fillet_radius_mm: { type: 'number', label: 'Fillet Radius', min: 0, max: 10, step: 0.5, unit: 'mm' },
};

export const RIB_DIMS: Record<string, NumberParamMeta> = {
  height_mm: { type: 'number', label: 'Height', min: 1, max: 50, step: 0.5, unit: 'mm' },
  thickness_mm: { type: 'number', label: 'Thickness', min: 0.5, max: 10, step: 0.5, unit: 'mm' },
  length_mm: { type: 'number', label: 'Length', min: 5, max: 100, step: 1, unit: 'mm' },
};

// --- ORGANIC/BIO (8 new) ---
export const BRANCH_DIMS: Record<string, NumberParamMeta> = {
  start_radius_mm: { type: 'number', label: 'Start Radius', min: 0.5, max: 10, step: 0.25, unit: 'mm' },
  end_radius_mm: { type: 'number', label: 'End Radius', min: 0.1, max: 8, step: 0.25, unit: 'mm' },
  length_mm: { type: 'number', label: 'Length', min: 2, max: 50, step: 1, unit: 'mm' },
};

export const BIFURCATION_DIMS: Record<string, NumberParamMeta> = {
  parent_radius_mm: { type: 'number', label: 'Parent Radius', min: 0.5, max: 10, step: 0.25, unit: 'mm' },
  child_radius_mm: { type: 'number', label: 'Child Radius', min: 0.2, max: 8, step: 0.25, unit: 'mm' },
  angle_degrees: { type: 'number', label: 'Branch Angle', min: 15, max: 90, step: 5, unit: '°' },
  length_mm: { type: 'number', label: 'Branch Length', min: 2, max: 30, step: 1, unit: 'mm' },
};

export const PORE_DIMS: Record<string, NumberParamMeta> = {
  diameter_mm: { type: 'number', label: 'Diameter', min: 0.1, max: 5, step: 0.1, unit: 'mm' },
  depth_mm: { type: 'number', label: 'Depth', min: 0.5, max: 20, step: 0.5, unit: 'mm' },
};

export const CHANNEL_DIMS: Record<string, NumberParamMeta> = {
  diameter_mm: { type: 'number', label: 'Diameter', min: 0.1, max: 5, step: 0.1, unit: 'mm' },
  length_mm: { type: 'number', label: 'Length', min: 1, max: 50, step: 1, unit: 'mm' },
};

export const FIBER_DIMS: Record<string, NumberParamMeta> = {
  diameter_mm: { type: 'number', label: 'Diameter', min: 0.05, max: 2, step: 0.05, unit: 'mm' },
  length_mm: { type: 'number', label: 'Length', min: 5, max: 100, step: 5, unit: 'mm' },
  crimp_amplitude_mm: { type: 'number', label: 'Crimp Amplitude', min: 0, max: 1, step: 0.1, unit: 'mm' },
  crimp_wavelength_mm: { type: 'number', label: 'Crimp Wavelength', min: 0.5, max: 10, step: 0.5, unit: 'mm' },
};

export const MEMBRANE_DIMS: Record<string, NumberParamMeta> = {
  thickness_mm: { type: 'number', label: 'Thickness', min: 0.1, max: 2, step: 0.1, unit: 'mm' },
  radius_mm: { type: 'number', label: 'Radius', min: 2, max: 50, step: 1, unit: 'mm' },
  curvature: { type: 'number', label: 'Curvature', min: 0, max: 1, step: 0.1 },
};

export const LATTICE_CELL_DIMS: Record<string, NumberParamMeta> = {
  cell_size_mm: { type: 'number', label: 'Cell Size', min: 1, max: 20, step: 0.5, unit: 'mm' },
  strut_diameter_mm: { type: 'number', label: 'Strut Diameter', min: 0.2, max: 3, step: 0.1, unit: 'mm' },
};

export const PORE_ARRAY_DIMS: Record<string, NumberParamMeta> = {
  pore_size_mm: { type: 'number', label: 'Pore Size', min: 0.1, max: 3, step: 0.1, unit: 'mm' },
  spacing_mm: { type: 'number', label: 'Spacing', min: 0.5, max: 5, step: 0.25, unit: 'mm' },
  count_x: { type: 'number', label: 'Count X', min: 1, max: 20, step: 1 },
  count_y: { type: 'number', label: 'Count Y', min: 1, max: 20, step: 1 },
  depth_mm: { type: 'number', label: 'Depth', min: 0.5, max: 10, step: 0.5, unit: 'mm' },
};

// ============================================================================
// All Primitive Shapes with Labels (for dropdown options)
// ============================================================================

export const PRIMITIVE_SHAPE_OPTIONS = [
  // Basic
  { value: 'cylinder', label: 'Cylinder', category: 'Basic' },
  { value: 'sphere', label: 'Sphere', category: 'Basic' },
  { value: 'box', label: 'Box', category: 'Basic' },
  { value: 'cone', label: 'Cone', category: 'Basic' },
  // Geometric
  { value: 'torus', label: 'Torus', category: 'Geometric' },
  { value: 'capsule', label: 'Capsule', category: 'Geometric' },
  { value: 'pyramid', label: 'Pyramid', category: 'Geometric' },
  { value: 'wedge', label: 'Wedge', category: 'Geometric' },
  { value: 'prism', label: 'Prism', category: 'Geometric' },
  { value: 'tube', label: 'Tube', category: 'Geometric' },
  { value: 'ellipsoid', label: 'Ellipsoid', category: 'Geometric' },
  { value: 'hemisphere', label: 'Hemisphere', category: 'Geometric' },
  // Architectural
  { value: 'fillet', label: 'Fillet', category: 'Architectural' },
  { value: 'chamfer', label: 'Chamfer', category: 'Architectural' },
  { value: 'slot', label: 'Slot', category: 'Architectural' },
  { value: 'counterbore', label: 'Counterbore', category: 'Architectural' },
  { value: 'countersink', label: 'Countersink', category: 'Architectural' },
  { value: 'boss', label: 'Boss', category: 'Architectural' },
  { value: 'rib', label: 'Rib', category: 'Architectural' },
  // Organic
  { value: 'branch', label: 'Branch', category: 'Organic' },
  { value: 'bifurcation', label: 'Bifurcation', category: 'Organic' },
  { value: 'pore', label: 'Pore', category: 'Organic' },
  { value: 'channel', label: 'Channel', category: 'Organic' },
  { value: 'fiber', label: 'Fiber', category: 'Organic' },
  { value: 'membrane', label: 'Membrane', category: 'Organic' },
  { value: 'lattice_cell', label: 'Lattice Cell', category: 'Organic' },
  { value: 'pore_array', label: 'Pore Array', category: 'Organic' },
];

// ============================================================================
// Dimensions Lookup by Shape
// ============================================================================

export const PRIMITIVE_DIMENSIONS: Record<string, Record<string, NumberParamMeta>> = {
  // Basic
  cylinder: CYLINDER_DIMS,
  sphere: SPHERE_DIMS,
  box: BOX_DIMS,
  cone: CONE_DIMS,
  // Geometric
  torus: TORUS_DIMS,
  capsule: CAPSULE_DIMS,
  pyramid: PYRAMID_DIMS,
  wedge: WEDGE_DIMS,
  prism: PRISM_DIMS,
  tube: TUBE_DIMS,
  ellipsoid: ELLIPSOID_DIMS,
  hemisphere: HEMISPHERE_DIMS,
  // Architectural
  fillet: FILLET_DIMS,
  chamfer: CHAMFER_DIMS,
  slot: SLOT_DIMS,
  counterbore: COUNTERBORE_DIMS,
  countersink: COUNTERSINK_DIMS,
  boss: BOSS_DIMS,
  rib: RIB_DIMS,
  // Organic
  branch: BRANCH_DIMS,
  bifurcation: BIFURCATION_DIMS,
  pore: PORE_DIMS,
  channel: CHANNEL_DIMS,
  fiber: FIBER_DIMS,
  membrane: MEMBRANE_DIMS,
  lattice_cell: LATTICE_CELL_DIMS,
  pore_array: PORE_ARRAY_DIMS,
};

/**
 * Get dimension metadata for a specific primitive shape
 */
export function getPrimitiveDimensions(shape: string): Record<string, NumberParamMeta> | undefined {
  return PRIMITIVE_DIMENSIONS[shape];
}

// ============================================================================
// Default Dimension Values by Shape
// ============================================================================

export const DEFAULT_PRIMITIVE_DIMENSIONS: Record<string, Record<string, number>> = {
  // Basic
  cylinder: { radius_mm: 5, height_mm: 10 },
  sphere: { radius_mm: 5 },
  box: { x_mm: 10, y_mm: 10, z_mm: 10 },
  cone: { bottom_radius_mm: 5, top_radius_mm: 0, height_mm: 10 },
  // Geometric
  torus: { major_radius_mm: 10, minor_radius_mm: 2, arc_degrees: 360 },
  capsule: { radius_mm: 5, length_mm: 15 },
  pyramid: { base_x_mm: 10, base_y_mm: 10, height_mm: 15 },
  wedge: { length_mm: 20, width_mm: 10, height_mm: 10, angle_degrees: 45 },
  prism: { sides: 6, radius_mm: 5, height_mm: 10 },
  tube: { outer_radius_mm: 6, inner_radius_mm: 4, length_mm: 15 },
  ellipsoid: { radius_x_mm: 8, radius_y_mm: 5, radius_z_mm: 5 },
  hemisphere: { radius_mm: 5 },
  // Architectural
  fillet: { radius_mm: 3, length_mm: 20 },
  chamfer: { size_mm: 3, length_mm: 20 },
  slot: { width_mm: 5, length_mm: 20, depth_mm: 5 },
  counterbore: { hole_diameter_mm: 5, bore_diameter_mm: 10, bore_depth_mm: 5, total_depth_mm: 15 },
  countersink: { hole_diameter_mm: 5, sink_diameter_mm: 10, sink_angle_degrees: 90, total_depth_mm: 15 },
  boss: { diameter_mm: 10, height_mm: 8, fillet_radius_mm: 1 },
  rib: { height_mm: 10, thickness_mm: 2, length_mm: 30 },
  // Organic
  branch: { start_radius_mm: 3, end_radius_mm: 1.5, length_mm: 15 },
  bifurcation: { parent_radius_mm: 3, child_radius_mm: 2, angle_degrees: 30, length_mm: 10 },
  pore: { diameter_mm: 1, depth_mm: 5 },
  channel: { diameter_mm: 1, length_mm: 15 },
  fiber: { diameter_mm: 0.2, length_mm: 30, crimp_amplitude_mm: 0.2, crimp_wavelength_mm: 3 },
  membrane: { thickness_mm: 0.5, radius_mm: 15, curvature: 0.3 },
  lattice_cell: { cell_size_mm: 5, strut_diameter_mm: 0.5 },
  pore_array: { pore_size_mm: 0.5, spacing_mm: 1.5, count_x: 5, count_y: 5, depth_mm: 3 },
};

/**
 * Get default dimension values for a specific primitive shape
 */
export function getDefaultDimensions(shape: string): Record<string, number> {
  return DEFAULT_PRIMITIVE_DIMENSIONS[shape] || DEFAULT_PRIMITIVE_DIMENSIONS.cylinder;
}
