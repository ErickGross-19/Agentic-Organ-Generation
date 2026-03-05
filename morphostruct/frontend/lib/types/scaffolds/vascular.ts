/**
 * Advanced vascular scaffold types (AOG-powered)
 * Synchronized with backend Python types
 */

import { BaseParams, ScaffoldType } from './base';

// ============================================================================
// Space Colonization
// ============================================================================

export interface InletSpec {
  position: [number, number, number];  // meters
  radius: number;  // meters
  direction: [number, number, number];  // normalized
}

export interface SpaceColonizationParams extends BaseParams {
  type: ScaffoldType.SPACE_COLONIZATION;

  // Inlets
  inlets: InletSpec[];

  // Domain (optional - if not specified, uses default cylinder)
  domain_radius: number;  // meters
  domain_height: number;  // meters

  // Growth parameters
  num_attractors: number;
  influence_radius: number;  // meters
  kill_radius: number;  // meters
  step_size: number;  // meters
  max_iterations: number;

  // Branching
  enable_bifurcation: boolean;
  bifurcation_probability: number;  // 0-1
  min_attractors_for_split: number;
  max_children_per_node: number;

  // Radius
  min_radius: number;  // meters
  max_radius: number;  // meters
  taper_factor: number;  // 0-1

  // Multi-inlet
  multi_inlet_mode: 'blended' | 'partitioned_xy' | 'forest';
  directional_bias: number;  // 0-1
  max_deviation_deg: number;  // degrees

  // Synthesis
  radial_resolution: number;

  // Advanced
  random_seed: number;
}

export const DEFAULT_SPACE_COLONIZATION: SpaceColonizationParams = {
  type: ScaffoldType.SPACE_COLONIZATION,
  resolution: 16,

  // Inlets
  inlets: [{
    position: [0.0, 0.0, 0.001],
    radius: 0.0002,
    direction: [0.0, 0.0, -1.0]
  }],

  // Domain
  domain_radius: 0.005,  // 5mm
  domain_height: 0.002,  // 2mm

  // Growth
  num_attractors: 50000,
  influence_radius: 0.002,  // 2mm
  kill_radius: 0.00025,  // 0.25mm
  step_size: 0.00018,  // 0.18mm
  max_iterations: 300,

  // Branching
  enable_bifurcation: true,
  bifurcation_probability: 0.35,
  min_attractors_for_split: 8,
  max_children_per_node: 2,

  // Radius
  min_radius: 0.00003,  // 30 microns
  max_radius: 0.0002,  // 200 microns
  taper_factor: 0.95,

  // Multi-inlet
  multi_inlet_mode: 'blended',
  directional_bias: 0.35,
  max_deviation_deg: 70.0,

  // Synthesis
  radial_resolution: 12,

  // Advanced
  random_seed: 42,
};

// ============================================================================
// Bifurcating Tree
// ============================================================================

export interface BifurcatingTreeParams extends BaseParams {
  type: ScaffoldType.BIFURCATING_TREE;

  // Root
  root_position: [number, number, number];  // meters
  root_direction: [number, number, number];  // normalized
  root_radius: number;  // meters

  // Structure
  branching_levels: number;
  branches_per_node: number;
  branching_angle_deg: number;

  // Geometry
  segment_length: number;  // meters
  taper_segment_length: boolean;
  length_taper_factor: number;  // 0-1

  // Radius
  radius_mode: 'murray' | 'linear' | 'fixed';
  murray_exponent: number;
  min_terminal_radius: number;  // meters

  // Variation
  add_variation: boolean;
  angle_variation_deg: number;
  length_variation_pct: number;

  // Synthesis
  radial_resolution: number;

  random_seed: number;
}

export const DEFAULT_BIFURCATING_TREE: BifurcatingTreeParams = {
  type: ScaffoldType.BIFURCATING_TREE,
  resolution: 16,

  // Root
  root_position: [0.0, 0.0, 0.001],
  root_direction: [0.0, 0.0, -1.0],
  root_radius: 0.0002,  // 200 microns

  // Structure
  branching_levels: 5,
  branches_per_node: 2,
  branching_angle_deg: 35.0,

  // Geometry
  segment_length: 0.0003,  // 0.3mm
  taper_segment_length: true,
  length_taper_factor: 0.85,

  // Radius
  radius_mode: 'murray',
  murray_exponent: 3.0,
  min_terminal_radius: 0.00003,  // 30 microns

  // Variation
  add_variation: false,
  angle_variation_deg: 10.0,
  length_variation_pct: 15.0,

  // Synthesis
  radial_resolution: 12,

  random_seed: 42,
};
