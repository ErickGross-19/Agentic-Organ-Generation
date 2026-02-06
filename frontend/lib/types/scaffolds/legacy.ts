/**
 * Legacy/Existing Type-Specific Parameters
 */

import { BaseParams, PorePattern, InnerTexture, PrimitiveShape, PrimitiveModification } from './base';

// ============================================================================
// Existing Type-Specific Parameters
// ============================================================================

export interface VascularNetworkParams extends BaseParams {
  inlets: number;
  levels: number;
  splits: number;
  spread: number;
  ratio: number;
  curvature: number;
  tips_down: boolean;
  deterministic: boolean;
  /** Outer ring radius in mm */
  outer_radius: number;
  /** Total height in mm */
  height: number;
  /** Radius of inlet channels in mm */
  inlet_radius: number;
  /** Inner scaffold radius in mm */
  inner_radius?: number;
  /** Active scaffold height in mm */
  scaffold_height?: number;
  /** Random seed for reproducibility */
  seed?: number;

  // Legacy aliases - deprecated, use non-suffixed versions
  /** @deprecated Use outer_radius instead */
  outer_radius_mm?: number;
  /** @deprecated Use height instead */
  height_mm?: number;
  /** @deprecated Use inlet_radius instead */
  inlet_radius_mm?: number;
}

export interface PorousDiscParams extends BaseParams {
  diameter_mm: number;
  height_mm: number;
  pore_diameter_um: number;
  pore_spacing_um: number;
  pore_pattern: PorePattern;
  porosity_target?: number;
}

// TubularConduitParams moved to tubular.ts - re-export for backward compatibility
export type { TubularConduitParams } from './tubular';
export { DEFAULT_TUBULAR_CONDUIT } from './tubular';

export interface PrimitiveParams extends BaseParams {
  shape: PrimitiveShape;
  dimensions: Record<string, number>;
  modifications?: PrimitiveModification[];
}

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULT_RESOLUTION = 16;

export const DEFAULT_VASCULAR_NETWORK: VascularNetworkParams = {
  resolution: DEFAULT_RESOLUTION,
  inlets: 4,
  levels: 3,
  splits: 2,
  spread: 0.5,
  ratio: 0.79,
  curvature: 0.3,
  tips_down: true,
  deterministic: false,
  outer_radius: 10,
  height: 5,
  inlet_radius: 1,
};

export const DEFAULT_POROUS_DISC: PorousDiscParams = {
  resolution: DEFAULT_RESOLUTION,
  diameter_mm: 10,
  height_mm: 2,
  pore_diameter_um: 200,
  pore_spacing_um: 400,
  pore_pattern: PorePattern.HEXAGONAL,
  porosity_target: 0.5,
};

// DEFAULT_TUBULAR_CONDUIT moved to tubular.ts and re-exported above

export const DEFAULT_PRIMITIVE: PrimitiveParams = {
  resolution: DEFAULT_RESOLUTION,
  shape: PrimitiveShape.CYLINDER,
  dimensions: { radius: 5, height: 10 },
};
