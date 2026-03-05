/**
 * Generation Results, Validation, Chat Interface, and Type Guards
 */

import { ScaffoldType } from './base';
import type { ScaffoldParams } from './index';

// Forward declare param types for type guards (they'll be imported via index)
import type { VascularNetworkParams, PorousDiscParams, PrimitiveParams } from './legacy';
import type {
  TrabecularBoneParams, OsteochondralParams, ArticularCartilageParams, MeniscusParams,
  TendonLigamentParams, IntervertebralDiscParams, HaversianBoneParams
} from './skeletal';
import type {
  HepaticLobuleParams, CardiacPatchParams, KidneyTubuleParams, LungAlveoliParams,
  PancreaticIsletParams, LiverSinusoidParams
} from './organ';
import type { MultilayerSkinParams, SkeletalMuscleParams, CorneaParams, AdiposeParams } from './soft-tissue';
import type {
  BloodVesselParams, NerveConduitParams, SpinalCordParams, BladderParams, TracheaParams,
  TubularConduitParams
} from './tubular';
import type { DentinPulpParams, EarAuricleParams, NasalSeptumParams } from './dental';
import type {
  LatticeParams, GyroidParams, SchwarzPParams, OctetTrussParams, VoronoiParams, HoneycombParams
} from './lattice';
import type { OrganOnChipParams, GradientScaffoldParams, PerfusableNetworkParams } from './microfluidic';

// ============================================================================
// Generation Results
// ============================================================================

export interface MeshData {
  vertices: number[];
  indices: number[];
  normals: number[];
}

export interface BoundingBox {
  min: [number, number, number];
  max: [number, number, number];
}

export interface GenerationResult {
  mesh_data: MeshData;
  stl_base64?: string;
  triangle_count: number;
  bounding_box: BoundingBox;
  volume_mm3: number;
  generation_time_ms: number;
}

// ============================================================================
// Validation Results
// ============================================================================

export interface ValidationCheck {
  passed: boolean;
  message?: string;
}

export interface MinWallThicknessCheck extends ValidationCheck {
  measured_mm: number;
  required_mm: number;
}

export interface OverhangCheck extends ValidationCheck {
  max_angle_deg: number;
}

export interface DimensionsCheck extends ValidationCheck {
  bounding_box: BoundingBox;
}

export interface ValidationChecks {
  watertight: ValidationCheck;
  min_wall_thickness: MinWallThicknessCheck;
  overhang: OverhangCheck;
  dimensions: DimensionsCheck;
}

export interface ValidationResult {
  is_valid: boolean;
  is_printable: boolean;
  checks: ValidationChecks;
  warnings: string[];
  errors: string[];
}

// ============================================================================
// Chat Interface
// ============================================================================

export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  content: string;
  timestamp: string;
  params_change?: Partial<ScaffoldParams>;
  suggestions?: string[];
}

// ============================================================================
// Type Guards (existing + new)
// ============================================================================

export function isVascularNetworkParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.VASCULAR_NETWORK } & VascularNetworkParams {
  return params.type === ScaffoldType.VASCULAR_NETWORK;
}

export function isPorousDiscParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.POROUS_DISC } & PorousDiscParams {
  return params.type === ScaffoldType.POROUS_DISC;
}

export function isTubularConduitParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.TUBULAR_CONDUIT } & TubularConduitParams {
  return params.type === ScaffoldType.TUBULAR_CONDUIT;
}

export function isLatticeParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.LATTICE } & LatticeParams {
  return params.type === ScaffoldType.LATTICE;
}

export function isPrimitiveParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.PRIMITIVE } & PrimitiveParams {
  return params.type === ScaffoldType.PRIMITIVE;
}

// Skeletal type guards
export function isTrabecularBoneParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.TRABECULAR_BONE } & TrabecularBoneParams {
  return params.type === ScaffoldType.TRABECULAR_BONE;
}

export function isOsteochondralParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.OSTEOCHONDRAL } & OsteochondralParams {
  return params.type === ScaffoldType.OSTEOCHONDRAL;
}

export function isArticularCartilageParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.ARTICULAR_CARTILAGE } & ArticularCartilageParams {
  return params.type === ScaffoldType.ARTICULAR_CARTILAGE;
}

export function isMeniscusParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.MENISCUS } & MeniscusParams {
  return params.type === ScaffoldType.MENISCUS;
}

export function isTendonLigamentParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.TENDON_LIGAMENT } & TendonLigamentParams {
  return params.type === ScaffoldType.TENDON_LIGAMENT;
}

export function isIntervertebralDiscParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.INTERVERTEBRAL_DISC } & IntervertebralDiscParams {
  return params.type === ScaffoldType.INTERVERTEBRAL_DISC;
}

export function isHaversianBoneParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.HAVERSIAN_BONE } & HaversianBoneParams {
  return params.type === ScaffoldType.HAVERSIAN_BONE;
}

// Organ type guards
export function isHepaticLobuleParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.HEPATIC_LOBULE } & HepaticLobuleParams {
  return params.type === ScaffoldType.HEPATIC_LOBULE;
}

export function isCardiacPatchParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.CARDIAC_PATCH } & CardiacPatchParams {
  return params.type === ScaffoldType.CARDIAC_PATCH;
}

export function isKidneyTubuleParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.KIDNEY_TUBULE } & KidneyTubuleParams {
  return params.type === ScaffoldType.KIDNEY_TUBULE;
}

export function isLungAlveoliParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.LUNG_ALVEOLI } & LungAlveoliParams {
  return params.type === ScaffoldType.LUNG_ALVEOLI;
}

export function isPancreaticIsletParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.PANCREATIC_ISLET } & PancreaticIsletParams {
  return params.type === ScaffoldType.PANCREATIC_ISLET;
}

export function isLiverSinusoidParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.LIVER_SINUSOID } & LiverSinusoidParams {
  return params.type === ScaffoldType.LIVER_SINUSOID;
}

// Soft tissue type guards
export function isMultilayerSkinParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.MULTILAYER_SKIN } & MultilayerSkinParams {
  return params.type === ScaffoldType.MULTILAYER_SKIN;
}

export function isSkeletalMuscleParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.SKELETAL_MUSCLE } & SkeletalMuscleParams {
  return params.type === ScaffoldType.SKELETAL_MUSCLE;
}

export function isCorneaParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.CORNEA } & CorneaParams {
  return params.type === ScaffoldType.CORNEA;
}

export function isAdiposeParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.ADIPOSE } & AdiposeParams {
  return params.type === ScaffoldType.ADIPOSE;
}

// Tubular type guards
export function isBloodVesselParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.BLOOD_VESSEL } & BloodVesselParams {
  return params.type === ScaffoldType.BLOOD_VESSEL;
}

export function isNerveConduitParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.NERVE_CONDUIT } & NerveConduitParams {
  return params.type === ScaffoldType.NERVE_CONDUIT;
}

export function isSpinalCordParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.SPINAL_CORD } & SpinalCordParams {
  return params.type === ScaffoldType.SPINAL_CORD;
}

export function isBladderParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.BLADDER } & BladderParams {
  return params.type === ScaffoldType.BLADDER;
}

export function isTracheaParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.TRACHEA } & TracheaParams {
  return params.type === ScaffoldType.TRACHEA;
}

// Dental type guards
export function isDentinPulpParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.DENTIN_PULP } & DentinPulpParams {
  return params.type === ScaffoldType.DENTIN_PULP;
}

export function isEarAuricleParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.EAR_AURICLE } & EarAuricleParams {
  return params.type === ScaffoldType.EAR_AURICLE;
}

export function isNasalSeptumParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.NASAL_SEPTUM } & NasalSeptumParams {
  return params.type === ScaffoldType.NASAL_SEPTUM;
}

// Advanced lattice type guards
export function isGyroidParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.GYROID } & GyroidParams {
  return params.type === ScaffoldType.GYROID;
}

export function isSchwarzPParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.SCHWARZ_P } & SchwarzPParams {
  return params.type === ScaffoldType.SCHWARZ_P;
}

export function isOctetTrussParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.OCTET_TRUSS } & OctetTrussParams {
  return params.type === ScaffoldType.OCTET_TRUSS;
}

export function isVoronoiParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.VORONOI } & VoronoiParams {
  return params.type === ScaffoldType.VORONOI;
}

export function isHoneycombParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.HONEYCOMB } & HoneycombParams {
  return params.type === ScaffoldType.HONEYCOMB;
}

// Microfluidic type guards
export function isOrganOnChipParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.ORGAN_ON_CHIP } & OrganOnChipParams {
  return params.type === ScaffoldType.ORGAN_ON_CHIP;
}

export function isGradientScaffoldParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.GRADIENT_SCAFFOLD } & GradientScaffoldParams {
  return params.type === ScaffoldType.GRADIENT_SCAFFOLD;
}

export function isPerfusableNetworkParams(
  params: ScaffoldParams
): params is { type: ScaffoldType.PERFUSABLE_NETWORK } & PerfusableNetworkParams {
  return params.type === ScaffoldType.PERFUSABLE_NETWORK;
}
