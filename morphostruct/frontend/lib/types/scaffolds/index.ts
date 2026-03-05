/**
 * Scaffold Types - Barrel Export
 * Re-exports all scaffold types from category files
 */

// Base types and enums
export * from './base';

// Category-specific types and defaults
export * from './skeletal';
export * from './organ';
export * from './soft-tissue';
export * from './tubular';
export * from './dental';
export * from './lattice';
export * from './microfluidic';
export * from './legacy';
export * from './vascular';

// Results, validation, and type guards
export * from './results';

// Import types needed for ScaffoldParams union
import { ScaffoldType } from './base';
import { VascularNetworkParams, PorousDiscParams, PrimitiveParams } from './legacy';
import {
  TrabecularBoneParams, OsteochondralParams, ArticularCartilageParams, MeniscusParams,
  TendonLigamentParams, IntervertebralDiscParams, HaversianBoneParams
} from './skeletal';
import {
  HepaticLobuleParams, CardiacPatchParams, KidneyTubuleParams, LungAlveoliParams,
  PancreaticIsletParams, LiverSinusoidParams
} from './organ';
import { MultilayerSkinParams, SkeletalMuscleParams, CorneaParams, AdiposeParams } from './soft-tissue';
import {
  BloodVesselParams, NerveConduitParams, SpinalCordParams, BladderParams, TracheaParams,
  TubularConduitParams, VascularPerfusionDishParams
} from './tubular';
import { DentinPulpParams, EarAuricleParams, NasalSeptumParams } from './dental';
import {
  LatticeParams, GyroidParams, SchwarzPParams, OctetTrussParams, VoronoiParams, HoneycombParams
} from './lattice';
import { OrganOnChipParams, GradientScaffoldParams, PerfusableNetworkParams } from './microfluidic';
import { SpaceColonizationParams, BifurcatingTreeParams } from './vascular';

// ============================================================================
// Discriminated Union
// ============================================================================

export type ScaffoldParams =
  // Existing
  | ({ type: ScaffoldType.VASCULAR_NETWORK } & VascularNetworkParams)
  | ({ type: ScaffoldType.POROUS_DISC } & PorousDiscParams)
  | ({ type: ScaffoldType.TUBULAR_CONDUIT } & TubularConduitParams)
  | ({ type: ScaffoldType.LATTICE } & LatticeParams)
  | ({ type: ScaffoldType.PRIMITIVE } & PrimitiveParams)
  // Skeletal
  | ({ type: ScaffoldType.TRABECULAR_BONE } & TrabecularBoneParams)
  | ({ type: ScaffoldType.OSTEOCHONDRAL } & OsteochondralParams)
  | ({ type: ScaffoldType.ARTICULAR_CARTILAGE } & ArticularCartilageParams)
  | ({ type: ScaffoldType.MENISCUS } & MeniscusParams)
  | ({ type: ScaffoldType.TENDON_LIGAMENT } & TendonLigamentParams)
  | ({ type: ScaffoldType.INTERVERTEBRAL_DISC } & IntervertebralDiscParams)
  | ({ type: ScaffoldType.HAVERSIAN_BONE } & HaversianBoneParams)
  // Organ
  | ({ type: ScaffoldType.HEPATIC_LOBULE } & HepaticLobuleParams)
  | ({ type: ScaffoldType.CARDIAC_PATCH } & CardiacPatchParams)
  | ({ type: ScaffoldType.KIDNEY_TUBULE } & KidneyTubuleParams)
  | ({ type: ScaffoldType.LUNG_ALVEOLI } & LungAlveoliParams)
  | ({ type: ScaffoldType.PANCREATIC_ISLET } & PancreaticIsletParams)
  | ({ type: ScaffoldType.LIVER_SINUSOID } & LiverSinusoidParams)
  // Soft Tissue
  | ({ type: ScaffoldType.MULTILAYER_SKIN } & MultilayerSkinParams)
  | ({ type: ScaffoldType.SKELETAL_MUSCLE } & SkeletalMuscleParams)
  | ({ type: ScaffoldType.CORNEA } & CorneaParams)
  | ({ type: ScaffoldType.ADIPOSE } & AdiposeParams)
  // Tubular
  | ({ type: ScaffoldType.BLOOD_VESSEL } & BloodVesselParams)
  | ({ type: ScaffoldType.NERVE_CONDUIT } & NerveConduitParams)
  | ({ type: ScaffoldType.SPINAL_CORD } & SpinalCordParams)
  | ({ type: ScaffoldType.BLADDER } & BladderParams)
  | ({ type: ScaffoldType.TRACHEA } & TracheaParams)
  | ({ type: ScaffoldType.VASCULAR_PERFUSION_DISH } & VascularPerfusionDishParams)
  // Dental
  | ({ type: ScaffoldType.DENTIN_PULP } & DentinPulpParams)
  | ({ type: ScaffoldType.EAR_AURICLE } & EarAuricleParams)
  | ({ type: ScaffoldType.NASAL_SEPTUM } & NasalSeptumParams)
  // Advanced Lattice
  | ({ type: ScaffoldType.GYROID } & GyroidParams)
  | ({ type: ScaffoldType.SCHWARZ_P } & SchwarzPParams)
  | ({ type: ScaffoldType.OCTET_TRUSS } & OctetTrussParams)
  | ({ type: ScaffoldType.VORONOI } & VoronoiParams)
  | ({ type: ScaffoldType.HONEYCOMB } & HoneycombParams)
  // Microfluidic
  | ({ type: ScaffoldType.ORGAN_ON_CHIP } & OrganOnChipParams)
  | ({ type: ScaffoldType.GRADIENT_SCAFFOLD } & GradientScaffoldParams)
  | ({ type: ScaffoldType.PERFUSABLE_NETWORK } & PerfusableNetworkParams)
  // Advanced Vascular
  | ({ type: ScaffoldType.SPACE_COLONIZATION } & SpaceColonizationParams)
  | ({ type: ScaffoldType.BIFURCATING_TREE } & BifurcatingTreeParams);
