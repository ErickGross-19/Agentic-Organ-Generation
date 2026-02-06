/**
 * Parameter metadata for dynamic control generation
 * Defines min/max/step/label/unit for each scaffold parameter
 *
 * This module re-exports all parameter metadata from category-specific files.
 */

import { ScaffoldType } from '../types/scaffolds';
import { ParamMeta } from './types';

// Re-export types
export * from './types';

// Re-export individual metadata objects
export { SKELETAL_PARAMETER_META } from './skeletal';
export { ORGAN_PARAMETER_META } from './organ';
export { SOFT_TISSUE_PARAMETER_META } from './soft-tissue';
export { TUBULAR_PARAMETER_META } from './tubular';
export { DENTAL_PARAMETER_META } from './dental';
export { LATTICE_PARAMETER_META } from './lattice';
export { MICROFLUIDIC_PARAMETER_META } from './microfluidic';
export { LEGACY_PARAMETER_META } from './legacy';
export { PRIMITIVE_SHAPE_OPTIONS, PRIMITIVE_DIMENSIONS, getPrimitiveDimensions } from './primitives';

// Import all category metadata for combined export
import { SKELETAL_PARAMETER_META } from './skeletal';
import { ORGAN_PARAMETER_META } from './organ';
import { SOFT_TISSUE_PARAMETER_META } from './soft-tissue';
import { TUBULAR_PARAMETER_META } from './tubular';
import { DENTAL_PARAMETER_META } from './dental';
import { LATTICE_PARAMETER_META } from './lattice';
import { MICROFLUIDIC_PARAMETER_META } from './microfluidic';
import { LEGACY_PARAMETER_META } from './legacy';

/**
 * Combined parameter metadata by scaffold type
 * This is the main export used by components like DynamicControls
 */
export const PARAMETER_META: Record<ScaffoldType, Record<string, ParamMeta>> = {
  ...LEGACY_PARAMETER_META,
  ...SKELETAL_PARAMETER_META,
  ...ORGAN_PARAMETER_META,
  ...SOFT_TISSUE_PARAMETER_META,
  ...TUBULAR_PARAMETER_META,
  ...DENTAL_PARAMETER_META,
  ...LATTICE_PARAMETER_META,
  ...MICROFLUIDIC_PARAMETER_META,
} as Record<ScaffoldType, Record<string, ParamMeta>>;

/**
 * Get human-readable label for a scaffold type
 */
export function getScaffoldTypeLabel(type: ScaffoldType): string {
  const labels: Record<ScaffoldType, string> = {
    [ScaffoldType.VASCULAR_NETWORK]: 'Vascular Network',
    [ScaffoldType.POROUS_DISC]: 'Porous Disc',
    [ScaffoldType.TUBULAR_CONDUIT]: 'Tubular Conduit',
    [ScaffoldType.LATTICE]: 'Lattice',
    [ScaffoldType.PRIMITIVE]: 'Primitive',
    [ScaffoldType.TRABECULAR_BONE]: 'Trabecular Bone',
    [ScaffoldType.OSTEOCHONDRAL]: 'Osteochondral',
    [ScaffoldType.ARTICULAR_CARTILAGE]: 'Articular Cartilage',
    [ScaffoldType.MENISCUS]: 'Meniscus',
    [ScaffoldType.TENDON_LIGAMENT]: 'Tendon/Ligament',
    [ScaffoldType.INTERVERTEBRAL_DISC]: 'Intervertebral Disc',
    [ScaffoldType.HAVERSIAN_BONE]: 'Haversian Bone',
    [ScaffoldType.HEPATIC_LOBULE]: 'Hepatic Lobule',
    [ScaffoldType.CARDIAC_PATCH]: 'Cardiac Patch',
    [ScaffoldType.KIDNEY_TUBULE]: 'Kidney Tubule',
    [ScaffoldType.LUNG_ALVEOLI]: 'Lung Alveoli',
    [ScaffoldType.PANCREATIC_ISLET]: 'Pancreatic Islet',
    [ScaffoldType.LIVER_SINUSOID]: 'Liver Sinusoid',
    [ScaffoldType.MULTILAYER_SKIN]: 'Multilayer Skin',
    [ScaffoldType.SKELETAL_MUSCLE]: 'Skeletal Muscle',
    [ScaffoldType.CORNEA]: 'Cornea',
    [ScaffoldType.ADIPOSE]: 'Adipose',
    [ScaffoldType.BLOOD_VESSEL]: 'Blood Vessel',
    [ScaffoldType.NERVE_CONDUIT]: 'Nerve Conduit',
    [ScaffoldType.SPINAL_CORD]: 'Spinal Cord',
    [ScaffoldType.BLADDER]: 'Bladder',
    [ScaffoldType.TRACHEA]: 'Trachea',
    [ScaffoldType.VASCULAR_PERFUSION_DISH]: 'Vascular Perfusion Dish',
    [ScaffoldType.DENTIN_PULP]: 'Dentin-Pulp',
    [ScaffoldType.EAR_AURICLE]: 'Ear Auricle',
    [ScaffoldType.NASAL_SEPTUM]: 'Nasal Septum',
    [ScaffoldType.GYROID]: 'Gyroid (TPMS)',
    [ScaffoldType.SCHWARZ_P]: 'Schwarz P (TPMS)',
    [ScaffoldType.OCTET_TRUSS]: 'Octet Truss',
    [ScaffoldType.VORONOI]: 'Voronoi',
    [ScaffoldType.HONEYCOMB]: 'Honeycomb',
    [ScaffoldType.ORGAN_ON_CHIP]: 'Organ-on-Chip',
    [ScaffoldType.GRADIENT_SCAFFOLD]: 'Gradient Scaffold',
    [ScaffoldType.PERFUSABLE_NETWORK]: 'Perfusable Network',
  };
  return labels[type] || type;
}
