/**
 * Base type definitions for MorphoStruct scaffold generation
 * Synchronized with backend Python types
 */

// ============================================================================
// Enums
// ============================================================================

export enum ScaffoldType {
  // Existing
  VASCULAR_NETWORK = 'vascular_network',
  POROUS_DISC = 'porous_disc',
  TUBULAR_CONDUIT = 'tubular_conduit',
  LATTICE = 'lattice',
  PRIMITIVE = 'primitive',

  // Skeletal (7)
  TRABECULAR_BONE = 'trabecular_bone',
  OSTEOCHONDRAL = 'osteochondral',
  ARTICULAR_CARTILAGE = 'articular_cartilage',
  MENISCUS = 'meniscus',
  TENDON_LIGAMENT = 'tendon_ligament',
  INTERVERTEBRAL_DISC = 'intervertebral_disc',
  HAVERSIAN_BONE = 'haversian_bone',

  // Organ (6)
  HEPATIC_LOBULE = 'hepatic_lobule',
  CARDIAC_PATCH = 'cardiac_patch',
  KIDNEY_TUBULE = 'kidney_tubule',
  LUNG_ALVEOLI = 'lung_alveoli',
  PANCREATIC_ISLET = 'pancreatic_islet',
  LIVER_SINUSOID = 'liver_sinusoid',

  // Soft Tissue (4)
  MULTILAYER_SKIN = 'multilayer_skin',
  SKELETAL_MUSCLE = 'skeletal_muscle',
  CORNEA = 'cornea',
  ADIPOSE = 'adipose',

  // Tubular (6)
  BLOOD_VESSEL = 'blood_vessel',
  NERVE_CONDUIT = 'nerve_conduit',
  SPINAL_CORD = 'spinal_cord',
  BLADDER = 'bladder',
  TRACHEA = 'trachea',
  VASCULAR_PERFUSION_DISH = 'vascular_perfusion_dish',

  // Dental (3)
  DENTIN_PULP = 'dentin_pulp',
  EAR_AURICLE = 'ear_auricle',
  NASAL_SEPTUM = 'nasal_septum',

  // Advanced Lattice (5)
  GYROID = 'gyroid',
  SCHWARZ_P = 'schwarz_p',
  OCTET_TRUSS = 'octet_truss',
  VORONOI = 'voronoi',
  HONEYCOMB = 'honeycomb',

  // Microfluidic (3)
  ORGAN_ON_CHIP = 'organ_on_chip',
  GRADIENT_SCAFFOLD = 'gradient_scaffold',
  PERFUSABLE_NETWORK = 'perfusable_network',

  // Advanced Vascular (2) - AOG-powered
  SPACE_COLONIZATION = 'space_colonization',
  BIFURCATING_TREE = 'bifurcating_tree',
}

export enum PorePattern {
  HEXAGONAL = 'hexagonal',
  GRID = 'grid',
}

export enum InnerTexture {
  SMOOTH = 'smooth',
  GROOVED = 'grooved',
  POROUS = 'porous',
}

export enum UnitCell {
  CUBIC = 'cubic',
  BCC = 'bcc',
}

export enum PrimitiveShape {
  CYLINDER = 'cylinder',
  SPHERE = 'sphere',
  BOX = 'box',
  CONE = 'cone',

  // Geometric (8 new)
  TORUS = 'torus',
  CAPSULE = 'capsule',
  PYRAMID = 'pyramid',
  WEDGE = 'wedge',
  PRISM = 'prism',
  TUBE = 'tube',
  ELLIPSOID = 'ellipsoid',
  HEMISPHERE = 'hemisphere',

  // Architectural (7 new)
  FILLET = 'fillet',
  CHAMFER = 'chamfer',
  SLOT = 'slot',
  COUNTERBORE = 'counterbore',
  COUNTERSINK = 'countersink',
  BOSS = 'boss',
  RIB = 'rib',

  // Organic/Bio (8 new)
  BRANCH = 'branch',
  BIFURCATION = 'bifurcation',
  PORE = 'pore',
  CHANNEL = 'channel',
  FIBER = 'fiber',
  MEMBRANE = 'membrane',
  LATTICE_CELL = 'lattice_cell',
  PORE_ARRAY = 'pore_array',
}

export enum ModificationOperation {
  HOLE = 'hole',
  SHELL = 'shell',
  FILLET_EDGES = 'fillet_edges',
  CHAMFER_EDGES = 'chamfer_edges',
  TAPER = 'taper',
  TWIST = 'twist',
}

export enum GradientType {
  LINEAR = 'linear',
  EXPONENTIAL = 'exponential',
}

export enum MuscleArchitecture {
  PARALLEL = 'parallel',
  UNIPENNATE = 'unipennate',
  BIPENNATE = 'bipennate',
  MULTIPENNATE = 'multipennate',
  CIRCULAR = 'circular',
  CONVERGENT = 'convergent',
}

export enum OsteonPattern {
  HEXAGONAL = 'hexagonal',
  RANDOM = 'random',
}

export enum CurveType {
  SINGLE = 'single',
  S_CURVE = 's_curve',
}

export enum NetworkType {
  ARTERIAL = 'arterial',
  VENOUS = 'venous',
  CAPILLARY = 'capillary',
}

export enum GradientDirection {
  X = 'x',
  Y = 'y',
  Z = 'z',
}

export enum GradientFunctionType {
  LINEAR = 'linear',
  EXPONENTIAL = 'exponential',
  SIGMOID = 'sigmoid',
}

// ============================================================================
// Base Parameters
// ============================================================================

export interface BaseParams {
  /** Mesh resolution (6-32, default 16) */
  resolution: number;
}

// ============================================================================
// Shared Helper Interfaces
// ============================================================================

export interface PrimitiveModification {
  operation: ModificationOperation;
  params: Record<string, number>;
}
