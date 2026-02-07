import { create } from 'zustand';
import { ScaffoldType } from '../types/scaffolds';

interface MeshData {
  vertices: number[];
  indices: number[];
  normals?: number[];
}

interface ValidationResult {
  is_valid: boolean;
  is_printable: boolean;
  warnings: string[];
  errors: string[];
}

interface ScaffoldStats {
  triangle_count: number;
  volume_mm3: number;
  generation_time_ms: number;

  // Vascular-specific stats (optional - only for vascular types)
  network_nodes?: number;
  network_segments?: number;
  total_length_m?: number;
  min_radius_m?: number;
  max_radius_m?: number;
  terminal_count?: number;
  num_inlets?: number;
  branching_levels?: number;
  branches_per_node?: number;
  scaffold_type?: string;
}

interface ScaffoldStore {
  // Current scaffold type
  scaffoldType: ScaffoldType;
  setScaffoldType: (type: ScaffoldType) => void;

  // Parameters for current scaffold
  params: Record<string, any>;
  setParams: (params: Record<string, any>) => void;
  updateParam: (key: string, value: any) => void;
  resetParams: () => void;

  // Invert geometry toggle (persists across scaffold type changes)
  invert: boolean;
  setInvert: (invert: boolean) => void;

  // Preview mode toggle (faster generation, lower resolution)
  previewMode: boolean;
  setPreviewMode: (previewMode: boolean) => void;

  // Generated mesh data
  meshData: MeshData | null;
  setMeshData: (data: MeshData | null) => void;

  // STL data for export
  stlBase64: string | null;
  setStlBase64: (data: string | null) => void;

  // Generation state
  isGenerating: boolean;
  setIsGenerating: (generating: boolean) => void;
  generationProgress: string;
  setGenerationProgress: (progress: string) => void;

  // Validation results
  validation: ValidationResult | null;
  setValidation: (result: ValidationResult | null) => void;

  // Statistics
  stats: ScaffoldStats | null;
  setStats: (stats: ScaffoldStats | null) => void;

  // Current scaffold ID (for export)
  scaffoldId: string | null;
  setScaffoldId: (id: string | null) => void;
}

const DEFAULT_PARAMS: Record<ScaffoldType, Record<string, any>> = {
  [ScaffoldType.VASCULAR_NETWORK]: {
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
  },
  [ScaffoldType.POROUS_DISC]: {
    diameter_mm: 10,
    height_mm: 2,
    pore_diameter_um: 200,
    pore_spacing_um: 400,
    pore_pattern: 'hexagonal',
    porosity_target: 0.5,
  },
  [ScaffoldType.TUBULAR_CONDUIT]: {
    outer_diameter_mm: 6,
    wall_thickness_mm: 1,
    length_mm: 20,
    inner_texture: 'smooth',
  },
  [ScaffoldType.LATTICE]: {
    bounding_box: { x: 10, y: 10, z: 10 },
    unit_cell: 'cubic',
    cell_size_mm: 2,
    strut_diameter_mm: 0.5,
  },
  [ScaffoldType.PRIMITIVE]: {
    shape: 'cylinder',
    dimensions: { radius_mm: 5, height_mm: 10 },
  },
  // Skeletal
  [ScaffoldType.TRABECULAR_BONE]: {
    porosity: 0.7,
    pore_size_um: 400.0,
    strut_thickness_um: 200.0,
    anisotropy_ratio: 1.5,
    bounding_box: { x: 10.0, y: 10.0, z: 10.0 },
    seed: 42,
  },
  [ScaffoldType.OSTEOCHONDRAL]: {
    bone_depth: 3.0,
    cartilage_depth: 2.0,
    transition_width: 1.0,
    gradient_type: 'linear',
    diameter: 8.0,
    bone_porosity: 0.7,
    cartilage_porosity: 0.85,
  },
  [ScaffoldType.ARTICULAR_CARTILAGE]: {
    total_thickness: 2.0,
    zone_ratios: [0.2, 0.4, 0.4],
    pore_gradient: [0.15, 0.25, 0.35],
    diameter: 8.0,
  },
  [ScaffoldType.MENISCUS]: {
    inner_radius: 12.0,
    outer_radius: 20.0,
    height: 8.0,
    wedge_angle_deg: 20.0,
    zone_count: 3,
    fiber_diameter: 0.2,
    arc_degrees: 300.0,
  },
  [ScaffoldType.TENDON_LIGAMENT]: {
    fiber_diameter: 0.15,
    fiber_spacing: 0.4,
    crimp_amplitude: 0.3,
    crimp_wavelength: 2.0,
    bundle_count: 5,
    length: 20.0,
    width: 5.0,
    thickness: 2.0,
  },
  [ScaffoldType.INTERVERTEBRAL_DISC]: {
    disc_diameter: 40.0,
    disc_height: 10.0,
    af_ring_count: 5,
    np_diameter: 15.0,
    af_layer_angle: 30.0,
    fiber_diameter: 0.15,
    np_porosity: 0.9,
  },
  [ScaffoldType.HAVERSIAN_BONE]: {
    canal_diameter_um: 50.0,
    canal_spacing_um: 250.0,
    cortical_thickness: 5.0,
    osteon_pattern: 'hexagonal',
    bounding_box: { x: 10.0, y: 10.0, z: 15.0 },
  },
  // Organ
  [ScaffoldType.HEPATIC_LOBULE]: {
    num_lobules: 1,
    lobule_radius: 1.5,
    lobule_height: 3.0,
    wall_thickness: 0.1,
    central_vein_radius: 0.15,
    portal_vein_radius: 0.12,
    show_sinusoids: false,
    sinusoid_radius: 0.025,
  },
  [ScaffoldType.CARDIAC_PATCH]: {
    fiber_spacing: 300.0,
    fiber_diameter: 100.0,
    patch_size: { x: 10.0, y: 10.0, z: 1.0 },
    layer_count: 3,
    alignment_angle: 0.0,
  },
  [ScaffoldType.KIDNEY_TUBULE]: {
    tubule_diameter: 100.0,
    wall_thickness: 15.0,
    convolution_amplitude: 500.0,
    convolution_frequency: 3.0,
    length: 10.0,
  },
  [ScaffoldType.LUNG_ALVEOLI]: {
    branch_generations: 3,
    alveoli_diameter: 200.0,
    airway_diameter: 1.0,
    branch_angle: 35.0,
    bounding_box: { x: 10.0, y: 10.0, z: 10.0 },
  },
  [ScaffoldType.PANCREATIC_ISLET]: {
    islet_diameter: 200.0,
    shell_thickness: 50.0,
    pore_size: 20.0,
    cluster_count: 7,
    spacing: 300.0,
  },
  [ScaffoldType.LIVER_SINUSOID]: {
    sinusoid_diameter: 500.0,  // um - scaled up from native 12um for visibility
    sinusoid_length: 2000.0,   // um - shorter for faster generation
    sinusoid_wall_thickness: 50.0,  // um
    fenestration_porosity: 0.0,  // disabled by default for performance
    enable_space_of_disse: false,  // disabled by default for performance
    scaffold_length: 3.0,  // mm
    resolution: 12,
  },
  // Soft Tissue
  [ScaffoldType.MULTILAYER_SKIN]: {
    epidermis_thickness_mm: 0.15,
    dermis_thickness_mm: 1.5,
    hypodermis_thickness_mm: 3.0,
    diameter_mm: 10.0,
    pore_gradient: [0.3, 0.5, 0.7],
    vascular_channel_diameter_mm: 0.2,
    vascular_channel_count: 8,
  },
  [ScaffoldType.SKELETAL_MUSCLE]: {
    fiber_diameter_um: 75.0,
    fiber_spacing_um: 150.0,
    pennation_angle_deg: 15.0,
    fascicle_count: 3,
    architecture_type: 'parallel',
    length_mm: 10.0,
    width_mm: 5.0,
    height_mm: 5.0,
  },
  [ScaffoldType.CORNEA]: {
    diameter_mm: 11.0,
    thickness_mm: 0.55,
    radius_of_curvature_mm: 7.8,
    lamella_count: 5,
  },
  [ScaffoldType.ADIPOSE]: {
    cell_size_um: 150.0,
    wall_thickness_um: 30.0,
    vascular_channel_spacing_mm: 2.0,
    vascular_channel_diameter_mm: 0.3,
    height_mm: 5.0,
    diameter_mm: 10.0,
  },
  // Tubular
  [ScaffoldType.BLOOD_VESSEL]: {
    inner_diameter_mm: 5.0,
    wall_thickness_mm: 1.0,
    layer_ratios: [0.1, 0.5, 0.4],
    length_mm: 30.0,
  },
  [ScaffoldType.NERVE_CONDUIT]: {
    outer_diameter_mm: 4.0,
    channel_count: 7,
    channel_diameter_um: 200.0,
    wall_thickness_mm: 0.5,
    length_mm: 20.0,
  },
  [ScaffoldType.SPINAL_CORD]: {
    cord_diameter_mm: 12.0,
    channel_diameter_um: 300.0,
    channel_count: 24,
    gray_matter_ratio: 0.4,
    length_mm: 50.0,
  },
  [ScaffoldType.BLADDER]: {
    diameter_mm: 70.0,
    wall_thickness_mm: 3.0,
    layer_count: 3,
    dome_curvature: 0.7,
  },
  [ScaffoldType.TRACHEA]: {
    outer_diameter_mm: 20.0,
    ring_thickness_mm: 3.0,
    ring_spacing_mm: 4.0,
    ring_count: 10,
    posterior_gap_angle_deg: 90.0,
  },
  [ScaffoldType.VASCULAR_PERFUSION_DISH]: {
    inlets: 4,
    levels: 2,
    splits: 2,
    spread: 0.35,
    ratio: 0.79,
    cone_angle: 60.0,
    curvature: 0.3,
    bottom_height: 0.06,
    radius_variation: 0.25,
    flip_chance: 0.30,
    z_variation: 0.35,
    angle_variation: 0.40,
    collision_buffer: 0.08,
    even_spread: true,
    deterministic: false,
    tips_down: false,
    outer_radius_mm: 4.875,
    height_mm: 2.0,
    inlet_radius_mm: 0.35,
    seed: 42,
    resolution: 12,
  },
  // Dental
  [ScaffoldType.DENTIN_PULP]: {
    tooth_height: 17.0,
    crown_diameter: 7.0,
    crown_height: 5.0,
    root_length: 12.0,
    root_diameter: 3.0,
    pulp_chamber_size: 0.4,
    dentin_thickness: 2.0,
    resolution: 16,
  },
  [ScaffoldType.EAR_AURICLE]: {
    scale_factor: 1.0,
    thickness: 2.0,
    helix_definition: 0.7,
    antihelix_depth: 0.3,
  },
  [ScaffoldType.NASAL_SEPTUM]: {
    length: 40.0,
    height: 30.0,
    thickness: 3.0,
    curvature_radius: 75.0,
    curve_type: 'single',
  },
  // Advanced Lattice
  [ScaffoldType.GYROID]: {
    bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
    cell_size_mm: 2.0,
    wall_thickness_mm: 0.3,
    isovalue: 0.0,
    samples_per_cell: 20,
  },
  [ScaffoldType.SCHWARZ_P]: {
    bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
    cell_size_mm: 2.0,
    wall_thickness_mm: 0.3,
    isovalue: 0.0,
    samples_per_cell: 20,
  },
  [ScaffoldType.OCTET_TRUSS]: {
    bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
    cell_size_mm: 2.0,
    strut_diameter_mm: 0.3,
  },
  [ScaffoldType.VORONOI]: {
    bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
    cell_count: 30,
    strut_diameter_mm: 0.3,
    margin_factor: 0.2,
  },
  [ScaffoldType.HONEYCOMB]: {
    bounding_box_mm: { x: 10.0, y: 10.0, z: 5.0 },
    cell_size_mm: 2.0,
    wall_thickness_mm: 0.3,
  },
  // Microfluidic
  [ScaffoldType.ORGAN_ON_CHIP]: {
    channel_width_mm: 0.3,
    channel_depth_mm: 0.1,
    chamber_size_mm: { x: 3.0, y: 2.0, z: 0.15 },
    chamber_count: 2,
    inlet_count: 2,
    chip_size_mm: { x: 15.0, y: 10.0, z: 2.0 },
  },
  [ScaffoldType.GRADIENT_SCAFFOLD]: {
    dimensions_mm: { x: 10.0, y: 10.0, z: 10.0 },
    gradient_direction: 'z',
    start_porosity: 0.2,
    end_porosity: 0.8,
    gradient_type: 'linear',
    pore_base_size_mm: 0.5,
    grid_spacing_mm: 1.5,
  },
  [ScaffoldType.PERFUSABLE_NETWORK]: {
    inlet_diameter_mm: 2.0,
    branch_generations: 3,
    murray_ratio: 0.79,
    network_type: 'arterial',
    bounding_box_mm: { x: 10.0, y: 10.0, z: 10.0 },
    branching_angle_deg: 30.0,
  },
  // Advanced Vascular (AOG-powered)
  [ScaffoldType.SPACE_COLONIZATION]: {
    inlets: [{
      position: [0.0, 0.0, 0.001],
      radius: 0.0002,
      direction: [0.0, 0.0, -1.0]
    }],
    domain_radius: 0.005,
    domain_height: 0.002,
    num_attractors: 50000,
    influence_radius: 0.002,
    kill_radius: 0.00025,
    step_size: 0.00018,
    max_iterations: 300,
    enable_bifurcation: true,
    bifurcation_probability: 0.35,
    min_attractors_for_split: 8,
    max_children_per_node: 2,
    min_radius: 0.00003,
    max_radius: 0.0002,
    taper_factor: 0.95,
    multi_inlet_mode: 'blended',
    directional_bias: 0.35,
    max_deviation_deg: 70.0,
    radial_resolution: 12,
    random_seed: 42,
    resolution: 16,
  },
  [ScaffoldType.BIFURCATING_TREE]:{
    root_position: [0.0, 0.0, 0.001],
    root_direction: [0.0, 0.0, -1.0],
    root_radius: 0.0002,
    branching_levels: 5,
    branches_per_node: 2,
    branching_angle_deg: 35.0,
    segment_length: 0.0003,
    taper_segment_length: true,
    length_taper_factor: 0.85,
    radius_mode: 'murray',
    murray_exponent: 3.0,
    min_terminal_radius: 0.00003,
    add_variation: false,
    angle_variation_deg: 10.0,
    length_variation_pct: 15.0,
    radial_resolution: 12,
    random_seed: 42,
    resolution: 16,
  },
};

export const useScaffoldStore = create<ScaffoldStore>((set, get) => ({
  scaffoldType: ScaffoldType.VASCULAR_NETWORK,
  setScaffoldType: (type) => set({
    scaffoldType: type,
    params: DEFAULT_PARAMS[type],
    meshData: null,
    stlBase64: null,
    validation: null,
    stats: null,
    // Note: invert is NOT reset when changing scaffold type (persists)
  }),

  params: DEFAULT_PARAMS[ScaffoldType.VASCULAR_NETWORK],
  setParams: (params) => set({ params }),
  updateParam: (key, value) => set((state) => ({
    params: { ...state.params, [key]: value }
  })),
  resetParams: () => set((state) => ({
    params: DEFAULT_PARAMS[state.scaffoldType]
  })),

  invert: false,
  setInvert: (invert) => set({ invert }),

  previewMode: true,  // Default to preview mode for faster generation
  setPreviewMode: (previewMode) => set({ previewMode }),

  meshData: null,
  setMeshData: (data) => set({ meshData: data }),

  stlBase64: null,
  setStlBase64: (data) => set({ stlBase64: data }),

  isGenerating: false,
  setIsGenerating: (generating) => set({ isGenerating: generating }),
  generationProgress: '',
  setGenerationProgress: (progress) => set({ generationProgress: progress }),

  validation: null,
  setValidation: (result) => set({ validation: result }),

  stats: null,
  setStats: (stats) => set({ stats: stats }),

  scaffoldId: null,
  setScaffoldId: (id) => set({ scaffoldId: id }),
}));
