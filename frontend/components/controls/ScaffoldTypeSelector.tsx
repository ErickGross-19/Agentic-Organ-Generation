'use client';

import { ScaffoldType } from '@/lib/types/scaffolds';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Label } from '@/components/ui/label';

const SCAFFOLD_TYPES: { value: ScaffoldType; label: string; description: string }[] = [
  // Original (5)
  { value: ScaffoldType.VASCULAR_NETWORK, label: 'Vascular Network', description: 'Branching vascular channel network' },
  { value: ScaffoldType.POROUS_DISC, label: 'Porous Disc', description: 'Flat disc with uniform porosity' },
  { value: ScaffoldType.TUBULAR_CONDUIT, label: 'Tubular Conduit', description: 'Hollow tube for tissue guidance' },
  { value: ScaffoldType.LATTICE, label: 'Lattice', description: 'Basic 3D lattice structure' },
  { value: ScaffoldType.PRIMITIVE, label: 'Primitive', description: 'Simple geometric shape' },

  // Skeletal (7)
  { value: ScaffoldType.TRABECULAR_BONE, label: 'Trabecular Bone', description: 'Cancellous bone architecture' },
  { value: ScaffoldType.OSTEOCHONDRAL, label: 'Osteochondral', description: 'Bone-cartilage interface scaffold' },
  { value: ScaffoldType.ARTICULAR_CARTILAGE, label: 'Articular Cartilage', description: 'Joint surface cartilage structure' },
  { value: ScaffoldType.MENISCUS, label: 'Meniscus', description: 'Fibrocartilaginous knee structure' },
  { value: ScaffoldType.TENDON_LIGAMENT, label: 'Tendon/Ligament', description: 'Aligned fibrous tissue scaffold' },
  { value: ScaffoldType.INTERVERTEBRAL_DISC, label: 'Intervertebral Disc', description: 'Spinal disc with annulus and nucleus' },
  { value: ScaffoldType.HAVERSIAN_BONE, label: 'Haversian Bone', description: 'Compact bone with osteon structure' },

  // Organ (6)
  { value: ScaffoldType.HEPATIC_LOBULE, label: 'Hepatic Lobule', description: 'Liver functional unit structure' },
  { value: ScaffoldType.CARDIAC_PATCH, label: 'Cardiac Patch', description: 'Myocardial tissue scaffold' },
  { value: ScaffoldType.KIDNEY_TUBULE, label: 'Kidney Tubule', description: 'Renal tubular network' },
  { value: ScaffoldType.LUNG_ALVEOLI, label: 'Lung Alveoli', description: 'Alveolar gas exchange structure' },
  { value: ScaffoldType.PANCREATIC_ISLET, label: 'Pancreatic Islet', description: 'Islet of Langerhans scaffold' },
  { value: ScaffoldType.LIVER_SINUSOID, label: 'Liver Sinusoid', description: 'Hepatic sinusoidal network' },

  // Soft Tissue (4)
  { value: ScaffoldType.MULTILAYER_SKIN, label: 'Multilayer Skin', description: 'Epidermis and dermis layers' },
  { value: ScaffoldType.SKELETAL_MUSCLE, label: 'Skeletal Muscle', description: 'Aligned myofiber structure' },
  { value: ScaffoldType.CORNEA, label: 'Cornea', description: 'Transparent lamellar eye tissue' },
  { value: ScaffoldType.ADIPOSE, label: 'Adipose', description: 'Fat tissue scaffold' },

  // Tubular (6)
  { value: ScaffoldType.BLOOD_VESSEL, label: 'Blood Vessel', description: 'Vascular graft with wall layers' },
  { value: ScaffoldType.NERVE_CONDUIT, label: 'Nerve Conduit', description: 'Neural guidance channel' },
  { value: ScaffoldType.SPINAL_CORD, label: 'Spinal Cord', description: 'Central nervous tissue scaffold' },
  { value: ScaffoldType.BLADDER, label: 'Bladder', description: 'Urinary bladder wall scaffold' },
  { value: ScaffoldType.TRACHEA, label: 'Trachea', description: 'Airway cartilage ring structure' },
  { value: ScaffoldType.VASCULAR_PERFUSION_DISH, label: 'Vascular Perfusion Dish', description: 'Collision-aware branching vascular network' },

  // Dental (3)
  { value: ScaffoldType.DENTIN_PULP, label: 'Dentin/Pulp', description: 'Tooth dentin and pulp chamber' },
  { value: ScaffoldType.EAR_AURICLE, label: 'Ear Auricle', description: 'External ear cartilage scaffold' },
  { value: ScaffoldType.NASAL_SEPTUM, label: 'Nasal Septum', description: 'Nasal cartilage partition' },

  // Advanced Lattice (5)
  { value: ScaffoldType.GYROID, label: 'Gyroid', description: 'Triply periodic minimal surface (TPMS)' },
  { value: ScaffoldType.SCHWARZ_P, label: 'Schwarz P', description: 'Primitive TPMS with cubic symmetry' },
  { value: ScaffoldType.OCTET_TRUSS, label: 'Octet Truss', description: 'High-strength lattice framework' },
  { value: ScaffoldType.VORONOI, label: 'Voronoi', description: 'Natural tessellation pattern' },
  { value: ScaffoldType.HONEYCOMB, label: 'Honeycomb', description: 'Hexagonal cell structure' },

  // Microfluidic (3)
  { value: ScaffoldType.ORGAN_ON_CHIP, label: 'Organ-on-Chip', description: 'Microfluidic organ model' },
  { value: ScaffoldType.GRADIENT_SCAFFOLD, label: 'Gradient Scaffold', description: 'Spatially graded properties' },
  { value: ScaffoldType.PERFUSABLE_NETWORK, label: 'Perfusable Network', description: 'Flow-optimized channel system' },
];

interface ScaffoldTypeSelectorProps {
  value: ScaffoldType;
  onChange: (value: ScaffoldType) => void;
}

export function ScaffoldTypeSelector({ value, onChange }: ScaffoldTypeSelectorProps) {
  return (
    <div className="space-y-2">
      <Label>Scaffold Type</Label>
      <Select value={value} onValueChange={(v) => onChange(v as ScaffoldType)}>
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {SCAFFOLD_TYPES.map((type) => (
            <SelectItem key={type.value} value={type.value}>
              <div className="flex flex-col">
                <span>{type.label}</span>
                <span className="text-xs text-slate-500">{type.description}</span>
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
