'use client';

import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

interface TubularControlsProps {
  params: Record<string, any>;
  onChange: (key: string, value: any) => void;
}

export function TubularControls({ params, onChange }: TubularControlsProps) {
  return (
    <div className="space-y-4">
      {/* Outer Diameter */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Outer Diameter (mm)</Label>
          <span className="text-sm text-slate-500">{params.outer_diameter_mm || 5}</span>
        </div>
        <Slider
          value={[params.outer_diameter_mm || 5]}
          onValueChange={([v]) => onChange('outer_diameter_mm', v)}
          min={2}
          max={20}
          step={0.5}
        />
      </div>

      {/* Wall Thickness */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Wall Thickness (mm)</Label>
          <span className="text-sm text-slate-500">{params.wall_thickness_mm || 1}</span>
        </div>
        <Slider
          value={[params.wall_thickness_mm || 1]}
          onValueChange={([v]) => onChange('wall_thickness_mm', v)}
          min={0.2}
          max={5}
          step={0.1}
        />
      </div>

      {/* Length */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Length (mm)</Label>
          <span className="text-sm text-slate-500">{params.length_mm || 20}</span>
        </div>
        <Slider
          value={[params.length_mm || 20]}
          onValueChange={([v]) => onChange('length_mm', v)}
          min={5}
          max={50}
          step={1}
        />
      </div>

      {/* Inner Texture */}
      <div className="space-y-2">
        <Label>Inner Texture</Label>
        <Select
          value={params.inner_texture || 'smooth'}
          onValueChange={(v) => onChange('inner_texture', v)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="smooth">Smooth</SelectItem>
            <SelectItem value="grooved">Grooved</SelectItem>
            <SelectItem value="textured">Textured</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
