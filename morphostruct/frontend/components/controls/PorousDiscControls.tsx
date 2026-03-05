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

interface PorousDiscControlsProps {
  params: Record<string, any>;
  onChange: (key: string, value: any) => void;
}

export function PorousDiscControls({ params, onChange }: PorousDiscControlsProps) {
  return (
    <div className="space-y-4">
      {/* Diameter */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Diameter (mm)</Label>
          <span className="text-sm text-slate-500">{params.diameter_mm || 10}</span>
        </div>
        <Slider
          value={[params.diameter_mm || 10]}
          onValueChange={([v]) => onChange('diameter_mm', v)}
          min={5}
          max={50}
          step={1}
        />
      </div>

      {/* Height */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Height (mm)</Label>
          <span className="text-sm text-slate-500">{params.height_mm || 2}</span>
        </div>
        <Slider
          value={[params.height_mm || 2]}
          onValueChange={([v]) => onChange('height_mm', v)}
          min={0.5}
          max={10}
          step={0.5}
        />
      </div>

      {/* Pore Diameter */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Pore Diameter (μm)</Label>
          <span className="text-sm text-slate-500">{params.pore_diameter_um || 300}</span>
        </div>
        <Slider
          value={[params.pore_diameter_um || 300]}
          onValueChange={([v]) => onChange('pore_diameter_um', v)}
          min={50}
          max={1000}
          step={50}
        />
      </div>

      {/* Pore Spacing */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Pore Spacing (μm)</Label>
          <span className="text-sm text-slate-500">{params.pore_spacing_um || 600}</span>
        </div>
        <Slider
          value={[params.pore_spacing_um || 600]}
          onValueChange={([v]) => onChange('pore_spacing_um', v)}
          min={100}
          max={2000}
          step={100}
        />
      </div>

      {/* Pore Pattern */}
      <div className="space-y-2">
        <Label>Pore Pattern</Label>
        <Select
          value={params.pore_pattern || 'hexagonal'}
          onValueChange={(v) => onChange('pore_pattern', v)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="hexagonal">Hexagonal</SelectItem>
            <SelectItem value="square">Square</SelectItem>
            <SelectItem value="random">Random</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
