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

interface LatticeControlsProps {
  params: Record<string, any>;
  onChange: (key: string, value: any) => void;
}

export function LatticeControls({ params, onChange }: LatticeControlsProps) {
  const boundingBox = params.bounding_box || { x: 10, y: 10, z: 10 };

  const updateBoundingBox = (axis: 'x' | 'y' | 'z', value: number) => {
    onChange('bounding_box', { ...boundingBox, [axis]: value });
  };

  return (
    <div className="space-y-4">
      {/* Bounding Box X */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Width (mm)</Label>
          <span className="text-sm text-slate-500">{boundingBox.x}</span>
        </div>
        <Slider
          value={[boundingBox.x]}
          onValueChange={([v]) => updateBoundingBox('x', v)}
          min={5}
          max={50}
          step={1}
        />
      </div>

      {/* Bounding Box Y */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Depth (mm)</Label>
          <span className="text-sm text-slate-500">{boundingBox.y}</span>
        </div>
        <Slider
          value={[boundingBox.y]}
          onValueChange={([v]) => updateBoundingBox('y', v)}
          min={5}
          max={50}
          step={1}
        />
      </div>

      {/* Bounding Box Z */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Height (mm)</Label>
          <span className="text-sm text-slate-500">{boundingBox.z}</span>
        </div>
        <Slider
          value={[boundingBox.z]}
          onValueChange={([v]) => updateBoundingBox('z', v)}
          min={5}
          max={50}
          step={1}
        />
      </div>

      {/* Unit Cell Type */}
      <div className="space-y-2">
        <Label>Unit Cell Type</Label>
        <Select
          value={params.unit_cell || 'cubic'}
          onValueChange={(v) => onChange('unit_cell', v)}
        >
          <SelectTrigger>
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cubic">Cubic</SelectItem>
            <SelectItem value="bcc">Body-Centered Cubic (BCC)</SelectItem>
            <SelectItem value="fcc">Face-Centered Cubic (FCC)</SelectItem>
            <SelectItem value="gyroid">Gyroid</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {/* Cell Size */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Cell Size (mm)</Label>
          <span className="text-sm text-slate-500">{params.cell_size_mm || 2}</span>
        </div>
        <Slider
          value={[params.cell_size_mm || 2]}
          onValueChange={([v]) => onChange('cell_size_mm', v)}
          min={0.5}
          max={5}
          step={0.1}
        />
      </div>

      {/* Strut Diameter */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Strut Diameter (mm)</Label>
          <span className="text-sm text-slate-500">{(params.strut_diameter_mm || 0.4).toFixed(1)}</span>
        </div>
        <Slider
          value={[params.strut_diameter_mm || 0.4]}
          onValueChange={([v]) => onChange('strut_diameter_mm', v)}
          min={0.1}
          max={2}
          step={0.1}
        />
      </div>
    </div>
  );
}
