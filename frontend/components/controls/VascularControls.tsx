'use client';

import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

interface VascularControlsProps {
  params: Record<string, any>;
  onChange: (key: string, value: any) => void;
}

export function VascularControls({ params, onChange }: VascularControlsProps) {
  return (
    <div className="space-y-4">
      {/* Inlets */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Inlets</Label>
          <span className="text-sm text-slate-500">{params.inlets || 4}</span>
        </div>
        <Slider
          value={[params.inlets || 4]}
          onValueChange={([v]) => onChange('inlets', v)}
          min={1}
          max={25}
          step={1}
        />
      </div>

      {/* Levels */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Branching Levels</Label>
          <span className="text-sm text-slate-500">{params.levels || 3}</span>
        </div>
        <Slider
          value={[params.levels || 3]}
          onValueChange={([v]) => onChange('levels', v)}
          min={0}
          max={8}
          step={1}
        />
      </div>

      {/* Splits */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Splits per Branch</Label>
          <span className="text-sm text-slate-500">{params.splits || 2}</span>
        </div>
        <Slider
          value={[params.splits || 2]}
          onValueChange={([v]) => onChange('splits', v)}
          min={1}
          max={6}
          step={1}
        />
      </div>

      {/* Ratio (Murray's Law) */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>
            Radius Ratio
            <span className="text-xs text-slate-500 ml-1">(Murray&apos;s law: 0.79)</span>
          </Label>
          <span className="text-sm text-slate-500">{(params.ratio || 0.79).toFixed(2)}</span>
        </div>
        <Slider
          value={[params.ratio || 0.79]}
          onValueChange={([v]) => onChange('ratio', v)}
          min={0.5}
          max={0.95}
          step={0.01}
        />
      </div>

      {/* Spread */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Spread</Label>
          <span className="text-sm text-slate-500">{(params.spread || 0.5).toFixed(2)}</span>
        </div>
        <Slider
          value={[params.spread || 0.5]}
          onValueChange={([v]) => onChange('spread', v)}
          min={0.1}
          max={0.8}
          step={0.05}
        />
      </div>

      {/* Curvature */}
      <div className="space-y-2">
        <div className="flex justify-between">
          <Label>Curvature</Label>
          <span className="text-sm text-slate-500">{(params.curvature || 0.3).toFixed(2)}</span>
        </div>
        <Slider
          value={[params.curvature || 0.3]}
          onValueChange={([v]) => onChange('curvature', v)}
          min={0}
          max={1}
          step={0.05}
        />
      </div>

      {/* Toggles */}
      <div className="flex items-center justify-between">
        <Label>Deterministic Pattern</Label>
        <Switch
          checked={params.deterministic ?? true}
          onCheckedChange={(v) => onChange('deterministic', v)}
        />
      </div>

      <div className="flex items-center justify-between">
        <Label>Tips Down</Label>
        <Switch
          checked={params.tips_down ?? false}
          onCheckedChange={(v) => onChange('tips_down', v)}
        />
      </div>
    </div>
  );
}
