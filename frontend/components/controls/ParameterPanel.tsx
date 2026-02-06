'use client';

import { ScaffoldType } from '@/lib/types/scaffolds';
import { ScaffoldTypeSelector } from './ScaffoldTypeSelector';
import { VascularControls } from './VascularControls';
import { PorousDiscControls } from './PorousDiscControls';
import { TubularControls } from './TubularControls';
import { LatticeControls } from './LatticeControls';
import { PrimitiveControls } from './PrimitiveControls';
import { DynamicControls } from './DynamicControls';
import { AdvancedVascularControls } from './AdvancedVascularControls';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Play, RotateCcw } from 'lucide-react';

// Scaffold types that have custom control components
const CUSTOM_CONTROL_TYPES = new Set([
  ScaffoldType.VASCULAR_NETWORK,
  ScaffoldType.POROUS_DISC,
  ScaffoldType.TUBULAR_CONDUIT,
  ScaffoldType.LATTICE,
  ScaffoldType.PRIMITIVE,
  ScaffoldType.SPACE_COLONIZATION,
  ScaffoldType.BIFURCATING_TREE,
]);

interface ParameterPanelProps {
  scaffoldType: ScaffoldType;
  onScaffoldTypeChange: (type: ScaffoldType) => void;
  params: Record<string, any>;
  onParamsChange: (params: Record<string, any>) => void;
  onGenerate: () => void;
  onReset: () => void;
  isGenerating?: boolean;
  invert?: boolean;
  onInvertChange?: (invert: boolean) => void;
  previewMode?: boolean;
  onPreviewModeChange?: (previewMode: boolean) => void;
}

export function ParameterPanel({
  scaffoldType,
  onScaffoldTypeChange,
  params,
  onParamsChange,
  onGenerate,
  onReset,
  isGenerating = false,
  invert = false,
  onInvertChange,
  previewMode = true,
  onPreviewModeChange,
}: ParameterPanelProps) {
  const updateParam = (key: string, value: any) => {
    onParamsChange({ ...params, [key]: value });
  };

  return (
    <div className="flex flex-col h-full bg-white dark:bg-slate-800 rounded-lg shadow-sm">
      <div className="p-4 border-b dark:border-slate-700">
        <h2 className="font-semibold text-lg">Scaffold Parameters</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Scaffold Type Selector */}
        <ScaffoldTypeSelector
          value={scaffoldType}
          onChange={onScaffoldTypeChange}
        />

        {/* Type-specific controls - use custom controls for known types, DynamicControls for others */}
        {scaffoldType === ScaffoldType.VASCULAR_NETWORK && (
          <VascularControls params={params} onChange={updateParam} />
        )}
        {scaffoldType === ScaffoldType.POROUS_DISC && (
          <PorousDiscControls params={params} onChange={updateParam} />
        )}
        {scaffoldType === ScaffoldType.TUBULAR_CONDUIT && (
          <TubularControls params={params} onChange={updateParam} />
        )}
        {scaffoldType === ScaffoldType.LATTICE && (
          <LatticeControls params={params} onChange={updateParam} />
        )}
        {scaffoldType === ScaffoldType.PRIMITIVE && (
          <PrimitiveControls params={params} onChange={updateParam} onParamsChange={onParamsChange} />
        )}
        {(scaffoldType === ScaffoldType.SPACE_COLONIZATION || scaffoldType === ScaffoldType.BIFURCATING_TREE) && (
          <AdvancedVascularControls scaffoldType={scaffoldType} params={params} onChange={updateParam} />
        )}
        {/* DynamicControls for all other scaffold types */}
        {!CUSTOM_CONTROL_TYPES.has(scaffoldType) && (
          <DynamicControls
            scaffoldType={scaffoldType}
            params={params}
            onChange={updateParam}
          />
        )}
      </div>

      {/* Toggles and action buttons */}
      <div className="p-4 border-t dark:border-slate-700 space-y-3">
        {/* Preview Mode Toggle */}
        <div className="flex items-center justify-between">
          <div className="flex flex-col">
            <label htmlFor="preview-toggle" className="text-sm font-medium text-slate-700 dark:text-slate-200">
              Fast Preview
            </label>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              Lower resolution, faster generation
            </span>
          </div>
          <Switch
            id="preview-toggle"
            checked={previewMode}
            onCheckedChange={onPreviewModeChange}
            className="data-[state=checked]:bg-blue-500 dark:data-[state=checked]:bg-blue-500"
          />
        </div>

        {/* Invert Geometry Toggle */}
        <div className="flex items-center justify-between">
          <div className="flex flex-col">
            <label htmlFor="invert-toggle" className="text-sm font-medium text-slate-700 dark:text-slate-200">
              Invert Geometry
            </label>
            <span className="text-xs text-slate-500 dark:text-slate-400">
              Swap solid/void
            </span>
          </div>
          <Switch
            id="invert-toggle"
            checked={invert}
            onCheckedChange={onInvertChange}
            className="data-[state=checked]:bg-emerald-500 dark:data-[state=checked]:bg-emerald-500"
          />
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <Button
            onClick={onGenerate}
            disabled={isGenerating}
            className="flex-1"
          >
            <Play className="w-4 h-4 mr-2" />
            {isGenerating ? 'Generating...' : 'Generate'}
          </Button>
          <Button variant="outline" onClick={onReset}>
            <RotateCcw className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}
