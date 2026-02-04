'use client';

import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  PRIMITIVE_SHAPE_OPTIONS,
  getPrimitiveDimensions,
  getDefaultDimensions,
} from '@/lib/parameterMeta/primitives';
import { NumberParamMeta } from '@/lib/parameterMeta/types';

interface PrimitiveControlsProps {
  params: Record<string, any>;
  onChange: (key: string, value: any) => void;
  onParamsChange?: (params: Record<string, any>) => void;
}

// Group shapes by category
const SHAPE_CATEGORIES = ['Basic', 'Geometric', 'Architectural', 'Organic'] as const;

function getShapesByCategory(category: string) {
  return PRIMITIVE_SHAPE_OPTIONS.filter((opt) => opt.category === category);
}

/**
 * Renders a slider control for numeric dimension parameters
 */
function DimensionSlider({
  paramKey,
  meta,
  value,
  onChange,
}: {
  paramKey: string;
  meta: NumberParamMeta;
  value: number;
  onChange: (key: string, value: number) => void;
}) {
  const displayValue = typeof value === 'number' ? value : meta.min;
  const formattedValue = meta.step < 1 ? displayValue.toFixed(2) : displayValue.toString();

  return (
    <div className="space-y-2">
      <div className="flex justify-between">
        <Label className="text-sm">
          {meta.label}
          {meta.unit && <span className="text-xs text-slate-500 ml-1">({meta.unit})</span>}
        </Label>
        <span className="text-sm text-slate-500">{formattedValue}</span>
      </div>
      <Slider
        value={[displayValue]}
        onValueChange={([v]) => onChange(paramKey, v)}
        min={meta.min}
        max={meta.max}
        step={meta.step}
      />
    </div>
  );
}

/**
 * PrimitiveControls component
 * Renders shape dropdown with categories and dynamic dimension controls
 */
export function PrimitiveControls({ params, onChange, onParamsChange }: PrimitiveControlsProps) {
  const currentShape = params.shape || 'cylinder';
  const dimensions = params.dimensions || getDefaultDimensions(currentShape);
  const dimensionMeta = getPrimitiveDimensions(currentShape);

  const handleShapeChange = (newShape: string) => {
    // Update shape and dimensions atomically to avoid race condition
    if (onParamsChange) {
      onParamsChange({
        ...params,
        shape: newShape,
        dimensions: getDefaultDimensions(newShape),
      });
    } else {
      // Fallback: update both (may have race condition)
      onChange('shape', newShape);
      onChange('dimensions', getDefaultDimensions(newShape));
    }
  };

  const handleDimensionChange = (key: string, value: number) => {
    onChange('dimensions', { ...dimensions, [key]: value });
  };

  return (
    <div className="space-y-4">
      {/* Shape Dropdown with Categories */}
      <div className="space-y-2">
        <Label className="text-sm">Shape</Label>
        <Select value={currentShape} onValueChange={handleShapeChange}>
          <SelectTrigger>
            <SelectValue placeholder="Select a shape" />
          </SelectTrigger>
          <SelectContent>
            {SHAPE_CATEGORIES.map((category) => (
              <SelectGroup key={category}>
                <SelectLabel className="text-xs font-semibold text-slate-500">
                  {category}
                </SelectLabel>
                {getShapesByCategory(category).map((shape) => (
                  <SelectItem key={shape.value} value={shape.value}>
                    {shape.label}
                  </SelectItem>
                ))}
              </SelectGroup>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Dynamic Dimension Controls */}
      {dimensionMeta && (
        <div className="space-y-4 pt-2 border-t dark:border-slate-700">
          <Label className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
            Dimensions
          </Label>
          {Object.entries(dimensionMeta).map(([key, meta]) => (
            <DimensionSlider
              key={key}
              paramKey={key}
              meta={meta}
              value={dimensions[key] ?? meta.min}
              onChange={handleDimensionChange}
            />
          ))}
        </div>
      )}
    </div>
  );
}
