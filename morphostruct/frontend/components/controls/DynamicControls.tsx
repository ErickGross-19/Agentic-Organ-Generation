'use client';

import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { ScaffoldType } from '@/lib/types/scaffolds';
import {
  PARAMETER_META,
  ParamMeta,
  NumberParamMeta,
  BooleanParamMeta,
  EnumParamMeta,
  ObjectParamMeta,
  ArrayParamMeta,
} from '@/lib/parameterMeta';

interface DynamicControlsProps {
  scaffoldType: ScaffoldType;
  params: Record<string, any>;
  onChange: (key: string, value: any) => void;
}

/**
 * Renders a slider control for numeric parameters
 */
function NumberControl({
  paramKey,
  meta,
  value,
  onChange,
}: {
  paramKey: string;
  meta: NumberParamMeta;
  value: number;
  onChange: (key: string, value: any) => void;
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
      {meta.description && (
        <p className="text-xs text-slate-400">{meta.description}</p>
      )}
    </div>
  );
}

/**
 * Renders a toggle switch for boolean parameters
 */
function BooleanControl({
  paramKey,
  meta,
  value,
  onChange,
}: {
  paramKey: string;
  meta: BooleanParamMeta;
  value: boolean;
  onChange: (key: string, value: any) => void;
}) {
  return (
    <div className="flex items-center justify-between py-1">
      <div>
        <Label className="text-sm">{meta.label}</Label>
        {meta.description && (
          <p className="text-xs text-slate-400">{meta.description}</p>
        )}
      </div>
      <Switch
        checked={value ?? false}
        onCheckedChange={(v) => onChange(paramKey, v)}
      />
    </div>
  );
}

/**
 * Renders a dropdown select for enum parameters
 */
function EnumControl({
  paramKey,
  meta,
  value,
  onChange,
}: {
  paramKey: string;
  meta: EnumParamMeta;
  value: string;
  onChange: (key: string, value: any) => void;
}) {
  return (
    <div className="space-y-2">
      <Label className="text-sm">{meta.label}</Label>
      <Select
        value={value || meta.options[0]?.value}
        onValueChange={(v) => onChange(paramKey, v)}
      >
        <SelectTrigger>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {meta.options.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {meta.description && (
        <p className="text-xs text-slate-400">{meta.description}</p>
      )}
    </div>
  );
}

/**
 * Renders grouped number inputs for object parameters (like bounding_box)
 */
function ObjectControl({
  paramKey,
  meta,
  value,
  onChange,
}: {
  paramKey: string;
  meta: ObjectParamMeta;
  value: Record<string, number>;
  onChange: (key: string, value: any) => void;
}) {
  const objValue = value || {};

  const handlePropertyChange = (prop: string, newVal: number) => {
    onChange(paramKey, { ...objValue, [prop]: newVal });
  };

  return (
    <div className="space-y-3 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
      <Label className="text-sm font-medium">{meta.label}</Label>
      <div className="grid grid-cols-3 gap-3">
        {Object.entries(meta.properties).map(([prop, propMeta]) => (
          <div key={prop} className="space-y-1">
            <Label className="text-xs text-slate-500">{propMeta.label}</Label>
            <Input
              type="number"
              value={objValue[prop] ?? propMeta.min}
              onChange={(e) => handlePropertyChange(prop, parseFloat(e.target.value) || propMeta.min)}
              min={propMeta.min}
              max={propMeta.max}
              step={propMeta.step}
              className="h-8 text-sm"
            />
          </div>
        ))}
      </div>
      {meta.description && (
        <p className="text-xs text-slate-400">{meta.description}</p>
      )}
    </div>
  );
}

/**
 * Renders multiple number inputs for array parameters (like zone_ratios)
 */
function ArrayControl({
  paramKey,
  meta,
  value,
  onChange,
}: {
  paramKey: string;
  meta: ArrayParamMeta;
  value: number[];
  onChange: (key: string, value: any) => void;
}) {
  const arrValue = Array.isArray(value) ? value : Array(meta.itemCount).fill(meta.min);

  const handleItemChange = (index: number, newVal: number) => {
    const newArr = [...arrValue];
    newArr[index] = newVal;
    onChange(paramKey, newArr);
  };

  return (
    <div className="space-y-3 p-3 bg-slate-50 dark:bg-slate-900 rounded-lg">
      <Label className="text-sm font-medium">{meta.label}</Label>
      <div className="space-y-2">
        {Array(meta.itemCount)
          .fill(0)
          .map((_, index) => {
            const itemLabel = meta.itemLabels?.[index] || `Item ${index + 1}`;
            const itemValue = arrValue[index] ?? meta.min;
            const formattedValue = meta.step < 1 ? itemValue.toFixed(2) : itemValue.toString();

            return (
              <div key={index} className="space-y-1">
                <div className="flex justify-between">
                  <Label className="text-xs text-slate-500">{itemLabel}</Label>
                  <span className="text-xs text-slate-500">{formattedValue}</span>
                </div>
                <Slider
                  value={[itemValue]}
                  onValueChange={([v]) => handleItemChange(index, v)}
                  min={meta.min}
                  max={meta.max}
                  step={meta.step}
                />
              </div>
            );
          })}
      </div>
      {meta.description && (
        <p className="text-xs text-slate-400">{meta.description}</p>
      )}
    </div>
  );
}

/**
 * Renders a single parameter control based on its metadata type
 */
function ParameterControl({
  paramKey,
  meta,
  value,
  onChange,
}: {
  paramKey: string;
  meta: ParamMeta;
  value: any;
  onChange: (key: string, value: any) => void;
}) {
  switch (meta.type) {
    case 'number':
      return (
        <NumberControl
          paramKey={paramKey}
          meta={meta}
          value={value}
          onChange={onChange}
        />
      );
    case 'boolean':
      return (
        <BooleanControl
          paramKey={paramKey}
          meta={meta}
          value={value}
          onChange={onChange}
        />
      );
    case 'enum':
      return (
        <EnumControl
          paramKey={paramKey}
          meta={meta}
          value={value}
          onChange={onChange}
        />
      );
    case 'object':
      return (
        <ObjectControl
          paramKey={paramKey}
          meta={meta}
          value={value}
          onChange={onChange}
        />
      );
    case 'array':
      return (
        <ArrayControl
          paramKey={paramKey}
          meta={meta}
          value={value}
          onChange={onChange}
        />
      );
    default:
      return null;
  }
}

/**
 * DynamicControls component
 * Renders parameter controls dynamically based on scaffold type and parameter metadata
 */
export function DynamicControls({ scaffoldType, params, onChange }: DynamicControlsProps) {
  const paramMeta = PARAMETER_META[scaffoldType];

  if (!paramMeta) {
    return (
      <div className="text-sm text-slate-500 italic">
        No parameter controls available for this scaffold type.
      </div>
    );
  }

  // Filter out resolution (handled separately) and sort parameters
  const paramEntries = Object.entries(paramMeta).filter(
    ([key]) => key !== 'resolution'
  );

  return (
    <div className="space-y-4">
      {paramEntries.map(([key, meta]) => (
        <ParameterControl
          key={key}
          paramKey={key}
          meta={meta}
          value={params[key]}
          onChange={onChange}
        />
      ))}
    </div>
  );
}
