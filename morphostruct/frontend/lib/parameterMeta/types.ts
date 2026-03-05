/**
 * Shared types for parameter metadata
 * Defines the structure for dynamic control generation
 */

export interface NumberParamMeta {
  type: 'number';
  label: string;
  min: number;
  max: number;
  step: number;
  unit?: string;
  description?: string;
  advanced?: boolean;
}

export interface BooleanParamMeta {
  type: 'boolean';
  label: string;
  description?: string;
  advanced?: boolean;
}

export interface EnumParamMeta {
  type: 'enum';
  label: string;
  options: { value: string; label: string }[];
  description?: string;
  advanced?: boolean;
}

export interface ObjectParamMeta {
  type: 'object';
  label: string;
  properties: Record<string, NumberParamMeta>;
  description?: string;
  advanced?: boolean;
}

export interface ArrayParamMeta {
  type: 'array';
  label: string;
  itemCount: number;
  itemLabels?: string[];
  min: number;
  max: number;
  step: number;
  description?: string;
  advanced?: boolean;
}

/**
 * Complex array of objects with variable structure.
 * Used for modifications, custom operations, etc.
 */
export interface ComplexArrayParamMeta {
  type: 'complex_array';
  label: string;
  /** Schema for array items - defines the structure of each item */
  itemSchema: {
    /** Discriminator field that determines which params apply */
    discriminator: string;
    /** Options for the discriminator enum */
    discriminatorOptions: { value: string; label: string }[];
    /** Params available for each discriminator value */
    variantParams: Record<string, Record<string, ParamMetaBase>>;
  };
  description?: string;
  advanced?: boolean;
}

/** Base param meta without type (used in variantParams) */
export type ParamMetaBase = Omit<NumberParamMeta, 'type'> & { type: 'number' } |
                           Omit<EnumParamMeta, 'type'> & { type: 'enum' };

export type ParamMeta = NumberParamMeta | BooleanParamMeta | EnumParamMeta | ObjectParamMeta | ArrayParamMeta | ComplexArrayParamMeta;

// Common bounding box metadata used across multiple scaffold types
export const boundingBoxMeta: ObjectParamMeta = {
  type: 'object',
  label: 'Bounding Box (mm)',
  properties: {
    x: { type: 'number', label: 'X', min: 1, max: 100, step: 1, unit: 'mm' },
    y: { type: 'number', label: 'Y', min: 1, max: 100, step: 1, unit: 'mm' },
    z: { type: 'number', label: 'Z', min: 1, max: 100, step: 1, unit: 'mm' },
  },
};
