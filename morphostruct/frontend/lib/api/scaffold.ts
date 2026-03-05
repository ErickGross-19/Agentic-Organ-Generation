import { apiRequest, apiBlobRequest } from './client';
import { ScaffoldType } from '../types/scaffolds';

// Default timeout for scaffold generation (used if preference not set)
// This should match the backend default in config.py
const DEFAULT_GENERATION_TIMEOUT_MS = 60000; // 60 seconds

interface GenerateRequest {
  type: ScaffoldType;
  params: Record<string, any>;
  preview_only?: boolean;
  invert?: boolean;
}

interface MeshData {
  vertices: number[];
  indices: number[];
  normals?: number[];
}

interface ValidationResult {
  is_valid: boolean;
  is_printable: boolean;
  checks: Record<string, any>;
  warnings: string[];
  errors: string[];
}

interface GenerateResponse {
  success: boolean;
  scaffold_id: string;
  mesh: MeshData;
  stl_base64: string;
  stats: {
    triangle_count: number;
    volume_mm3: number;
    generation_time_ms: number;
  };
  validation: ValidationResult;
  inverted?: boolean;
}

interface Preset {
  id: string;
  name: string;
  type: ScaffoldType;
  description: string;
  params: Record<string, any>;
}

interface PresetsResponse {
  presets: Preset[];
}

/**
 * Generate a scaffold with the specified parameters.
 * @param type - The scaffold type
 * @param params - The scaffold parameters
 * @param previewOnly - If true, generate a faster preview
 * @param timeoutSeconds - Optional timeout in seconds (from user preferences). Defaults to 60s.
 * @param invert - If true, invert the geometry (swap solid/void spaces)
 */
export async function generateScaffold(
  type: ScaffoldType,
  params: Record<string, any>,
  previewOnly = false,
  timeoutSeconds?: number,
  invert = false
): Promise<GenerateResponse> {
  const endpoint = previewOnly ? '/api/preview' : '/api/generate';
  // Convert seconds to milliseconds, use default if not provided
  const timeoutMs = timeoutSeconds ? timeoutSeconds * 1000 : DEFAULT_GENERATION_TIMEOUT_MS;
  return apiRequest<GenerateResponse>(endpoint, {
    method: 'POST',
    body: JSON.stringify({ type, params, preview_only: previewOnly, invert }),
  }, timeoutMs);
}

export async function previewScaffold(
  type: ScaffoldType,
  params: Record<string, any>,
  timeoutSeconds?: number,
  invert = false
): Promise<GenerateResponse> {
  return generateScaffold(type, params, true, timeoutSeconds, invert);
}

export async function validateParams(
  type: ScaffoldType,
  params: Record<string, any>
): Promise<{ valid: boolean; errors: string[]; warnings: string[] }> {
  return apiRequest('/api/validate', {
    method: 'POST',
    body: JSON.stringify({ type, params }),
  });
}

/**
 * Export a scaffold as STL file.
 * @param scaffoldId - The scaffold ID from generation
 * @param format - STL format (binary or ascii)
 * @param timeoutSeconds - Optional timeout in seconds. Defaults to 60s.
 */
export async function exportSTL(
  scaffoldId: string,
  format: 'binary' | 'ascii' = 'binary',
  timeoutSeconds?: number
): Promise<Blob> {
  const timeoutMs = timeoutSeconds ? timeoutSeconds * 1000 : DEFAULT_GENERATION_TIMEOUT_MS;
  return apiBlobRequest(`/api/export/${scaffoldId}?format=${format}`, {}, timeoutMs);
}

export async function getPresets(): Promise<PresetsResponse> {
  return apiRequest<PresetsResponse>('/api/presets');
}

export async function getPreset(presetId: string): Promise<Preset> {
  return apiRequest<Preset>(`/api/presets/${presetId}`);
}

// Helper to trigger download
export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
