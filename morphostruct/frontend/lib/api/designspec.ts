import { apiRequest } from './client';

export interface DesignSpecEvent {
  event_type: string;
  data: Record<string, any>;
  message: string;
  timestamp: number;
}

export interface PatchProposal {
  patch_id: string;
  description: string;
  diff: Record<string, any>;
  [key: string]: any;
}

export interface DesignSpecMessageResponse {
  messages: string[];
  patches: PatchProposal[];
  questions: Record<string, any>[];
  run_request: Record<string, any> | null;
  spec: Record<string, any> | null;
  status: string;
  events: DesignSpecEvent[];
}

export interface DesignSpecProjectResponse {
  success: boolean;
  project_dir: string;
  spec: Record<string, any>;
}

export interface PatchActionResponse {
  success?: boolean;
  patch_id: string;
  spec?: Record<string, any>;
  reason?: string;
  events: DesignSpecEvent[];
}

export interface PipelineRunResponse {
  success: boolean;
  result: Record<string, any> | null;
  status: string;
  events: DesignSpecEvent[];
}

export interface DesignSpecStatus {
  status: string;
  project_dir: string | null;
  llm_initialized: boolean;
}

export async function initDesignSpecLLM(
  provider: string,
  apiKey?: string,
  model?: string,
  apiBase?: string,
): Promise<{ success: boolean }> {
  return apiRequest('/api/designspec/llm/init', {
    method: 'POST',
    body: JSON.stringify({
      provider,
      api_key: apiKey,
      model,
      api_base: apiBase,
    }),
  });
}

export async function createDesignSpecProject(
  projectName: string,
  templateSpec?: Record<string, any>,
): Promise<DesignSpecProjectResponse> {
  return apiRequest('/api/designspec/projects', {
    method: 'POST',
    body: JSON.stringify({
      project_name: projectName,
      template_spec: templateSpec,
    }),
  });
}

export async function loadDesignSpecProject(
  projectDir: string,
): Promise<DesignSpecProjectResponse> {
  return apiRequest('/api/designspec/projects/load', {
    method: 'POST',
    body: JSON.stringify({ project_dir: projectDir }),
  });
}

export async function sendDesignSpecMessage(
  message: string,
): Promise<DesignSpecMessageResponse> {
  return apiRequest('/api/designspec/message', {
    method: 'POST',
    body: JSON.stringify({ message }),
  }, 60000);
}

export async function approveDesignSpecPatch(
  patchId: string,
): Promise<PatchActionResponse> {
  return apiRequest(`/api/designspec/patches/${patchId}/approve`, {
    method: 'POST',
  });
}

export async function rejectDesignSpecPatch(
  patchId: string,
  reason?: string,
): Promise<PatchActionResponse> {
  return apiRequest(`/api/designspec/patches/${patchId}/reject`, {
    method: 'POST',
    body: JSON.stringify({ reason: reason || '' }),
  });
}

export async function runDesignSpecPipeline(): Promise<PipelineRunResponse> {
  return apiRequest('/api/designspec/run', {
    method: 'POST',
  }, 300000);
}

export async function runDesignSpecPipelineUntil(
  stage: string,
): Promise<PipelineRunResponse> {
  return apiRequest('/api/designspec/run-until', {
    method: 'POST',
    body: JSON.stringify({ stage }),
  }, 300000);
}

export async function compileDesignSpec(): Promise<{
  events: DesignSpecEvent[];
  spec: Record<string, any>;
}> {
  return apiRequest('/api/designspec/compile', {
    method: 'POST',
  });
}

export async function getDesignSpec(): Promise<{ spec: Record<string, any> }> {
  return apiRequest('/api/designspec/spec');
}

export async function getDesignSpecPatches(): Promise<{
  patches: Record<string, any>;
}> {
  return apiRequest('/api/designspec/patches');
}

export async function getDesignSpecStatus(): Promise<DesignSpecStatus> {
  return apiRequest('/api/designspec/status');
}

export async function getDesignSpecArtifacts(): Promise<{
  artifacts: Record<string, any>[];
}> {
  return apiRequest('/api/designspec/artifacts');
}

export async function pollDesignSpecEvents(): Promise<{
  events: DesignSpecEvent[];
}> {
  return apiRequest('/api/designspec/events');
}
