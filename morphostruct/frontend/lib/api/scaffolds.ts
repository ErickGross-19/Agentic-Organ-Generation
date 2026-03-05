import { apiRequest } from './client';
import { ScaffoldType } from '../types/scaffolds';

export interface SavedScaffold {
  uuid: string;
  name: string;
  type: ScaffoldType;
  parameters: Record<string, any>;
  thumbnail_url?: string;
  created_at: string;
  updated_at: string;
  user_id: number;
}

export interface ListScaffoldsResponse {
  scaffolds: SavedScaffold[];
  total: number;
}

export async function listScaffolds(): Promise<ListScaffoldsResponse> {
  return apiRequest<ListScaffoldsResponse>('/api/scaffolds');
}

export async function getScaffold(uuid: string): Promise<SavedScaffold> {
  return apiRequest<SavedScaffold>(`/api/scaffolds/${uuid}`);
}

export async function deleteScaffold(uuid: string): Promise<{ success: boolean }> {
  return apiRequest<{ success: boolean }>(`/api/scaffolds/${uuid}`, {
    method: 'DELETE',
  });
}

export async function duplicateScaffold(uuid: string): Promise<SavedScaffold> {
  return apiRequest<SavedScaffold>(`/api/scaffolds/${uuid}/duplicate`, {
    method: 'POST',
  });
}
