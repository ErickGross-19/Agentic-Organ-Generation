import { authFetch } from '../store/authStore';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Scaffold {
  id: number;
  uuid: string;
  name: string;
  scaffold_type: string;
  parameters: Record<string, any>;
  thumbnail_path: string | null;
  stl_path: string | null;
  created_at: string;
  updated_at: string;
}

export interface ScaffoldListResponse {
  scaffolds: Scaffold[];
  total: number;
}

export interface ScaffoldStats {
  total: number;
  this_week: number;
  most_used_type: string;
}

/**
 * Get all scaffolds for the current user
 */
export async function getUserScaffolds(): Promise<ScaffoldListResponse> {
  const response = await authFetch(`${API_BASE}/api/scaffolds`);
  if (!response.ok) {
    throw new Error('Failed to fetch scaffolds');
  }
  return response.json();
}

/**
 * Get a specific scaffold by UUID
 */
export async function getScaffold(uuid: string): Promise<Scaffold> {
  const response = await authFetch(`${API_BASE}/api/scaffolds/${uuid}`);
  if (!response.ok) {
    throw new Error('Failed to fetch scaffold');
  }
  return response.json();
}

/**
 * Calculate statistics from scaffolds
 */
export function calculateScaffoldStats(scaffolds: Scaffold[]): ScaffoldStats {
  const total = scaffolds.length;

  // Calculate this week's count
  const oneWeekAgo = new Date();
  oneWeekAgo.setDate(oneWeekAgo.getDate() - 7);
  const thisWeek = scaffolds.filter(s => new Date(s.created_at) > oneWeekAgo).length;

  // Find most used type
  const typeCounts: Record<string, number> = {};
  scaffolds.forEach(s => {
    typeCounts[s.scaffold_type] = (typeCounts[s.scaffold_type] || 0) + 1;
  });

  const mostUsedType = Object.keys(typeCounts).length > 0
    ? Object.entries(typeCounts).sort((a, b) => b[1] - a[1])[0][0]
    : 'None';

  return { total, this_week: thisWeek, most_used_type: mostUsedType };
}

/**
 * Format date for display
 */
export function formatScaffoldDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
}
