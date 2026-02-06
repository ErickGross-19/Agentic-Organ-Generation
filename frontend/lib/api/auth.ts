import { authFetch } from '../store/authStore';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface ApiKey {
  id: number;
  provider: 'openai' | 'anthropic';
  has_key: boolean;
}

export interface Preferences {
  // Core required fields
  default_scaffold_type: string;
  theme: string;
  auto_generate: boolean;

  // Provider settings
  default_provider?: 'openai' | 'anthropic' | 'custom';

  // Appearance additions
  accent_color?: 'emerald' | 'blue' | 'purple' | 'orange' | 'pink';
  sidebar_position?: 'left' | 'right';
  compact_mode?: boolean;
  show_tooltips?: boolean;

  // Defaults additions
  default_porosity?: number;
  default_wall_thickness?: number;
  auto_save_drafts?: boolean;
  auto_save_interval?: '1min' | '5min' | '10min' | '30min';
  generation_timeout_seconds?: number;  // Must be multiple of 30, default 60

  // Viewer settings
  camera_type?: 'perspective' | 'orthographic';
  show_grid?: boolean;
  show_axis_helpers?: boolean;
  grid_snap?: boolean;
  grid_snap_size?: number;
  background_color?: 'black' | 'dark_gray' | 'light_gray' | 'white';
  ambient_occlusion?: boolean;
  anti_aliasing?: boolean;

  // Export settings
  default_export_format?: 'stl_binary' | 'stl_ascii' | 'obj';
  include_textures?: boolean;
  auto_download_after_generation?: boolean;
  default_filename_pattern?: string;
  coordinate_system?: 'y_up' | 'z_up';

  // Units settings
  measurement_units?: 'mm' | 'inches' | 'um';
  default_bbox_x?: number;
  default_bbox_y?: number;
  default_bbox_z?: number;
  default_resolution?: 'low' | 'medium' | 'high' | 'ultra';
  show_dimensions_in_viewport?: boolean;

  // Accessibility settings
  reduced_motion?: boolean;
  high_contrast_mode?: boolean;
  larger_text?: boolean;
  screen_reader_descriptions?: boolean;
  keyboard_shortcuts_enabled?: boolean;
  show_keyboard_shortcut_hints?: boolean;
}

export interface UserProfile {
  id: number;
  uuid: string;
  username: string;
  email: string | null;
  email_verified: boolean;
  display_name: string | null;
  avatar_url: string | null;
  created_at: string;
}

export interface Session {
  id: number;
  user_agent: string;
  ip_address: string;
  last_active: string;
  is_current: boolean;
}

// API Keys
export async function saveApiKey(provider: string, apiKey: string): Promise<ApiKey> {
  const response = await authFetch(`${API_BASE}/api/auth/api-keys`, {
    method: 'POST',
    body: JSON.stringify({ provider, api_key: apiKey }),
  });
  if (!response.ok) throw new Error('Failed to save API key');
  return response.json();
}

export async function getApiKeys(): Promise<ApiKey[]> {
  const response = await authFetch(`${API_BASE}/api/auth/api-keys`);
  if (!response.ok) throw new Error('Failed to get API keys');
  return response.json();
}

export async function deleteApiKey(provider: string): Promise<void> {
  const response = await authFetch(`${API_BASE}/api/auth/api-keys/${provider}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to delete API key');
}

// Preferences
export async function getPreferences(): Promise<Preferences> {
  const response = await authFetch(`${API_BASE}/api/auth/preferences`);
  if (!response.ok) throw new Error('Failed to get preferences');
  return response.json();
}

export async function updatePreferences(prefs: Partial<Preferences>): Promise<Preferences> {
  const response = await authFetch(`${API_BASE}/api/auth/preferences`, {
    method: 'PUT',
    body: JSON.stringify(prefs),
  });
  if (!response.ok) throw new Error('Failed to update preferences');
  return response.json();
}

// Profile Management
export async function getUserProfile(): Promise<UserProfile> {
  const response = await authFetch(`${API_BASE}/api/auth/profile`);
  if (!response.ok) throw new Error('Failed to get profile');
  return response.json();
}

export async function updateProfile(data: Partial<UserProfile>): Promise<UserProfile> {
  const response = await authFetch(`${API_BASE}/api/auth/profile`, {
    method: 'PUT',
    body: JSON.stringify(data),
  });
  if (!response.ok) throw new Error('Failed to update profile');
  return response.json();
}

export async function sendVerificationEmail(): Promise<void> {
  const response = await authFetch(`${API_BASE}/api/auth/send-verification`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to send verification email');
}

// Password Management
export async function changePassword(currentPassword: string, newPassword: string): Promise<void> {
  const response = await authFetch(`${API_BASE}/api/auth/change-password`, {
    method: 'POST',
    body: JSON.stringify({ current_password: currentPassword, new_password: newPassword }),
  });
  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.detail || 'Failed to change password');
  }
}

// Session Management
export async function getSessions(): Promise<Session[]> {
  const response = await authFetch(`${API_BASE}/api/auth/sessions`);
  if (!response.ok) throw new Error('Failed to get sessions');
  return response.json();
}

export async function revokeSession(sessionId: number): Promise<void> {
  const response = await authFetch(`${API_BASE}/api/auth/sessions/${sessionId}`, {
    method: 'DELETE',
  });
  if (!response.ok) throw new Error('Failed to revoke session');
}

export async function revokeAllSessions(): Promise<void> {
  const response = await authFetch(`${API_BASE}/api/auth/sessions/revoke-all`, {
    method: 'POST',
  });
  if (!response.ok) throw new Error('Failed to revoke sessions');
}

// Account Deletion
export async function deleteAccount(password: string): Promise<void> {
  const response = await authFetch(`${API_BASE}/api/auth/account`, {
    method: 'DELETE',
    body: JSON.stringify({ password }),
  });
  if (!response.ok) {
    const data = await response.json();
    throw new Error(data.detail || 'Failed to delete account');
  }
}

// Test API Key
export interface TestApiKeyResponse {
  success: boolean;
  provider: string;
  model_info?: string;
  error?: string;
}

export async function testApiKey(provider: 'openai' | 'anthropic' | 'custom', apiKey: string, customEndpoint?: string): Promise<TestApiKeyResponse> {
  const response = await authFetch(`${API_BASE}/api/auth/test-api-key`, {
    method: 'POST',
    body: JSON.stringify({
      provider,
      api_key: apiKey,
      custom_endpoint: customEndpoint
    }),
  });
  if (!response.ok) throw new Error('Failed to test API key');
  return response.json();
}
