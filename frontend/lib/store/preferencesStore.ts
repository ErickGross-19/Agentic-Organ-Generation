'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { Preferences, getPreferences, updatePreferences } from '../api/auth';

interface PreferencesState {
  preferences: Preferences | null;
  isLoading: boolean;
  isHydrated: boolean;

  // Actions
  setPreferences: (preferences: Preferences) => void;
  updatePreference: <K extends keyof Preferences>(key: K, value: Preferences[K]) => Promise<void>;
  loadPreferences: () => Promise<void>;
  setHydrated: (hydrated: boolean) => void;
}

// Default preferences for unauthenticated users or initial load
export const defaultPreferences: Preferences = {
  default_scaffold_type: 'gyroid',
  theme: 'dark',
  auto_generate: false,
  accent_color: 'emerald',
  sidebar_position: 'left',
  compact_mode: false,
  show_tooltips: true,
  default_porosity: 70,
  default_wall_thickness: 0.5,
  auto_save_drafts: true,
  auto_save_interval: '5min',
  generation_timeout_seconds: 60,
  camera_type: 'perspective',
  show_grid: true,
  show_axis_helpers: true,
  grid_snap: false,
  grid_snap_size: 1,
  background_color: 'black',
  ambient_occlusion: false,
  anti_aliasing: true,
  default_export_format: 'stl_binary',
  include_textures: false,
  auto_download_after_generation: false,
  default_filename_pattern: '{type}_{date}_{time}',
  coordinate_system: 'y_up',
  measurement_units: 'mm',
  default_bbox_x: 10,
  default_bbox_y: 10,
  default_bbox_z: 10,
  default_resolution: 'medium',
  show_dimensions_in_viewport: false,
  reduced_motion: false,
  high_contrast_mode: false,
  larger_text: false,
  screen_reader_descriptions: true,
  keyboard_shortcuts_enabled: true,
  show_keyboard_shortcut_hints: true,
};

export const usePreferencesStore = create<PreferencesState>()(
  persist(
    (set, get) => ({
      preferences: defaultPreferences,
      isLoading: false,
      isHydrated: false,

      setPreferences: (preferences) => set({ preferences }),

      setHydrated: (hydrated) => set({ isHydrated: hydrated }),

      updatePreference: async (key, value) => {
        const current = get().preferences;
        if (!current) return;

        // Optimistically update local state
        const updated = { ...current, [key]: value };
        set({ preferences: updated });

        // Try to persist to server (will fail if not authenticated, which is fine)
        try {
          await updatePreferences({ [key]: value });
        } catch (error) {
          // If server update fails, the local state is still updated
          // This allows preferences to work for non-authenticated users via localStorage
          console.debug('Server preferences update failed (user may not be authenticated):', error);
        }
      },

      loadPreferences: async () => {
        set({ isLoading: true });
        try {
          const serverPrefs = await getPreferences();
          // Merge with defaults to ensure all fields exist
          set({ preferences: { ...defaultPreferences, ...serverPrefs } });
        } catch (error) {
          // If loading fails, keep using local/default preferences
          console.debug('Failed to load server preferences:', error);
        } finally {
          set({ isLoading: false });
        }
      },
    }),
    {
      name: 'morphostruct-preferences',
      // Only persist preferences, not loading state
      partialize: (state) => ({ preferences: state.preferences }),
      onRehydrateStorage: () => (state) => {
        state?.setHydrated(true);
      },
    }
  )
);

// Selector hooks for common preferences
export const useTheme = () => usePreferencesStore((state) => state.preferences?.theme ?? 'dark');
export const useAccentColor = () => usePreferencesStore((state) => state.preferences?.accent_color ?? 'emerald');
export const useCompactMode = () => usePreferencesStore((state) => state.preferences?.compact_mode ?? false);
export const useReducedMotion = () => usePreferencesStore((state) => state.preferences?.reduced_motion ?? false);
export const useLargerText = () => usePreferencesStore((state) => state.preferences?.larger_text ?? false);
export const useHighContrast = () => usePreferencesStore((state) => state.preferences?.high_contrast_mode ?? false);
