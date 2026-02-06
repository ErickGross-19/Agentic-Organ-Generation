'use client';

import { ReactNode, useEffect, useState } from 'react';
import { usePreferencesStore, useTheme, useAccentColor, useReducedMotion, useLargerText, useHighContrast, useCompactMode } from '@/lib/store/preferencesStore';
import { useAuthStore } from '@/lib/store/authStore';

interface ProvidersProps {
  children: ReactNode;
}

// Accent color CSS variable mappings (HSL values)
const accentColorMap: Record<string, string> = {
  emerald: '160 84.1% 39.4%',
  blue: '217.2 91.2% 59.8%',
  purple: '270 50% 60%',
  orange: '24.6 95% 53.1%',
  pink: '330 81% 60%',
};

/**
 * Theme and preferences provider
 * Applies CSS classes and variables based on user preferences
 */
function ThemeProvider({ children }: { children: ReactNode }) {
  const theme = useTheme();
  const accentColor = useAccentColor();
  const reducedMotion = useReducedMotion();
  const largerText = useLargerText();
  const highContrast = useHighContrast();
  const compactMode = useCompactMode();
  const isHydrated = usePreferencesStore((state) => state.isHydrated);
  const [mounted, setMounted] = useState(false);

  // Wait for client-side hydration
  useEffect(() => {
    setMounted(true);
  }, []);

  // Apply theme class to document
  useEffect(() => {
    if (!mounted || !isHydrated) return;

    const root = document.documentElement;

    // Theme (dark/light)
    if (theme === 'dark') {
      root.classList.add('dark');
      root.classList.remove('light');
    } else {
      root.classList.remove('dark');
      root.classList.add('light');
    }

    // Accent color - update CSS variable
    const accentHSL = accentColorMap[accentColor] || accentColorMap.emerald;
    root.style.setProperty('--accent', accentHSL);

    // Reduced motion
    if (reducedMotion) {
      root.classList.add('reduce-motion');
    } else {
      root.classList.remove('reduce-motion');
    }

    // Larger text
    if (largerText) {
      root.classList.add('larger-text');
    } else {
      root.classList.remove('larger-text');
    }

    // High contrast
    if (highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }

    // Compact mode
    if (compactMode) {
      root.classList.add('compact');
    } else {
      root.classList.remove('compact');
    }
  }, [theme, accentColor, reducedMotion, largerText, highContrast, compactMode, mounted, isHydrated]);

  // Prevent flash of unstyled content
  if (!mounted) {
    return (
      <div className="min-h-screen bg-black">
        {children}
      </div>
    );
  }

  return <>{children}</>;
}

/**
 * Preferences sync provider
 * Loads preferences from server when user is authenticated
 */
function PreferencesSyncProvider({ children }: { children: ReactNode }) {
  const { isAuthenticated } = useAuthStore();
  const loadPreferences = usePreferencesStore((state) => state.loadPreferences);

  useEffect(() => {
    if (isAuthenticated) {
      loadPreferences();
    }
  }, [isAuthenticated, loadPreferences]);

  return <>{children}</>;
}

/**
 * Client-side providers wrapper
 * Wraps the application with theme and preferences providers
 */
export function Providers({ children }: ProvidersProps) {
  return (
    <PreferencesSyncProvider>
      <ThemeProvider>
        {children}
      </ThemeProvider>
    </PreferencesSyncProvider>
  );
}
