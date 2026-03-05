'use client';

import { useState, useEffect, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { useAuthStore } from '@/lib/store/authStore';
import { usePreferencesStore } from '@/lib/store/preferencesStore';
import { getApiKeys, saveApiKey, deleteApiKey, getPreferences, updatePreferences, testApiKey, ApiKey, Preferences, TestApiKeyResponse } from '@/lib/api/auth';
import { NavHeader } from '@/components/NavHeader';

type TestStatus = 'idle' | 'testing' | 'success' | 'error';

interface ProviderTestState {
  status: TestStatus;
  message?: string;
  modelInfo?: string;
}

type TabId = 'llm' | 'appearance' | 'defaults' | 'viewer' | 'export' | 'units' | 'accessibility' | 'data';

// Color picker component
function ColorPicker({
  colors,
  value,
  onChange,
  labels
}: {
  colors: string[];
  value: string;
  onChange: (color: string) => void;
  labels?: Record<string, string>;
}) {
  const colorClasses: Record<string, string> = {
    emerald: 'bg-emerald-500',
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
    orange: 'bg-orange-500',
    pink: 'bg-pink-500',
    black: 'bg-black border border-white/20',
    dark_gray: 'bg-zinc-800',
    light_gray: 'bg-zinc-400',
    white: 'bg-white',
  };

  return (
    <div className="flex gap-2">
      {colors.map(color => (
        <button
          key={color}
          onClick={() => onChange(color)}
          title={labels?.[color] || color.toUpperCase().replace('_', ' ')}
          className={`w-10 h-10 rounded-sm transition-all ${colorClasses[color] || 'bg-gray-500'} ${
            value === color
              ? 'ring-2 ring-offset-2 ring-offset-black ring-emerald-400 scale-110'
              : 'hover:scale-105'
          }`}
        />
      ))}
    </div>
  );
}

// Toggle switch component
function ToggleSwitch({
  enabled,
  onChange,
  disabled = false
}: {
  enabled: boolean;
  onChange: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onChange}
      disabled={disabled}
      className={`relative w-20 h-10 border-2 transition-all ${
        enabled
          ? 'bg-emerald-400 border-emerald-400'
          : 'bg-black border-white/20'
      } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      <div className={`absolute top-1 w-6 h-6 bg-black transition-all ${
        enabled ? 'right-1' : 'left-1'
      }`} />
      <span className={`absolute inset-0 flex items-center justify-center text-xs font-bold tracking-tight ${
        enabled ? 'text-black' : 'text-white/40'
      }`}>
        {enabled ? 'ON' : 'OFF'}
      </span>
    </button>
  );
}

// Setting row component
function SettingRow({
  title,
  description,
  children,
  noBorder = false
}: {
  title: string;
  description: string;
  children: React.ReactNode;
  noBorder?: boolean;
}) {
  return (
    <div className={`flex items-center justify-between py-5 ${noBorder ? '' : 'border-b border-white/10'}`}>
      <div className="flex-1 pr-8">
        <label className="text-lg font-black tracking-tight mb-1 block">{title}</label>
        <p className="text-xs font-mono text-white/40 leading-relaxed">{description}</p>
      </div>
      <div className="flex-shrink-0">
        {children}
      </div>
    </div>
  );
}

// Section header component
function SectionHeader({ title }: { title: string }) {
  return (
    <div className="border-l-4 border-emerald-400/50 pl-4 mb-6">
      <h4 className="text-sm font-black tracking-widest text-emerald-400">{title}</h4>
    </div>
  );
}

export default function SettingsPage() {
  const router = useRouter();
  const { user, logout } = useAuthStore();
  const storeUpdatePreference = usePreferencesStore((state) => state.updatePreference);
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([]);
  const [preferences, setPreferences] = useState<Preferences | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState<'success' | 'error'>('success');
  const [activeTab, setActiveTab] = useState<TabId>('llm');
  const [storageUsed, setStorageUsed] = useState<string>('0 KB');

  // LLM Config Form states
  const [provider, setProvider] = useState<'openai' | 'anthropic' | 'custom'>('openai');
  const [openaiKey, setOpenaiKey] = useState('');
  const [anthropicKey, setAnthropicKey] = useState('');
  const [customKey, setCustomKey] = useState('');
  const [customEndpoint, setCustomEndpoint] = useState('https://api.openai.com/v1');
  const [defaultProvider, setDefaultProvider] = useState<'openai' | 'anthropic' | 'custom'>('openai');

  // Show/hide states
  const [showOpenAI, setShowOpenAI] = useState(false);
  const [showAnthropic, setShowAnthropic] = useState(false);
  const [showCustom, setShowCustom] = useState(false);

  // Test states
  const [openaiTest, setOpenaiTest] = useState<ProviderTestState>({ status: 'idle' });
  const [anthropicTest, setAnthropicTest] = useState<ProviderTestState>({ status: 'idle' });
  const [customTest, setCustomTest] = useState<ProviderTestState>({ status: 'idle' });

  // Confirmation dialogs
  const [showClearDataConfirm, setShowClearDataConfirm] = useState(false);

  const calculateStorageUsed = useCallback(() => {
    if (typeof window === 'undefined') return;
    let total = 0;
    for (const key in localStorage) {
      if (localStorage.hasOwnProperty(key)) {
        total += localStorage[key].length * 2; // UTF-16 = 2 bytes per char
      }
    }
    if (total < 1024) {
      setStorageUsed(`${total} B`);
    } else if (total < 1024 * 1024) {
      setStorageUsed(`${(total / 1024).toFixed(2)} KB`);
    } else {
      setStorageUsed(`${(total / (1024 * 1024)).toFixed(2)} MB`);
    }
  }, []);

  useEffect(() => {
    loadData();
    calculateStorageUsed();
  }, [calculateStorageUsed]);

  const loadData = async () => {
    try {
      const [keys, prefs] = await Promise.all([getApiKeys(), getPreferences()]);
      setApiKeys(keys);
      setPreferences(prefs);
      // Sync with global store
      usePreferencesStore.getState().setPreferences(prefs);
      const prefsWithProvider = prefs as Preferences & { default_provider?: 'openai' | 'anthropic' | 'custom'; custom_endpoint?: string };
      const savedProvider = prefsWithProvider.default_provider || 'openai';
      setDefaultProvider(savedProvider);
      // Also set the provider tab to match saved default
      setProvider(savedProvider);
      // Load custom endpoint if saved
      if (prefsWithProvider.custom_endpoint) {
        setCustomEndpoint(prefsWithProvider.custom_endpoint);
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveApiKey = async (provider: 'openai' | 'anthropic') => {
    const key = provider === 'openai' ? openaiKey : anthropicKey;
    if (!key.trim()) return;

    setSaving(true);
    try {
      await saveApiKey(provider, key);
      setMessage(`${provider.toUpperCase()} KEY SAVED`);
      setMessageType('success');
      if (provider === 'openai') setOpenaiKey('');
      else setAnthropicKey('');
      await loadData();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setMessage(`SAVE FAILED: ${errorMessage}`);
      setMessageType('error');
    } finally {
      setSaving(false);
      setTimeout(() => setMessage(''), 3000);
    }
  };

  const handleDeleteApiKey = async (provider: string) => {
    if (!confirm(`DELETE ${provider.toUpperCase()} API KEY?`)) return;

    try {
      await deleteApiKey(provider);
      setMessage('KEY DELETED');
      setMessageType('success');
      await loadData();
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setMessage(`DELETE FAILED: ${errorMessage}`);
      setMessageType('error');
    }
    setTimeout(() => setMessage(''), 3000);
  };

  const handleTestConnection = async (testProvider: 'openai' | 'anthropic' | 'custom') => {
    let key = '';
    let endpoint = undefined;
    let setTestState: (state: ProviderTestState) => void;

    switch (testProvider) {
      case 'openai':
        key = openaiKey;
        setTestState = setOpenaiTest;
        break;
      case 'anthropic':
        key = anthropicKey;
        setTestState = setAnthropicTest;
        break;
      case 'custom':
        key = customKey;
        endpoint = customEndpoint;
        setTestState = setCustomTest;
        break;
    }

    if (!key.trim()) {
      setTestState({ status: 'error', message: 'NO API KEY PROVIDED' });
      setTimeout(() => setTestState({ status: 'idle' }), 3000);
      return;
    }

    setTestState({ status: 'testing' });

    try {
      const result = await testApiKey(testProvider, key, endpoint);
      if (result.success) {
        setTestState({
          status: 'success',
          message: 'CONNECTION SUCCESSFUL',
          modelInfo: result.model_info
        });
      } else {
        setTestState({
          status: 'error',
          message: result.error || 'CONNECTION FAILED'
        });
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'TEST FAILED';
      setTestState({
        status: 'error',
        message: errorMessage
      });
    }

    setTimeout(() => setTestState({ status: 'idle' }), 5000);
  };

  const handleUpdatePreference = async (key: keyof Preferences | 'default_provider', value: string | number | boolean) => {
    if (!preferences) return;

    setSaving(true);

    // Update local state immediately for UI feedback
    const updated = { ...preferences, [key]: value };
    setPreferences(updated);

    // Update the global preferences store for immediate theme/accent changes
    storeUpdatePreference(key as keyof Preferences, value as any);

    if (key === 'default_provider') setDefaultProvider(value as 'openai' | 'anthropic' | 'custom');

    try {
      await updatePreferences({ [key]: value });
      setMessage('PREFERENCE SAVED');
      setMessageType('success');
      setTimeout(() => setMessage(''), 2000);
    } catch (error) {
      console.error('Failed to update preference:', error);
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      setMessage(`UPDATE FAILED: ${errorMessage}`);
      setMessageType('error');
      setTimeout(() => setMessage(''), 3000);
    } finally {
      setSaving(false);
    }
  };

  const handleExportSettings = () => {
    if (!preferences) return;
    const blob = new Blob([JSON.stringify(preferences, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'morphostruct-settings.json';
    a.click();
    URL.revokeObjectURL(url);
    setMessage('SETTINGS EXPORTED');
    setMessageType('success');
    setTimeout(() => setMessage(''), 2000);
  };

  const handleImportSettings = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const imported = JSON.parse(text);

        // Validate imported settings structure
        if (typeof imported !== 'object' || imported === null || Array.isArray(imported)) {
          throw new Error('Invalid settings file: must be a JSON object');
        }

        // Define allowed preference keys with their expected types
        const allowedKeys = new Map<string, string>([
          ['theme', 'string'],
          ['accent_color', 'string'],
          ['sidebar_position', 'string'],
          ['compact_mode', 'boolean'],
          ['show_tooltips', 'boolean'],
          ['auto_generate', 'boolean'],
          ['default_scaffold_type', 'string'],
          ['default_porosity', 'number'],
          ['default_wall_thickness', 'number'],
          ['auto_save_drafts', 'boolean'],
          ['auto_save_interval', 'string'],
          ['generation_timeout_seconds', 'number'],
          ['camera_type', 'string'],
          ['background_color', 'string'],
          ['show_grid', 'boolean'],
          ['show_axis_helpers', 'boolean'],
          ['grid_snap', 'boolean'],
          ['grid_snap_size', 'number'],
          ['ambient_occlusion', 'boolean'],
          ['anti_aliasing', 'boolean'],
          ['default_export_format', 'string'],
          ['include_textures', 'boolean'],
          ['coordinate_system', 'string'],
          ['auto_download_after_generation', 'boolean'],
          ['default_filename_pattern', 'string'],
          ['measurement_units', 'string'],
          ['show_dimensions_in_viewport', 'boolean'],
          ['default_bbox_x', 'number'],
          ['default_bbox_y', 'number'],
          ['default_bbox_z', 'number'],
          ['default_resolution', 'string'],
          ['reduced_motion', 'boolean'],
          ['high_contrast_mode', 'boolean'],
          ['larger_text', 'boolean'],
          ['screen_reader_descriptions', 'boolean'],
          ['keyboard_shortcuts_enabled', 'boolean'],
          ['show_keyboard_shortcut_hints', 'boolean'],
          ['default_provider', 'string'],
          ['custom_endpoint', 'string'],
        ]);

        // Validate and filter settings
        const validated: Record<string, unknown> = {};
        for (const [key, value] of Object.entries(imported)) {
          const expectedType = allowedKeys.get(key);

          if (expectedType && typeof value === expectedType) {
            // Additional validation for specific fields
            if (expectedType === 'string' && (value as string).length > 500) {
              continue; // Skip strings that are too long
            }
            if (expectedType === 'number' && (!isFinite(value as number) || isNaN(value as number))) {
              continue; // Skip invalid numbers
            }
            validated[key] = value;
          }
        }

        if (Object.keys(validated).length === 0) {
          throw new Error('No valid settings found in file');
        }

        await updatePreferences(validated);
        await loadData();
        setMessage(`SETTINGS IMPORTED (${Object.keys(validated).length} preferences)`);
        setMessageType('success');
        setTimeout(() => setMessage(''), 2000);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'INVALID FILE';
        setMessage(`IMPORT FAILED: ${errorMessage}`);
        setMessageType('error');
        setTimeout(() => setMessage(''), 3000);
      }
    };
    input.click();
  };

  const handleClearAllData = () => {
    localStorage.clear();
    setShowClearDataConfirm(false);
    calculateStorageUsed();
    setMessage('ALL LOCAL DATA CLEARED');
    setMessageType('success');
    setTimeout(() => setMessage(''), 2000);
  };

  const handleExportAllScaffolds = async () => {
    try {
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/scaffolds`, {
        headers: {
          'Authorization': `Bearer ${useAuthStore.getState().token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Failed to fetch scaffolds');
      }

      const scaffolds = await response.json();

      if (!scaffolds || scaffolds.length === 0) {
        setMessage('NO SCAFFOLDS TO EXPORT');
        setMessageType('error');
        setTimeout(() => setMessage(''), 2000);
        return;
      }

      const exportData = {
        exported_at: new Date().toISOString(),
        scaffold_count: scaffolds.length,
        scaffolds: scaffolds
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const date = new Date().toISOString().split('T')[0];
      const a = document.createElement('a');
      a.href = url;
      a.download = `morphostruct-scaffolds-${date}.json`;
      a.click();
      URL.revokeObjectURL(url);

      setMessage(`EXPORTED ${scaffolds.length} SCAFFOLDS`);
      setMessageType('success');
      setTimeout(() => setMessage(''), 2000);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'EXPORT FAILED';
      setMessage(errorMessage.toUpperCase());
      setMessageType('error');
      setTimeout(() => setMessage(''), 3000);
    }
  };

  const handleLogout = () => {
    logout();
    router.push('/login');
  };

  const hasOpenAI = apiKeys.some(k => k.provider === 'openai');
  const hasAnthropic = apiKeys.some(k => k.provider === 'anthropic');

  const tabs: { id: TabId; label: string }[] = [
    { id: 'llm', label: 'LLM CONFIG' },
    { id: 'appearance', label: 'APPEARANCE' },
    { id: 'defaults', label: 'DEFAULTS' },
    { id: 'viewer', label: 'VIEWER' },
    { id: 'export', label: 'EXPORT' },
    { id: 'units', label: 'UNITS' },
    { id: 'accessibility', label: 'ACCESS.' },
    { id: 'data', label: 'DATA' }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-black flex items-center justify-center">
        <div className="relative">
          <div className="w-16 h-16 border-4 border-white/20 border-t-white rounded-full animate-spin" />
          <div className="absolute inset-0 w-16 h-16 border-4 border-transparent border-l-emerald-400/50 rounded-full animate-spin" style={{ animationDuration: '1.5s' }} />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      <NavHeader currentPage="settings" />

      {/* Cellular background pattern */}
      <div className="fixed inset-0 opacity-[0.02] pointer-events-none" style={{
        backgroundImage: `radial-gradient(circle at 20% 50%, white 1px, transparent 1px),
                         radial-gradient(circle at 80% 80%, white 1px, transparent 1px),
                         radial-gradient(circle at 40% 20%, white 1px, transparent 1px)`,
        backgroundSize: '100px 100px, 150px 150px, 80px 80px'
      }} />

      {/* Grid overlay */}
      <div className="fixed inset-0 opacity-[0.03] pointer-events-none" style={{
        backgroundImage: 'linear-gradient(rgba(255,255,255,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.05) 1px, transparent 1px)',
        backgroundSize: '50px 50px'
      }} />

      <div className="relative max-w-7xl mx-auto px-8 py-12">
        {/* Page Title */}
        <div className="mb-12 pb-8 border-b-4 border-white/10">
          <div className="text-xs font-mono text-emerald-400 tracking-widest mb-4">SYSTEM_CONFIGURATION</div>
          <h1 className="text-6xl font-black tracking-tighter mb-4" style={{ fontFamily: 'system-ui' }}>
            SETTINGS
          </h1>
          <div className="flex items-center gap-4 text-sm font-mono">
            <span className="text-white/40">USER:</span>
            <span className="text-white font-bold">{user?.username}</span>
            <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse" />
            <span className="text-emerald-400">ACTIVE</span>
          </div>
        </div>

        {/* Status message */}
        {message && (
          <div className={`mb-8 p-4 border-l-4 font-mono text-sm tracking-wide animate-pulse ${
            messageType === 'success'
              ? 'bg-emerald-400/10 border-emerald-400'
              : 'bg-red-400/10 border-red-400'
          }`}>
            <span className={messageType === 'success' ? 'text-emerald-400' : 'text-red-400'}>
              {messageType === 'success' ? '✓' : '✗'}
            </span> {message}
          </div>
        )}

        {/* Tab Navigation */}
        <nav className="mb-12 flex flex-wrap gap-1 border-b-2 border-white/10">
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-4 font-black tracking-tight transition-all relative text-sm ${
                activeTab === tab.id
                  ? 'bg-white text-black'
                  : 'text-white/40 hover:text-white hover:bg-white/5'
              }`}
            >
              {tab.label}
              {activeTab === tab.id && (
                <div className="absolute bottom-0 left-0 right-0 h-1 bg-emerald-400 shadow-[0_0_20px_rgba(16,185,129,0.5)]" />
              )}
            </button>
          ))}
        </nav>

        {/* Content sections */}
        <div className="space-y-8">
          {/* LLM CONFIG Section */}
          {activeTab === 'llm' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-emerald-400/50 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  CONNECT YOUR AI PROVIDER CREDENTIALS TO ENABLE SCAFFOLD GENERATION ASSISTANCE.
                  <br />
                  KEYS ARE ENCRYPTED AT REST USING AES-256-GCM. TEST CONNECTIONS BEFORE SAVING.
                </p>
              </div>

              {/* Provider Selector */}
              <div className="bg-white/[0.02] border-2 border-white/10 p-8 hover:border-emerald-400/30 transition-all">
                <h3 className="text-lg font-black tracking-tight mb-4 text-emerald-400">PROVIDER SELECTION</h3>
                <div className="flex gap-3">
                  {(['openai', 'anthropic', 'custom'] as const).map(p => (
                    <button
                      key={p}
                      onClick={() => setProvider(p)}
                      className={`flex-1 px-6 py-3 font-bold tracking-tight transition-all ${
                        provider === p
                          ? 'bg-emerald-400 text-black'
                          : 'bg-white/5 text-white/40 hover:text-white hover:bg-white/10'
                      }`}
                    >
                      {p.toUpperCase()}
                    </button>
                  ))}
                </div>
              </div>

              {/* OpenAI Configuration */}
              {provider === 'openai' && (
                <div className="bg-white/[0.02] border-2 border-white/10 p-8 hover:border-emerald-400/30 transition-all duration-300 relative group">
                  <div className="absolute top-0 right-0 w-48 h-48 bg-gradient-radial from-emerald-400/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h3 className="text-3xl font-black tracking-tight mb-1">OPENAI</h3>
                      <p className="text-xs font-mono text-white/40">GPT-4-TURBO, GPT-4, GPT-3.5-TURBO</p>
                    </div>
                    {hasOpenAI && (
                      <div className="flex items-center gap-2 px-3 py-1 bg-emerald-400/20 border border-emerald-400/50">
                        <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                        <span className="text-xs font-mono text-emerald-400 tracking-wide">CONFIGURED</span>
                      </div>
                    )}
                  </div>

                  <div className="space-y-4">
                    <div className="flex gap-2">
                      <input
                        type={showOpenAI ? "text" : "password"}
                        value={openaiKey}
                        onChange={(e) => setOpenaiKey(e.target.value)}
                        placeholder={hasOpenAI ? '                        ' : 'sk-proj-...'}
                        className="flex-1 px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                      />
                      <button
                        onClick={() => setShowOpenAI(!showOpenAI)}
                        className="px-4 py-3 bg-white/5 border-2 border-white/20 hover:border-white/40 transition-colors font-mono text-xs"
                      >
                        {showOpenAI ? 'HIDE' : 'SHOW'}
                      </button>
                    </div>

                    {/* Test result */}
                    {openaiTest.status !== 'idle' && (
                      <div className={`p-3 border-l-4 font-mono text-xs ${
                        openaiTest.status === 'testing' ? 'border-blue-400 bg-blue-400/10 text-blue-400' :
                        openaiTest.status === 'success' ? 'border-emerald-400 bg-emerald-400/10 text-emerald-400' :
                        'border-red-400 bg-red-400/10 text-red-400'
                      }`}>
                        {openaiTest.status === 'testing' && (
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                            <span>TESTING CONNECTION...</span>
                          </div>
                        )}
                        {openaiTest.status === 'success' && (
                          <>
                            <div>+ {openaiTest.message}</div>
                            {openaiTest.modelInfo && <div className="mt-1 text-white/60">MODEL: {openaiTest.modelInfo}</div>}
                          </>
                        )}
                        {openaiTest.status === 'error' && (
                          <div>x {openaiTest.message}</div>
                        )}
                      </div>
                    )}

                    <div className="flex gap-2">
                      <button
                        onClick={() => handleTestConnection('openai')}
                        disabled={!openaiKey.trim() || openaiTest.status === 'testing'}
                        className="px-6 py-3 bg-blue-500 text-white font-bold tracking-tight hover:bg-blue-400 disabled:bg-white/10 disabled:text-white/30 transition-colors"
                      >
                        TEST CONNECTION
                      </button>
                      <button
                        onClick={() => handleSaveApiKey('openai')}
                        disabled={!openaiKey.trim() || saving}
                        className="flex-1 px-6 py-3 bg-emerald-400 text-black font-bold tracking-tight hover:bg-emerald-300 disabled:bg-white/10 disabled:text-white/30 transition-colors"
                      >
                        {hasOpenAI ? 'UPDATE KEY' : 'SAVE KEY'}
                      </button>
                      {hasOpenAI && (
                        <button
                          onClick={() => handleDeleteApiKey('openai')}
                          className="px-6 py-3 bg-red-500/20 border-2 border-red-500/50 text-red-400 font-bold tracking-tight hover:bg-red-500/30 transition-colors"
                        >
                          DELETE
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Anthropic Configuration */}
              {provider === 'anthropic' && (
                <div className="bg-white/[0.02] border-2 border-white/10 p-8 hover:border-emerald-400/30 transition-all duration-300 relative group">
                  <div className="absolute top-0 right-0 w-48 h-48 bg-gradient-radial from-emerald-400/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h3 className="text-3xl font-black tracking-tight mb-1">ANTHROPIC</h3>
                      <p className="text-xs font-mono text-white/40">CLAUDE-3-OPUS, SONNET, HAIKU</p>
                    </div>
                    {hasAnthropic && (
                      <div className="flex items-center gap-2 px-3 py-1 bg-emerald-400/20 border border-emerald-400/50">
                        <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                        <span className="text-xs font-mono text-emerald-400 tracking-wide">CONFIGURED</span>
                      </div>
                    )}
                  </div>

                  <div className="space-y-4">
                    <div className="flex gap-2">
                      <input
                        type={showAnthropic ? "text" : "password"}
                        value={anthropicKey}
                        onChange={(e) => setAnthropicKey(e.target.value)}
                        placeholder={hasAnthropic ? '                        ' : 'sk-ant-...'}
                        className="flex-1 px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                      />
                      <button
                        onClick={() => setShowAnthropic(!showAnthropic)}
                        className="px-4 py-3 bg-white/5 border-2 border-white/20 hover:border-white/40 transition-colors font-mono text-xs"
                      >
                        {showAnthropic ? 'HIDE' : 'SHOW'}
                      </button>
                    </div>

                    {/* Test result */}
                    {anthropicTest.status !== 'idle' && (
                      <div className={`p-3 border-l-4 font-mono text-xs ${
                        anthropicTest.status === 'testing' ? 'border-blue-400 bg-blue-400/10 text-blue-400' :
                        anthropicTest.status === 'success' ? 'border-emerald-400 bg-emerald-400/10 text-emerald-400' :
                        'border-red-400 bg-red-400/10 text-red-400'
                      }`}>
                        {anthropicTest.status === 'testing' && (
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                            <span>TESTING CONNECTION...</span>
                          </div>
                        )}
                        {anthropicTest.status === 'success' && (
                          <>
                            <div>+ {anthropicTest.message}</div>
                            {anthropicTest.modelInfo && <div className="mt-1 text-white/60">MODEL: {anthropicTest.modelInfo}</div>}
                          </>
                        )}
                        {anthropicTest.status === 'error' && (
                          <div>x {anthropicTest.message}</div>
                        )}
                      </div>
                    )}

                    <div className="flex gap-2">
                      <button
                        onClick={() => handleTestConnection('anthropic')}
                        disabled={!anthropicKey.trim() || anthropicTest.status === 'testing'}
                        className="px-6 py-3 bg-blue-500 text-white font-bold tracking-tight hover:bg-blue-400 disabled:bg-white/10 disabled:text-white/30 transition-colors"
                      >
                        TEST CONNECTION
                      </button>
                      <button
                        onClick={() => handleSaveApiKey('anthropic')}
                        disabled={!anthropicKey.trim() || saving}
                        className="flex-1 px-6 py-3 bg-emerald-400 text-black font-bold tracking-tight hover:bg-emerald-300 disabled:bg-white/10 disabled:text-white/30 transition-colors"
                      >
                        {hasAnthropic ? 'UPDATE KEY' : 'SAVE KEY'}
                      </button>
                      {hasAnthropic && (
                        <button
                          onClick={() => handleDeleteApiKey('anthropic')}
                          className="px-6 py-3 bg-red-500/20 border-2 border-red-500/50 text-red-400 font-bold tracking-tight hover:bg-red-500/30 transition-colors"
                        >
                          DELETE
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* Custom Configuration */}
              {provider === 'custom' && (
                <div className="bg-white/[0.02] border-2 border-white/10 p-8 hover:border-emerald-400/30 transition-all duration-300 relative group">
                  <div className="absolute top-0 right-0 w-48 h-48 bg-gradient-radial from-emerald-400/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

                  <div className="flex items-start justify-between mb-6">
                    <div>
                      <h3 className="text-3xl font-black tracking-tight mb-1">CUSTOM PROVIDER</h3>
                      <p className="text-xs font-mono text-white/40">OPENAI-COMPATIBLE API (HARVARD, LOCAL)</p>
                    </div>
                    {hasOpenAI && defaultProvider === 'custom' && (
                      <div className="flex items-center gap-2 px-3 py-1 bg-emerald-400/20 border border-emerald-400/50">
                        <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
                        <span className="text-xs font-mono text-emerald-400 tracking-wide">CONFIGURED</span>
                      </div>
                    )}
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="block text-xs font-mono text-white/60 mb-2 tracking-wide">ENDPOINT URL</label>
                      <input
                        type="text"
                        value={customEndpoint}
                        onChange={(e) => setCustomEndpoint(e.target.value)}
                        placeholder="https://api.openai.com/v1"
                        className="w-full px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                      />
                    </div>

                    <div className="flex gap-2">
                      <input
                        type={showCustom ? "text" : "password"}
                        value={customKey}
                        onChange={(e) => setCustomKey(e.target.value)}
                        placeholder="API Key"
                        className="flex-1 px-4 py-3 bg-black border-2 border-white/20 text-white placeholder-white/20 focus:outline-none focus:border-emerald-400 font-mono text-sm transition-colors"
                      />
                      <button
                        onClick={() => setShowCustom(!showCustom)}
                        className="px-4 py-3 bg-white/5 border-2 border-white/20 hover:border-white/40 transition-colors font-mono text-xs"
                      >
                        {showCustom ? 'HIDE' : 'SHOW'}
                      </button>
                    </div>

                    {/* Test result */}
                    {customTest.status !== 'idle' && (
                      <div className={`p-3 border-l-4 font-mono text-xs ${
                        customTest.status === 'testing' ? 'border-blue-400 bg-blue-400/10 text-blue-400' :
                        customTest.status === 'success' ? 'border-emerald-400 bg-emerald-400/10 text-emerald-400' :
                        'border-red-400 bg-red-400/10 text-red-400'
                      }`}>
                        {customTest.status === 'testing' && (
                          <div className="flex items-center gap-2">
                            <div className="w-3 h-3 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                            <span>TESTING CONNECTION...</span>
                          </div>
                        )}
                        {customTest.status === 'success' && (
                          <>
                            <div>+ {customTest.message}</div>
                            {customTest.modelInfo && <div className="mt-1 text-white/60">MODEL: {customTest.modelInfo}</div>}
                          </>
                        )}
                        {customTest.status === 'error' && (
                          <div>x {customTest.message}</div>
                        )}
                      </div>
                    )}

                    <div className="flex gap-2">
                      <button
                        onClick={() => handleTestConnection('custom')}
                        disabled={!customKey.trim() || !customEndpoint.trim() || customTest.status === 'testing'}
                        className="px-6 py-3 bg-blue-500 text-white font-bold tracking-tight hover:bg-blue-400 disabled:bg-white/10 disabled:text-white/30 transition-colors"
                      >
                        TEST CONNECTION
                      </button>
                      <button
                        onClick={async () => {
                          if (!customKey.trim() || !customEndpoint.trim()) return;
                          setSaving(true);
                          try {
                            // Save API key as 'openai' since custom uses OpenAI-compatible API
                            await saveApiKey('openai', customKey);
                            // Save custom endpoint URL and default provider together
                            await updatePreferences({
                              custom_endpoint: customEndpoint,
                              default_provider: 'custom'
                            });
                            setMessage('CUSTOM PROVIDER SAVED');
                            setMessageType('success');
                            setCustomKey('');
                            setDefaultProvider('custom');
                            await loadData();
                          } catch (error) {
                            const errorMessage = error instanceof Error ? error.message : 'Unknown error';
                            setMessage(`SAVE FAILED: ${errorMessage}`);
                            setMessageType('error');
                          } finally {
                            setSaving(false);
                            setTimeout(() => setMessage(''), 3000);
                          }
                        }}
                        disabled={!customKey.trim() || !customEndpoint.trim() || saving}
                        className="flex-1 px-6 py-3 bg-emerald-400 text-black font-bold tracking-tight hover:bg-emerald-300 disabled:bg-white/10 disabled:text-white/30 transition-colors"
                      >
                        {hasOpenAI ? 'UPDATE & SAVE' : 'SAVE KEY & ENDPOINT'}
                      </button>
                    </div>

                    <div className="mt-4 p-4 bg-yellow-400/10 border-l-4 border-yellow-400">
                      <p className="text-xs font-mono text-yellow-400 leading-relaxed">
                        ! CUSTOM PROVIDER USES OPENAI-COMPATIBLE API. YOUR KEY WILL BE SAVED AND THE ENDPOINT URL STORED.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Default Provider Selection */}
              <div className="bg-white/[0.02] border-2 border-white/10 p-8">
                <h3 className="text-lg font-black tracking-tight mb-4 text-emerald-400">DEFAULT PROVIDER</h3>
                <p className="text-xs font-mono text-white/40 mb-4">SELECT WHICH PROVIDER TO USE FOR SCAFFOLD GENERATION</p>
                <select
                  value={defaultProvider}
                  onChange={(e) => handleUpdatePreference('default_provider', e.target.value)}
                  disabled={saving}
                  className="w-full px-6 py-3 bg-black border-2 border-white/20 text-white font-bold tracking-tight focus:outline-none focus:border-emerald-400 transition-colors disabled:opacity-50"
                >
                  <option value="openai">OPENAI</option>
                  <option value="anthropic">ANTHROPIC</option>
                  <option value="custom">CUSTOM</option>
                </select>
              </div>
            </div>
          )}

          {/* APPEARANCE Section */}
          {activeTab === 'appearance' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-emerald-400/50 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  CUSTOMIZE YOUR INTERFACE APPEARANCE, COLORS, AND VISUAL BEHAVIOR.
                </p>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="COLOR SCHEME" />

                {/* Theme */}
                <SettingRow
                  title="THEME"
                  description="INTERFACE COLOR SCHEME - DARK OR LIGHT MODE"
                >
                  <div className="flex gap-2">
                    {['dark', 'light'].map(theme => (
                      <button
                        key={theme}
                        onClick={() => handleUpdatePreference('theme', theme)}
                        disabled={saving}
                        className={`px-8 py-3 font-bold tracking-tight transition-all disabled:opacity-50 ${
                          preferences?.theme === theme
                            ? 'bg-white text-black'
                            : 'bg-white/5 text-white/40 hover:text-white hover:bg-white/10'
                        }`}
                      >
                        {theme.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </SettingRow>

                {/* Accent Color */}
                <SettingRow
                  title="ACCENT COLOR"
                  description="PRIMARY HIGHLIGHT COLOR FOR INTERACTIVE ELEMENTS"
                >
                  <div className={saving ? 'opacity-50 pointer-events-none' : ''}>
                    <ColorPicker
                      colors={['emerald', 'blue', 'purple', 'orange', 'pink']}
                      value={preferences?.accent_color || 'emerald'}
                      onChange={(color) => handleUpdatePreference('accent_color', color)}
                    />
                  </div>
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="LAYOUT" />

                {/* Sidebar Position */}
                <SettingRow
                  title="SIDEBAR POSITION"
                  description="PLACE SIDEBAR ON LEFT OR RIGHT SIDE OF VIEWPORT"
                >
                  <div className="flex gap-2">
                    {['left', 'right'].map(pos => (
                      <button
                        key={pos}
                        onClick={() => handleUpdatePreference('sidebar_position', pos)}
                        className={`px-8 py-3 font-bold tracking-tight transition-all ${
                          preferences?.sidebar_position === pos
                            ? 'bg-white text-black'
                            : 'bg-white/5 text-white/40 hover:text-white hover:bg-white/10'
                        }`}
                      >
                        {pos.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </SettingRow>

                {/* Compact Mode */}
                <SettingRow
                  title="COMPACT MODE"
                  description="REDUCE PADDING AND SPACING FOR DENSER UI"
                >
                  <ToggleSwitch
                    enabled={preferences?.compact_mode || false}
                    onChange={() => handleUpdatePreference('compact_mode', !preferences?.compact_mode)}
                    disabled={saving}
                  />
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="BEHAVIOR" />

                {/* Show Tooltips */}
                <SettingRow
                  title="SHOW TOOLTIPS"
                  description="DISPLAY HELPFUL HINTS ON HOVER"
                >
                  <ToggleSwitch
                    enabled={preferences?.show_tooltips !== false}
                    onChange={() => handleUpdatePreference('show_tooltips', !preferences?.show_tooltips)}
                    disabled={saving}
                  />
                </SettingRow>

                {/* Auto-generate */}
                <SettingRow
                  title="AUTO-GENERATE"
                  description="REGENERATE SCAFFOLD AUTOMATICALLY ON PARAMETER CHANGE"
                  noBorder
                >
                  <ToggleSwitch
                    enabled={preferences?.auto_generate || false}
                    onChange={() => handleUpdatePreference('auto_generate', !preferences?.auto_generate)}
                    disabled={saving}
                  />
                </SettingRow>
              </div>
            </div>
          )}

          {/* DEFAULTS Section */}
          {activeTab === 'defaults' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-emerald-400/50 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  SET DEFAULT VALUES FOR SCAFFOLD GENERATION PARAMETERS AND AUTO-SAVE BEHAVIOR.
                </p>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="SCAFFOLD DEFAULTS" />

                {/* Default scaffold type */}
                <SettingRow
                  title="DEFAULT SCAFFOLD TYPE"
                  description="INITIAL SCAFFOLD GEOMETRY WHEN CREATING NEW PROJECTS"
                >
                  <select
                    value={preferences?.default_scaffold_type || 'gyroid'}
                    onChange={(e) => handleUpdatePreference('default_scaffold_type', e.target.value)}
                    disabled={saving}
                    className="px-6 py-3 bg-black border-2 border-white/20 text-white font-bold tracking-tight focus:outline-none focus:border-emerald-400 transition-colors min-w-[200px] disabled:opacity-50"
                  >
                    <option value="gyroid">GYROID</option>
                    <option value="cubic">CUBIC</option>
                    <option value="hexagonal">HEXAGONAL</option>
                    <option value="diamond">DIAMOND</option>
                    <option value="octet">OCTET</option>
                    <option value="custom">CUSTOM</option>
                  </select>
                </SettingRow>

                {/* Default Porosity */}
                <SettingRow
                  title="DEFAULT POROSITY"
                  description="INITIAL POROSITY PERCENTAGE FOR NEW SCAFFOLDS"
                >
                  <div className="flex items-center gap-4">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={preferences?.default_porosity || 70}
                      onChange={(e) => {
                        const value = parseInt(e.target.value, 10);
                        if (!isNaN(value)) {
                          handleUpdatePreference('default_porosity', value);
                        }
                      }}
                      disabled={saving}
                      className="w-32 h-2 bg-white/10 appearance-none cursor-pointer accent-emerald-400 disabled:opacity-50"
                    />
                    <span className="font-mono text-sm w-12">{preferences?.default_porosity || 70}%</span>
                  </div>
                </SettingRow>

                {/* Default Wall Thickness */}
                <SettingRow
                  title="DEFAULT WALL THICKNESS"
                  description="INITIAL STRUT/WALL THICKNESS IN MILLIMETERS"
                  noBorder
                >
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min="0.1"
                      max="10"
                      step="0.1"
                      value={preferences?.default_wall_thickness || 0.5}
                      onChange={(e) => {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value) && value >= 0.1 && value <= 10) {
                          handleUpdatePreference('default_wall_thickness', value);
                        }
                      }}
                      disabled={saving}
                      className="w-24 px-3 py-2 bg-black border-2 border-white/20 text-white font-mono text-sm focus:outline-none focus:border-emerald-400 transition-colors text-center disabled:opacity-50"
                    />
                    <span className="font-mono text-sm text-white/40">mm</span>
                  </div>
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="AUTO-SAVE" />

                {/* Auto-save drafts */}
                <SettingRow
                  title="AUTO-SAVE DRAFTS"
                  description="AUTOMATICALLY SAVE WORK-IN-PROGRESS SCAFFOLDS"
                >
                  <ToggleSwitch
                    enabled={preferences?.auto_save_drafts !== false}
                    onChange={() => handleUpdatePreference('auto_save_drafts', !preferences?.auto_save_drafts)}
                    disabled={saving}
                  />
                </SettingRow>

                {/* Auto-save interval */}
                <SettingRow
                  title="AUTO-SAVE INTERVAL"
                  description="HOW OFTEN TO AUTO-SAVE WHEN ENABLED"
                  noBorder
                >
                  <select
                    value={preferences?.auto_save_interval || '5min'}
                    onChange={(e) => handleUpdatePreference('auto_save_interval', e.target.value)}
                    disabled={!preferences?.auto_save_drafts || saving}
                    className="px-6 py-3 bg-black border-2 border-white/20 text-white font-bold tracking-tight focus:outline-none focus:border-emerald-400 transition-colors disabled:opacity-50"
                  >
                    <option value="1min">1 MINUTE</option>
                    <option value="5min">5 MINUTES</option>
                    <option value="10min">10 MINUTES</option>
                    <option value="30min">30 MINUTES</option>
                  </select>
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="PERFORMANCE" />

                {/* Generation Timeout */}
                <SettingRow
                  title="GENERATION TIMEOUT"
                  description="MAXIMUM TIME ALLOWED FOR SCAFFOLD GENERATION (30s INCREMENTS)"
                  noBorder
                >
                  <select
                    value={preferences?.generation_timeout_seconds || 60}
                    onChange={(e) => handleUpdatePreference('generation_timeout_seconds', parseInt(e.target.value))}
                    disabled={saving}
                    className="px-6 py-3 bg-black border-2 border-white/20 text-white font-bold tracking-tight focus:outline-none focus:border-emerald-400 transition-colors disabled:opacity-50"
                  >
                    <option value={30}>30 SECONDS</option>
                    <option value={60}>1 MINUTE</option>
                    <option value={90}>1.5 MINUTES</option>
                    <option value={120}>2 MINUTES</option>
                    <option value={180}>3 MINUTES</option>
                    <option value={300}>5 MINUTES</option>
                    <option value={600}>10 MINUTES</option>
                  </select>
                </SettingRow>
              </div>

              <div className="p-4 bg-blue-400/10 border-l-4 border-blue-400">
                <p className="text-xs font-mono text-blue-400 leading-relaxed">
                  LONGER TIMEOUTS ALLOW COMPLEX SCAFFOLDS WITH HIGH RESOLUTION TO COMPLETE. IF GENERATION FAILS WITH TIMEOUT, TRY REDUCING RESOLUTION OR COMPLEXITY.
                </p>
              </div>
            </div>
          )}

          {/* VIEWER Section */}
          {activeTab === 'viewer' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-emerald-400/50 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  CONFIGURE 3D VIEWPORT RENDERING, CAMERA, AND VISUAL HELPERS.
                </p>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="CAMERA" />

                {/* Camera Type */}
                <SettingRow
                  title="CAMERA TYPE"
                  description="PERSPECTIVE FOR REALISM, ORTHOGRAPHIC FOR PRECISION"
                >
                  <div className="flex gap-2">
                    {['perspective', 'orthographic'].map(type => (
                      <button
                        key={type}
                        onClick={() => handleUpdatePreference('camera_type', type)}
                        className={`px-6 py-3 font-bold tracking-tight transition-all text-sm ${
                          (preferences?.camera_type || 'perspective') === type
                            ? 'bg-white text-black'
                            : 'bg-white/5 text-white/40 hover:text-white hover:bg-white/10'
                        }`}
                      >
                        {type.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </SettingRow>

                {/* Background Color */}
                <SettingRow
                  title="BACKGROUND COLOR"
                  description="VIEWPORT BACKGROUND - DARK FOR CONTRAST, LIGHT FOR PRINTS"
                  noBorder
                >
                  <ColorPicker
                    colors={['black', 'dark_gray', 'light_gray', 'white']}
                    value={preferences?.background_color || 'black'}
                    onChange={(color) => handleUpdatePreference('background_color', color)}
                    labels={{
                      black: 'BLACK',
                      dark_gray: 'DARK GRAY',
                      light_gray: 'LIGHT GRAY',
                      white: 'WHITE'
                    }}
                  />
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="HELPERS" />

                {/* Show Grid */}
                <SettingRow
                  title="SHOW GRID"
                  description="DISPLAY REFERENCE GRID IN VIEWPORT"
                >
                  <ToggleSwitch
                    enabled={preferences?.show_grid !== false}
                    onChange={() => handleUpdatePreference('show_grid', !preferences?.show_grid)}
                  />
                </SettingRow>

                {/* Show Axis Helpers */}
                <SettingRow
                  title="SHOW AXIS HELPERS"
                  description="DISPLAY X/Y/Z AXIS INDICATORS"
                >
                  <ToggleSwitch
                    enabled={preferences?.show_axis_helpers !== false}
                    onChange={() => handleUpdatePreference('show_axis_helpers', !preferences?.show_axis_helpers)}
                  />
                </SettingRow>

                {/* Grid Snap */}
                <SettingRow
                  title="GRID SNAP"
                  description="SNAP OBJECTS TO GRID WHEN MOVING"
                >
                  <ToggleSwitch
                    enabled={preferences?.grid_snap || false}
                    onChange={() => handleUpdatePreference('grid_snap', !preferences?.grid_snap)}
                  />
                </SettingRow>

                {/* Grid Snap Size */}
                <SettingRow
                  title="SNAP SIZE"
                  description="GRID SNAP INCREMENT IN MILLIMETERS"
                  noBorder
                >
                  <select
                    value={preferences?.grid_snap_size || 1}
                    onChange={(e) => handleUpdatePreference('grid_snap_size', parseFloat(e.target.value))}
                    disabled={!preferences?.grid_snap}
                    className="px-6 py-3 bg-black border-2 border-white/20 text-white font-bold tracking-tight focus:outline-none focus:border-emerald-400 transition-colors disabled:opacity-50"
                  >
                    <option value="0.1">0.1 mm</option>
                    <option value="0.5">0.5 mm</option>
                    <option value="1">1.0 mm</option>
                    <option value="2">2.0 mm</option>
                    <option value="5">5.0 mm</option>
                  </select>
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="RENDERING" />

                {/* Ambient Occlusion */}
                <SettingRow
                  title="AMBIENT OCCLUSION"
                  description="SOFT SHADOWS FOR DEPTH PERCEPTION (PERFORMANCE IMPACT)"
                >
                  <ToggleSwitch
                    enabled={preferences?.ambient_occlusion || false}
                    onChange={() => handleUpdatePreference('ambient_occlusion', !preferences?.ambient_occlusion)}
                  />
                </SettingRow>

                {/* Anti-aliasing */}
                <SettingRow
                  title="ANTI-ALIASING"
                  description="SMOOTH JAGGED EDGES (PERFORMANCE IMPACT)"
                  noBorder
                >
                  <ToggleSwitch
                    enabled={preferences?.anti_aliasing !== false}
                    onChange={() => handleUpdatePreference('anti_aliasing', !preferences?.anti_aliasing)}
                  />
                </SettingRow>
              </div>
            </div>
          )}

          {/* EXPORT Section */}
          {activeTab === 'export' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-emerald-400/50 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  CONFIGURE DEFAULT EXPORT SETTINGS FOR 3D MODEL OUTPUT.
                </p>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="FORMAT" />

                {/* Default Format */}
                <SettingRow
                  title="DEFAULT FORMAT"
                  description="PREFERRED FILE FORMAT FOR EXPORTED SCAFFOLDS"
                >
                  <select
                    value={preferences?.default_export_format || 'stl_binary'}
                    onChange={(e) => handleUpdatePreference('default_export_format', e.target.value)}
                    className="px-6 py-3 bg-black border-2 border-white/20 text-white font-bold tracking-tight focus:outline-none focus:border-emerald-400 transition-colors min-w-[200px]"
                  >
                    <option value="stl_binary">STL BINARY</option>
                    <option value="stl_ascii">STL ASCII</option>
                    <option value="obj">OBJ</option>
                  </select>
                </SettingRow>

                {/* Include Textures */}
                <SettingRow
                  title="INCLUDE TEXTURES"
                  description="EXPORT TEXTURE FILES WITH OBJ FORMAT"
                >
                  <ToggleSwitch
                    enabled={preferences?.include_textures || false}
                    onChange={() => handleUpdatePreference('include_textures', !preferences?.include_textures)}
                    disabled={preferences?.default_export_format !== 'obj'}
                  />
                </SettingRow>

                {/* Coordinate System */}
                <SettingRow
                  title="COORDINATE SYSTEM"
                  description="AXIS ORIENTATION FOR EXPORT"
                  noBorder
                >
                  <div className="flex gap-2">
                    {['y_up', 'z_up'].map(sys => (
                      <button
                        key={sys}
                        onClick={() => handleUpdatePreference('coordinate_system', sys)}
                        className={`px-6 py-3 font-bold tracking-tight transition-all ${
                          (preferences?.coordinate_system || 'y_up') === sys
                            ? 'bg-white text-black'
                            : 'bg-white/5 text-white/40 hover:text-white hover:bg-white/10'
                        }`}
                      >
                        {sys === 'y_up' ? 'Y-UP' : 'Z-UP'}
                      </button>
                    ))}
                  </div>
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="BEHAVIOR" />

                {/* Auto-download */}
                <SettingRow
                  title="AUTO-DOWNLOAD AFTER GENERATION"
                  description="AUTOMATICALLY DOWNLOAD SCAFFOLD WHEN GENERATION COMPLETES"
                >
                  <ToggleSwitch
                    enabled={preferences?.auto_download_after_generation || false}
                    onChange={() => handleUpdatePreference('auto_download_after_generation', !preferences?.auto_download_after_generation)}
                  />
                </SettingRow>

                {/* Filename Pattern */}
                <SettingRow
                  title="FILENAME PATTERN"
                  description="DEFAULT NAMING CONVENTION FOR EXPORTS"
                  noBorder
                >
                  <input
                    type="text"
                    value={preferences?.default_filename_pattern || '{type}_{date}_{time}'}
                    onChange={(e) => handleUpdatePreference('default_filename_pattern', e.target.value)}
                    placeholder="{type}_{date}_{time}"
                    className="w-64 px-4 py-3 bg-black border-2 border-white/20 text-white font-mono text-sm focus:outline-none focus:border-emerald-400 transition-colors"
                  />
                </SettingRow>
              </div>

              <div className="p-4 bg-blue-400/10 border-l-4 border-blue-400">
                <p className="text-xs font-mono text-blue-400 leading-relaxed">
                  FILENAME VARIABLES: {'{type}'} = scaffold type, {'{date}'} = YYYY-MM-DD, {'{time}'} = HH-MM-SS
                </p>
              </div>
            </div>
          )}

          {/* UNITS Section */}
          {activeTab === 'units' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-emerald-400/50 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  CONFIGURE MEASUREMENT UNITS AND DEFAULT DIMENSIONS FOR SCAFFOLDS.
                </p>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="MEASUREMENT" />

                {/* Measurement Units */}
                <SettingRow
                  title="MEASUREMENT UNITS"
                  description="UNIT SYSTEM FOR DIMENSIONS AND VALUES"
                >
                  <select
                    value={preferences?.measurement_units || 'mm'}
                    onChange={(e) => handleUpdatePreference('measurement_units', e.target.value)}
                    className="px-6 py-3 bg-black border-2 border-white/20 text-white font-bold tracking-tight focus:outline-none focus:border-emerald-400 transition-colors min-w-[200px]"
                  >
                    <option value="mm">MILLIMETERS (mm)</option>
                    <option value="inches">INCHES (in)</option>
                    <option value="um">MICROMETERS (um)</option>
                  </select>
                </SettingRow>

                {/* Show Dimensions in Viewport */}
                <SettingRow
                  title="SHOW DIMENSIONS IN VIEWPORT"
                  description="DISPLAY SIZE ANNOTATIONS ON 3D MODEL"
                  noBorder
                >
                  <ToggleSwitch
                    enabled={preferences?.show_dimensions_in_viewport || false}
                    onChange={() => handleUpdatePreference('show_dimensions_in_viewport', !preferences?.show_dimensions_in_viewport)}
                  />
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="DEFAULT BOUNDING BOX" />

                {/* Default Bounding Box */}
                <div className="grid grid-cols-3 gap-6 py-4">
                  <div>
                    <label className="block text-sm font-black tracking-tight mb-2">X DIMENSION</label>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min="1"
                        max="1000"
                        step="1"
                        value={preferences?.default_bbox_x || 10}
                        onChange={(e) => {
                          const value = parseFloat(e.target.value);
                          if (!isNaN(value) && value >= 1 && value <= 1000) {
                            handleUpdatePreference('default_bbox_x', value);
                          }
                        }}
                        disabled={saving}
                        className="flex-1 px-3 py-2 bg-black border-2 border-white/20 text-white font-mono text-sm focus:outline-none focus:border-emerald-400 transition-colors text-center disabled:opacity-50"
                      />
                      <span className="text-xs font-mono text-white/40">{preferences?.measurement_units || 'mm'}</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-black tracking-tight mb-2">Y DIMENSION</label>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min="1"
                        max="1000"
                        step="1"
                        value={preferences?.default_bbox_y || 10}
                        onChange={(e) => {
                          const value = parseFloat(e.target.value);
                          if (!isNaN(value) && value >= 1 && value <= 1000) {
                            handleUpdatePreference('default_bbox_y', value);
                          }
                        }}
                        disabled={saving}
                        className="flex-1 px-3 py-2 bg-black border-2 border-white/20 text-white font-mono text-sm focus:outline-none focus:border-emerald-400 transition-colors text-center disabled:opacity-50"
                      />
                      <span className="text-xs font-mono text-white/40">{preferences?.measurement_units || 'mm'}</span>
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-black tracking-tight mb-2">Z DIMENSION</label>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min="1"
                        max="1000"
                        step="1"
                        value={preferences?.default_bbox_z || 10}
                        onChange={(e) => {
                          const value = parseFloat(e.target.value);
                          if (!isNaN(value) && value >= 1 && value <= 1000) {
                            handleUpdatePreference('default_bbox_z', value);
                          }
                        }}
                        disabled={saving}
                        className="flex-1 px-3 py-2 bg-black border-2 border-white/20 text-white font-mono text-sm focus:outline-none focus:border-emerald-400 transition-colors text-center disabled:opacity-50"
                      />
                      <span className="text-xs font-mono text-white/40">{preferences?.measurement_units || 'mm'}</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="QUALITY" />

                {/* Default Resolution */}
                <SettingRow
                  title="DEFAULT RESOLUTION"
                  description="MESH QUALITY/DENSITY FOR NEW SCAFFOLDS"
                  noBorder
                >
                  <select
                    value={preferences?.default_resolution || 'medium'}
                    onChange={(e) => handleUpdatePreference('default_resolution', e.target.value)}
                    className="px-6 py-3 bg-black border-2 border-white/20 text-white font-bold tracking-tight focus:outline-none focus:border-emerald-400 transition-colors min-w-[200px]"
                  >
                    <option value="low">LOW (FAST)</option>
                    <option value="medium">MEDIUM</option>
                    <option value="high">HIGH (SLOW)</option>
                    <option value="ultra">ULTRA (VERY SLOW)</option>
                  </select>
                </SettingRow>
              </div>
            </div>
          )}

          {/* ACCESSIBILITY Section */}
          {activeTab === 'accessibility' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-emerald-400/50 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  ACCESSIBILITY OPTIONS FOR IMPROVED USABILITY AND COMFORT.
                </p>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="VISUAL" />

                {/* Reduced Motion */}
                <SettingRow
                  title="REDUCED MOTION"
                  description="DISABLE ANIMATIONS AND TRANSITIONS"
                >
                  <ToggleSwitch
                    enabled={preferences?.reduced_motion || false}
                    onChange={() => handleUpdatePreference('reduced_motion', !preferences?.reduced_motion)}
                  />
                </SettingRow>

                {/* High Contrast Mode */}
                <SettingRow
                  title="HIGH CONTRAST MODE"
                  description="INCREASE COLOR CONTRAST FOR BETTER VISIBILITY"
                >
                  <ToggleSwitch
                    enabled={preferences?.high_contrast_mode || false}
                    onChange={() => handleUpdatePreference('high_contrast_mode', !preferences?.high_contrast_mode)}
                  />
                </SettingRow>

                {/* Larger Text */}
                <SettingRow
                  title="LARGER TEXT"
                  description="INCREASE BASE FONT SIZE THROUGHOUT THE INTERFACE"
                  noBorder
                >
                  <ToggleSwitch
                    enabled={preferences?.larger_text || false}
                    onChange={() => handleUpdatePreference('larger_text', !preferences?.larger_text)}
                  />
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="ASSISTIVE" />

                {/* Screen Reader Descriptions */}
                <SettingRow
                  title="SCREEN READER DESCRIPTIONS"
                  description="ENHANCED ARIA LABELS AND DESCRIPTIONS"
                >
                  <ToggleSwitch
                    enabled={preferences?.screen_reader_descriptions !== false}
                    onChange={() => handleUpdatePreference('screen_reader_descriptions', !preferences?.screen_reader_descriptions)}
                  />
                </SettingRow>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="KEYBOARD" />

                {/* Keyboard Shortcuts Enabled */}
                <SettingRow
                  title="KEYBOARD SHORTCUTS"
                  description="ENABLE KEYBOARD SHORTCUTS FOR COMMON ACTIONS"
                >
                  <ToggleSwitch
                    enabled={preferences?.keyboard_shortcuts_enabled !== false}
                    onChange={() => handleUpdatePreference('keyboard_shortcuts_enabled', !preferences?.keyboard_shortcuts_enabled)}
                  />
                </SettingRow>

                {/* Show Keyboard Shortcut Hints */}
                <SettingRow
                  title="SHOW SHORTCUT HINTS"
                  description="DISPLAY KEYBOARD SHORTCUTS IN TOOLTIPS AND MENUS"
                  noBorder
                >
                  <ToggleSwitch
                    enabled={preferences?.show_keyboard_shortcut_hints !== false}
                    onChange={() => handleUpdatePreference('show_keyboard_shortcut_hints', !preferences?.show_keyboard_shortcut_hints)}
                    disabled={!preferences?.keyboard_shortcuts_enabled}
                  />
                </SettingRow>
              </div>
            </div>
          )}

          {/* DATA Section */}
          {activeTab === 'data' && (
            <div className="space-y-8 animate-in fade-in duration-300">
              <div className="border-l-4 border-emerald-400/50 pl-6">
                <p className="font-mono text-sm text-white/60 leading-relaxed">
                  MANAGE YOUR DATA, EXPORT SETTINGS, AND CLEAR LOCAL STORAGE.
                </p>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="STORAGE" />

                {/* Storage Used */}
                <div className="flex items-center justify-between py-5 border-b border-white/10">
                  <div className="flex-1 pr-8">
                    <label className="text-lg font-black tracking-tight mb-1 block">LOCAL STORAGE USED</label>
                    <p className="text-xs font-mono text-white/40 leading-relaxed">BROWSER LOCAL STORAGE CONSUMPTION</p>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="px-6 py-3 bg-white/5 border-2 border-white/10">
                      <span className="font-mono text-lg font-bold text-emerald-400">{storageUsed}</span>
                    </div>
                    <button
                      onClick={calculateStorageUsed}
                      className="px-4 py-3 bg-white/5 border-2 border-white/20 hover:border-white/40 transition-colors font-mono text-xs"
                    >
                      REFRESH
                    </button>
                  </div>
                </div>

                {/* Clear All Data */}
                <div className="flex items-center justify-between py-5">
                  <div className="flex-1 pr-8">
                    <label className="text-lg font-black tracking-tight mb-1 block text-red-400">CLEAR ALL LOCAL DATA</label>
                    <p className="text-xs font-mono text-white/40 leading-relaxed">
                      PERMANENTLY DELETE ALL LOCALLY STORED DATA INCLUDING CACHED SCAFFOLDS
                    </p>
                  </div>
                  {showClearDataConfirm ? (
                    <div className="flex gap-2">
                      <button
                        onClick={handleClearAllData}
                        className="px-6 py-3 bg-red-500 text-white font-bold tracking-tight hover:bg-red-400 transition-colors"
                      >
                        CONFIRM DELETE
                      </button>
                      <button
                        onClick={() => setShowClearDataConfirm(false)}
                        className="px-6 py-3 bg-white/5 border-2 border-white/20 text-white font-bold tracking-tight hover:border-white/40 transition-colors"
                      >
                        CANCEL
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setShowClearDataConfirm(true)}
                      className="px-6 py-3 bg-red-500/20 border-2 border-red-500/50 text-red-400 font-bold tracking-tight hover:bg-red-500/30 transition-colors"
                    >
                      CLEAR DATA
                    </button>
                  )}
                </div>
              </div>

              <div className="bg-white/[0.02] border-2 border-white/10 p-8 space-y-2">
                <SectionHeader title="EXPORT / IMPORT" />

                {/* Export Scaffolds */}
                <div className="flex items-center justify-between py-5 border-b border-white/10">
                  <div className="flex-1 pr-8">
                    <label className="text-lg font-black tracking-tight mb-1 block">EXPORT ALL SCAFFOLDS</label>
                    <p className="text-xs font-mono text-white/40 leading-relaxed">
                      DOWNLOAD ALL YOUR SCAFFOLDS AS A ZIP ARCHIVE
                    </p>
                  </div>
                  <button
                    onClick={handleExportAllScaffolds}
                    className="px-6 py-3 bg-emerald-400 text-black font-bold tracking-tight hover:bg-emerald-300 transition-colors"
                  >
                    EXPORT ZIP
                  </button>
                </div>

                {/* Export Settings */}
                <div className="flex items-center justify-between py-5 border-b border-white/10">
                  <div className="flex-1 pr-8">
                    <label className="text-lg font-black tracking-tight mb-1 block">EXPORT SETTINGS</label>
                    <p className="text-xs font-mono text-white/40 leading-relaxed">
                      DOWNLOAD YOUR PREFERENCES AS JSON FILE
                    </p>
                  </div>
                  <button
                    onClick={handleExportSettings}
                    className="px-6 py-3 bg-white/5 border-2 border-white/20 text-white font-bold tracking-tight hover:border-emerald-400 transition-colors"
                  >
                    EXPORT JSON
                  </button>
                </div>

                {/* Import Settings */}
                <div className="flex items-center justify-between py-5">
                  <div className="flex-1 pr-8">
                    <label className="text-lg font-black tracking-tight mb-1 block">IMPORT SETTINGS</label>
                    <p className="text-xs font-mono text-white/40 leading-relaxed">
                      RESTORE PREFERENCES FROM A JSON FILE
                    </p>
                  </div>
                  <button
                    onClick={handleImportSettings}
                    className="px-6 py-3 bg-white/5 border-2 border-white/20 text-white font-bold tracking-tight hover:border-emerald-400 transition-colors"
                  >
                    IMPORT JSON
                  </button>
                </div>
              </div>

              <div className="p-4 bg-yellow-400/10 border-l-4 border-yellow-400">
                <p className="text-xs font-mono text-yellow-400 leading-relaxed">
                  ! CLEARING LOCAL DATA CANNOT BE UNDONE. SCAFFOLD DATA SAVED TO YOUR ACCOUNT IS NOT AFFECTED.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
