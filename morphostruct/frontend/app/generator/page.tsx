'use client';

import { useCallback, useState, useEffect, useRef } from 'react';
import { useRouter } from 'next/navigation';
import dynamic from 'next/dynamic';
import { ParameterPanel } from '@/components/controls';

const Viewport = dynamic(
  () => import('@/components/viewer/Viewport').then(mod => ({ default: mod.Viewport })),
  { ssr: false }
);
import { ChatPanel } from '@/components/chat';
import type { PipelineProgressData } from '@/components/chat';
import { ExportPanel } from '@/components/export';
import { SpecViewer } from '@/components/designspec';
import { useScaffoldStore, useChatStore } from '@/lib/store';
import { useAuthStore } from '@/lib/store/authStore';
import { usePreferencesStore } from '@/lib/store/preferencesStore';
import {
  generateScaffold, exportSTL, downloadBlob, sendChatMessage,
  sendDesignSpecMessage, approveDesignSpecPatch, rejectDesignSpecPatch,
  createDesignSpecProject,
} from '@/lib/api';
import { ScaffoldType } from '@/lib/types/scaffolds';
import { NavHeader } from '@/components/NavHeader';
import { Workflow, Wrench } from 'lucide-react';

export default function GeneratorPage() {
  const router = useRouter();
  const { isAuthenticated } = useAuthStore();

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!isAuthenticated) {
      router.push('/login');
    }
  }, [isAuthenticated, router]);

  // Scaffold store
  const {
    scaffoldType,
    setScaffoldType,
    params,
    setParams,
    resetParams,
    invert,
    setInvert,
    previewMode,
    setPreviewMode,
    meshData,
    setMeshData,
    isGenerating,
    setIsGenerating,
    validation,
    setValidation,
    stats,
    setStats,
    scaffoldId,
    setScaffoldId,
  } = useScaffoldStore();

  // Chat store
  const {
    messages,
    addMessage,
    isLoading: chatLoading,
    setIsLoading: setChatLoading,
    suggestions,
    setSuggestions,
    getConversationHistory,
  } = useChatStore();

  // Export state
  const [isExporting, setIsExporting] = useState(false);

  // DesignSpec mode state
  const [mode, setMode] = useState<'scaffold' | 'designspec'>('scaffold');
  const [designSpec, setDesignSpec] = useState<Record<string, any> | null>(null);
  const [pipelineProgress, setPipelineProgress] = useState<PipelineProgressData | null>(null);
  const [isPipelineRunning, setIsPipelineRunning] = useState(false);
  const designSpecInitialized = useRef(false);

  // Get generation timeout from preferences (default 60s)
  const generationTimeout = usePreferencesStore((state) => state.preferences?.generation_timeout_seconds) || 60;

  // Handle scaffold generation
  const handleGenerate = useCallback(async () => {
    setIsGenerating(true);
    try {
      // In preview mode, reduce resolution parameters for faster generation
      let effectiveParams = params;
      if (previewMode) {
        effectiveParams = { ...params };
        // Reduce resolution-related parameters
        if ('resolution' in effectiveParams && effectiveParams.resolution > 8) {
          effectiveParams.resolution = 8;
        }
        if ('samples_per_cell' in effectiveParams && effectiveParams.samples_per_cell > 12) {
          effectiveParams.samples_per_cell = 12;
        }
        // Reduce branch generations for tree-like structures
        if ('branch_generations' in effectiveParams && effectiveParams.branch_generations > 2) {
          effectiveParams.branch_generations = 2;
        }
        // Reduce cell counts for lattice structures
        if ('cell_count' in effectiveParams && effectiveParams.cell_count > 15) {
          effectiveParams.cell_count = 15;
        }
      }
      const result = await generateScaffold(scaffoldType, effectiveParams, previewMode, generationTimeout, invert);
      setMeshData(result.mesh);
      setValidation(result.validation);
      setStats(result.stats);
      setScaffoldId(result.scaffold_id);
    } catch (error) {
      console.error('Generation failed:', error);
    } finally {
      setIsGenerating(false);
    }
  }, [scaffoldType, params, previewMode, generationTimeout, invert, setIsGenerating, setMeshData, setValidation, setStats, setScaffoldId]);

  // Handle export
  const handleExport = useCallback(async (format: 'binary' | 'ascii') => {
    if (!scaffoldId) return;
    setIsExporting(true);
    try {
      const blob = await exportSTL(scaffoldId, format, generationTimeout);
      const filename = `scaffold_${scaffoldType}_${Date.now()}.stl`;
      downloadBlob(blob, filename);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  }, [scaffoldId, scaffoldType, generationTimeout]);

  // Handle scaffold-mode chat messages
  const handleSendMessage = useCallback(async (message: string) => {
    addMessage({ role: 'user', content: message });
    setChatLoading(true);

    try {
      if (mode === 'designspec') {
        if (!designSpecInitialized.current) {
          await createDesignSpecProject('web-session');
          designSpecInitialized.current = true;
        }

        const dsResponse = await sendDesignSpecMessage(message);

        const assistantContent = dsResponse.messages.join('\n\n') || 'Processing...';
        const patches = dsResponse.patches.map(p => ({
          patch_id: p.patch_id,
          description: p.description,
          diff: p.diff,
        }));

        addMessage({
          role: 'assistant',
          content: assistantContent,
          patches: patches.length > 0 ? patches : undefined,
        });

        if (dsResponse.spec) {
          setDesignSpec(dsResponse.spec);
        }
      } else {
        const response = await sendChatMessage(
          message,
          getConversationHistory(),
          { type: scaffoldType, ...params }
        );

        addMessage({
          role: 'assistant',
          content: response.message,
          suggestions: response.suggestions,
        });

        setSuggestions(response.suggestions);

        if (response.action === 'generate' && response.suggested_params) {
          const { type, ...newParams } = response.suggested_params;
          if (type && type !== scaffoldType) {
            setScaffoldType(type as ScaffoldType);
          }
          setParams({ ...params, ...newParams });
          setTimeout(() => handleGenerate(), 100);
        }
      }
    } catch (error) {
      console.error('Chat failed:', error);
      addMessage({
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
      });
    } finally {
      setChatLoading(false);
    }
  }, [mode, addMessage, setChatLoading, getConversationHistory, scaffoldType, params,
      setSuggestions, setScaffoldType, setParams, handleGenerate]);

  const handleApprovePatch = useCallback(async (patchId: string) => {
    setChatLoading(true);
    try {
      const result = await approveDesignSpecPatch(patchId);
      if (result.spec) {
        setDesignSpec(result.spec);
      }
      addMessage({
        role: 'assistant',
        content: `Patch applied successfully.`,
      });
    } catch (error) {
      console.error('Patch approval failed:', error);
      addMessage({
        role: 'assistant',
        content: 'Failed to apply patch. Please try again.',
      });
    } finally {
      setChatLoading(false);
    }
  }, [addMessage, setChatLoading]);

  const handleRejectPatch = useCallback(async (patchId: string, reason: string) => {
    setChatLoading(true);
    try {
      await rejectDesignSpecPatch(patchId, reason);
      addMessage({
        role: 'assistant',
        content: `Patch rejected.${reason ? ` Reason: ${reason}` : ''}`,
      });
    } catch (error) {
      console.error('Patch rejection failed:', error);
      addMessage({
        role: 'assistant',
        content: 'Failed to reject patch. Please try again.',
      });
    } finally {
      setChatLoading(false);
    }
  }, [addMessage, setChatLoading]);

  // Handle scaffold type change
  const handleScaffoldTypeChange = useCallback((type: ScaffoldType) => {
    setScaffoldType(type);
  }, [setScaffoldType]);

  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="h-screen flex flex-col bg-black">
      <NavHeader currentPage="generator" />

      {/* Mode Toggle */}
      <div className="flex items-center gap-2 px-4 py-2 border-b border-emerald-500/20 bg-black/50">
        <button
          onClick={() => setMode('scaffold')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
            mode === 'scaffold'
              ? 'bg-emerald-600 text-white'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <Wrench className="w-4 h-4" />
          Direct Generate
        </button>
        <button
          onClick={() => setMode('designspec')}
          className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
            mode === 'designspec'
              ? 'bg-purple-600 text-white'
              : 'text-slate-400 hover:text-white hover:bg-slate-800'
          }`}
        >
          <Workflow className="w-4 h-4" />
          DesignSpec Agent
        </button>
      </div>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left sidebar */}
        <aside className="w-80 border-r border-emerald-500/20 flex flex-col overflow-hidden shrink-0 bg-black/50">
          {mode === 'scaffold' ? (
            <>
              <div className="flex-1 overflow-y-auto">
                <ParameterPanel
                  scaffoldType={scaffoldType}
                  onScaffoldTypeChange={handleScaffoldTypeChange}
                  params={params}
                  onParamsChange={setParams}
                  onGenerate={handleGenerate}
                  onReset={resetParams}
                  isGenerating={isGenerating}
                  invert={invert}
                  onInvertChange={setInvert}
                  previewMode={previewMode}
                  onPreviewModeChange={setPreviewMode}
                />
              </div>
              <div className="border-t border-emerald-500/20">
                <ChatPanel
                  messages={messages}
                  onSendMessage={handleSendMessage}
                  isLoading={chatLoading}
                  suggestions={suggestions}
                  mode="scaffold"
                />
              </div>
            </>
          ) : (
            <>
              <div className="p-3 border-b border-emerald-500/20">
                <SpecViewer spec={designSpec} />
              </div>
              <div className="flex-1 overflow-hidden">
                <ChatPanel
                  messages={messages}
                  onSendMessage={handleSendMessage}
                  isLoading={chatLoading}
                  suggestions={suggestions}
                  mode="designspec"
                  onApprovePatch={handleApprovePatch}
                  onRejectPatch={handleRejectPatch}
                  pipelineProgress={pipelineProgress}
                  isPipelineRunning={isPipelineRunning}
                />
              </div>
            </>
          )}
        </aside>

        {/* Right content area */}
        <div className="flex-1 flex flex-col overflow-hidden p-4 gap-4 bg-gradient-to-br from-slate-950 via-slate-900 to-emerald-950">
          <div className="flex-1 min-h-0">
            <Viewport
              meshData={meshData || undefined}
              isLoading={isGenerating}
            />
          </div>

          <div className="h-auto shrink-0">
            <ExportPanel
              scaffoldId={scaffoldId}
              validation={validation}
              stats={stats}
              onExport={handleExport}
              isExporting={isExporting}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
