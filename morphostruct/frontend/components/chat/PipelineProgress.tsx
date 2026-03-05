'use client';

import { Loader2, CheckCircle2, XCircle, Clock } from 'lucide-react';

const PIPELINE_STAGES = [
  'compile_policies',
  'compile_domains',
  'component_ports',
  'component_build',
  'component_mesh',
  'union_voids',
  'mesh_domain',
  'embed',
  'port_recarve',
  'validity',
  'export',
];

const STAGE_LABELS: Record<string, string> = {
  compile_policies: 'Compile Policies',
  compile_domains: 'Compile Domains',
  component_ports: 'Component Ports',
  component_build: 'Component Build',
  component_mesh: 'Component Mesh',
  union_voids: 'Union Voids',
  mesh_domain: 'Mesh Domain',
  embed: 'Embed',
  port_recarve: 'Port Recarve',
  validity: 'Validity',
  export: 'Export',
};

export interface PipelineProgressData {
  stage: string;
  stage_index: number;
  total_stages: number;
  progress_percent: number;
  status: string;
  elapsed_time: number;
  estimated_remaining: number;
}

interface PipelineProgressProps {
  progress: PipelineProgressData | null;
  isRunning: boolean;
}

function formatTime(seconds: number): string {
  if (seconds < 1) return '<1s';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.round(seconds % 60);
  return `${mins}m ${secs}s`;
}

export function PipelineProgress({ progress, isRunning }: PipelineProgressProps) {
  if (!progress && !isRunning) return null;

  const progressPercent = progress?.progress_percent ?? 0;
  const currentStage = progress?.stage ?? '';
  const status = progress?.status ?? 'running';

  const StatusIcon = () => {
    if (status === 'completed') return <CheckCircle2 className="w-4 h-4 text-green-500" />;
    if (status === 'failed' || status === 'cancelled') return <XCircle className="w-4 h-4 text-red-500" />;
    if (isRunning) return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />;
    return <Clock className="w-4 h-4 text-slate-400" />;
  };

  return (
    <div className="border border-blue-200 dark:border-blue-800 rounded-lg p-3 my-2 bg-blue-50 dark:bg-blue-900/20">
      <div className="flex items-center gap-2 mb-2">
        <StatusIcon />
        <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
          Pipeline {status === 'completed' ? 'Complete' : status === 'failed' ? 'Failed' : 'Running'}
        </span>
        {progress?.elapsed_time !== undefined && progress.elapsed_time > 0 && (
          <span className="text-xs text-blue-600 dark:text-blue-400 ml-auto">
            {formatTime(progress.elapsed_time)}
            {progress.estimated_remaining > 0 && status === 'running' && (
              <> (~{formatTime(progress.estimated_remaining)} remaining)</>
            )}
          </span>
        )}
      </div>

      <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2 mb-2">
        <div
          className={`h-2 rounded-full transition-all duration-300 ${
            status === 'completed'
              ? 'bg-green-500'
              : status === 'failed'
              ? 'bg-red-500'
              : 'bg-blue-500'
          }`}
          style={{ width: `${Math.min(100, progressPercent)}%` }}
        />
      </div>

      <div className="flex flex-wrap gap-1">
        {PIPELINE_STAGES.map((stage, idx) => {
          const stageIdx = PIPELINE_STAGES.indexOf(currentStage);
          let stageStatus: 'done' | 'current' | 'pending' = 'pending';
          if (status === 'completed') {
            stageStatus = 'done';
          } else if (idx < stageIdx) {
            stageStatus = 'done';
          } else if (idx === stageIdx && isRunning) {
            stageStatus = 'current';
          }

          return (
            <span
              key={stage}
              className={`text-[10px] px-1.5 py-0.5 rounded ${
                stageStatus === 'done'
                  ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
                  : stageStatus === 'current'
                  ? 'bg-blue-200 text-blue-700 dark:bg-blue-800 dark:text-blue-300 font-medium'
                  : 'bg-slate-100 text-slate-400 dark:bg-slate-800 dark:text-slate-500'
              }`}
            >
              {STAGE_LABELS[stage] || stage}
            </span>
          );
        })}
      </div>
    </div>
  );
}
