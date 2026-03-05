'use client';

import { Download, FileType, AlertTriangle, CheckCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ValidationSummary } from './ValidationSummary';

interface ExportPanelProps {
  scaffoldId: string | null;
  validation: {
    is_valid: boolean;
    is_printable: boolean;
    warnings: string[];
    errors: string[];
  } | null;
  stats: {
    triangle_count: number;
    volume_mm3: number;
    generation_time_ms: number;
  } | null;
  onExport: (format: 'binary' | 'ascii') => void;
  isExporting?: boolean;
}

export function ExportPanel({
  scaffoldId,
  validation,
  stats,
  onExport,
  isExporting = false,
}: ExportPanelProps) {
  const canExport = scaffoldId && validation?.is_valid;

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow-sm p-4 space-y-4">
      <h3 className="font-semibold flex items-center gap-2">
        <FileType className="w-5 h-5 text-blue-500" />
        Export Scaffold
      </h3>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div className="bg-slate-50 dark:bg-slate-700/50 p-2 rounded">
            <div className="text-slate-500 dark:text-slate-400">Triangles</div>
            <div className="font-medium">{stats.triangle_count.toLocaleString()}</div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-700/50 p-2 rounded">
            <div className="text-slate-500 dark:text-slate-400">Volume</div>
            <div className="font-medium">{stats.volume_mm3.toFixed(2)} mm³</div>
          </div>
          <div className="bg-slate-50 dark:bg-slate-700/50 p-2 rounded col-span-2">
            <div className="text-slate-500 dark:text-slate-400">Generation Time</div>
            <div className="font-medium">{(stats.generation_time_ms / 1000).toFixed(2)}s</div>
          </div>
        </div>
      )}

      {/* Validation Summary */}
      {validation && (
        <ValidationSummary validation={validation} />
      )}

      {/* Export Buttons */}
      <div className="flex gap-2">
        <Button
          onClick={() => onExport('binary')}
          disabled={!canExport || isExporting}
          className="flex-1"
        >
          <Download className="w-4 h-4 mr-2" />
          {isExporting ? 'Exporting...' : 'STL (Binary)'}
        </Button>
        <Button
          onClick={() => onExport('ascii')}
          disabled={!canExport || isExporting}
          variant="outline"
          className="flex-1"
        >
          <Download className="w-4 h-4 mr-2" />
          STL (ASCII)
        </Button>
      </div>

      {!scaffoldId && (
        <p className="text-sm text-slate-500 text-center">
          Generate a scaffold to enable export
        </p>
      )}
    </div>
  );
}
