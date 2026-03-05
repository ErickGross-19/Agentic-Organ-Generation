'use client';

import { Eye, Grid3X3, RotateCw, Box, Layers } from 'lucide-react';

interface ViewControlsProps {
  viewMode: 'normal' | 'wireframe' | 'xray';
  onViewModeChange: (mode: 'normal' | 'wireframe' | 'xray') => void;
  showGrid: boolean;
  onShowGridChange: (show: boolean) => void;
  autoRotate: boolean;
  onAutoRotateChange: (rotate: boolean) => void;
}

export function ViewControls({
  viewMode,
  onViewModeChange,
  showGrid,
  onShowGridChange,
  autoRotate,
  onAutoRotateChange,
}: ViewControlsProps) {
  return (
    <div className="absolute top-2 right-2 z-10 flex flex-col gap-1">
      {/* View mode buttons */}
      <div className="flex bg-slate-800/90 rounded-md p-1 gap-1">
        <button
          onClick={() => onViewModeChange('normal')}
          className={`p-2 rounded ${viewMode === 'normal' ? 'bg-blue-600' : 'hover:bg-slate-700'}`}
          title="Solid view"
        >
          <Box className="w-4 h-4 text-white" />
        </button>
        <button
          onClick={() => onViewModeChange('wireframe')}
          className={`p-2 rounded ${viewMode === 'wireframe' ? 'bg-blue-600' : 'hover:bg-slate-700'}`}
          title="Wireframe view"
        >
          <Layers className="w-4 h-4 text-white" />
        </button>
        <button
          onClick={() => onViewModeChange('xray')}
          className={`p-2 rounded ${viewMode === 'xray' ? 'bg-blue-600' : 'hover:bg-slate-700'}`}
          title="X-ray view"
        >
          <Eye className="w-4 h-4 text-white" />
        </button>
      </div>

      {/* Toggle buttons */}
      <div className="flex bg-slate-800/90 rounded-md p-1 gap-1">
        <button
          onClick={() => onShowGridChange(!showGrid)}
          className={`p-2 rounded ${showGrid ? 'bg-blue-600' : 'hover:bg-slate-700'}`}
          title="Toggle grid"
        >
          <Grid3X3 className="w-4 h-4 text-white" />
        </button>
        <button
          onClick={() => onAutoRotateChange(!autoRotate)}
          className={`p-2 rounded ${autoRotate ? 'bg-blue-600' : 'hover:bg-slate-700'}`}
          title="Auto-rotate"
        >
          <RotateCw className="w-4 h-4 text-white" />
        </button>
      </div>
    </div>
  );
}
