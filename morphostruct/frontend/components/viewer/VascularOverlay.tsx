'use client';

import { useScaffoldStore } from '@/lib/store/scaffoldStore';
import { ScaffoldType } from '@/lib/types/scaffolds';
import { Activity, GitBranch, Layers, Ruler } from 'lucide-react';

/**
 * VascularOverlay - Displays network statistics for vascular scaffolds
 *
 * Shows additional information for Space Colonization and Bifurcating Tree
 * scaffold types, including node count, segment count, and network metrics.
 */
export function VascularOverlay() {
  const scaffoldType = useScaffoldStore((state) => state.scaffoldType);
  const stats = useScaffoldStore((state) => state.stats);

  // Only show for vascular scaffold types
  const isVascularType =
    scaffoldType === ScaffoldType.SPACE_COLONIZATION ||
    scaffoldType === ScaffoldType.BIFURCATING_TREE;

  if (!isVascularType || !stats) {
    return null;
  }

  // Cast stats to include vascular-specific fields
  const vascularStats = stats as any;

  // Extract vascular-specific stats (if available)
  const networkNodes = vascularStats.network_nodes ?? 0;
  const networkSegments = vascularStats.network_segments ?? 0;
  const totalLength = vascularStats.total_length_m ?? 0;
  const minRadius = vascularStats.min_radius_m ?? 0;
  const maxRadius = vascularStats.max_radius_m ?? 0;
  const terminalCount = vascularStats.terminal_count ?? 0;
  const numInlets = vascularStats.num_inlets ?? 1;
  const branchingLevels = vascularStats.branching_levels ?? 0;
  const branchesPerNode = vascularStats.branches_per_node ?? 0;

  // Format numbers
  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`;
    } else if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toLocaleString();
  };

  const formatMicrometers = (meters: number) => {
    return `${(meters * 1000000).toFixed(0)} μm`;
  };

  const formatMillimeters = (meters: number) => {
    return `${(meters * 1000).toFixed(1)} mm`;
  };

  return (
    <div className="absolute top-4 left-4 bg-white/95 dark:bg-slate-800/95 backdrop-blur-sm p-4 rounded-lg shadow-lg border border-slate-200 dark:border-slate-700 min-w-[240px] z-10">
      {/* Header */}
      <div className="flex items-center gap-2 mb-3 pb-2 border-b border-slate-200 dark:border-slate-700">
        <Activity className="h-4 w-4 text-blue-600 dark:text-blue-400" />
        <h3 className="font-semibold text-sm text-slate-900 dark:text-slate-100">
          Vascular Network
        </h3>
      </div>

      {/* Network Structure Stats */}
      <div className="space-y-2 text-xs">
        {/* Nodes */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
            <GitBranch className="h-3.5 w-3.5" />
            <span>Nodes:</span>
          </div>
          <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
            {formatNumber(networkNodes)}
          </span>
        </div>

        {/* Segments */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
            <Layers className="h-3.5 w-3.5" />
            <span>Segments:</span>
          </div>
          <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
            {formatNumber(networkSegments)}
          </span>
        </div>

        {/* Total Length (if available) */}
        {totalLength > 0 && (
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5 text-slate-600 dark:text-slate-400">
              <Ruler className="h-3.5 w-3.5" />
              <span>Total Length:</span>
            </div>
            <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
              {formatMillimeters(totalLength)}
            </span>
          </div>
        )}

        {/* Terminals (if available) */}
        {terminalCount > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-slate-600 dark:text-slate-400">Terminals:</span>
            <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
              {formatNumber(terminalCount)}
            </span>
          </div>
        )}

        {/* Divider */}
        <div className="border-t border-slate-200 dark:border-slate-700 my-2" />

        {/* Radius Range (if available) */}
        {maxRadius > 0 && (
          <>
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Min Radius:</span>
              <span className="font-mono text-[10px] text-slate-900 dark:text-slate-100">
                {formatMicrometers(minRadius)}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Max Radius:</span>
              <span className="font-mono text-[10px] text-slate-900 dark:text-slate-100">
                {formatMicrometers(maxRadius)}
              </span>
            </div>
          </>
        )}

        {/* Type-specific info */}
        {scaffoldType === ScaffoldType.SPACE_COLONIZATION && numInlets > 0 && (
          <>
            <div className="border-t border-slate-200 dark:border-slate-700 my-2" />
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Inlets:</span>
              <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
                {numInlets}
              </span>
            </div>
          </>
        )}

        {scaffoldType === ScaffoldType.BIFURCATING_TREE && branchingLevels > 0 && (
          <>
            <div className="border-t border-slate-200 dark:border-slate-700 my-2" />
            <div className="flex items-center justify-between">
              <span className="text-slate-600 dark:text-slate-400">Levels:</span>
              <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
                {branchingLevels}
              </span>
            </div>
            {branchesPerNode > 0 && (
              <div className="flex items-center justify-between">
                <span className="text-slate-600 dark:text-slate-400">Branching:</span>
                <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
                  {branchesPerNode}-way
                </span>
              </div>
            )}
          </>
        )}

        {/* Mesh Stats */}
        <div className="border-t border-slate-200 dark:border-slate-700 my-2" />
        <div className="flex items-center justify-between">
          <span className="text-slate-600 dark:text-slate-400">Triangles:</span>
          <span className="font-mono font-medium text-slate-900 dark:text-slate-100">
            {formatNumber(stats.triangle_count)}
          </span>
        </div>

        {stats.volume_mm3 > 0 && (
          <div className="flex items-center justify-between">
            <span className="text-slate-600 dark:text-slate-400">Volume:</span>
            <span className="font-mono text-[10px] text-slate-900 dark:text-slate-100">
              {stats.volume_mm3.toFixed(2)} mm³
            </span>
          </div>
        )}
      </div>

      {/* Algorithm Badge */}
      <div className="mt-3 pt-2 border-t border-slate-200 dark:border-slate-700">
        <div className="inline-flex items-center gap-1.5 px-2 py-1 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-[10px] font-medium">
          <Activity className="h-3 w-3" />
          {scaffoldType === ScaffoldType.SPACE_COLONIZATION ? 'Space Colonization' : 'Bifurcating Tree'}
        </div>
      </div>
    </div>
  );
}
