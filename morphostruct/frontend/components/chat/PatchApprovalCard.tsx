'use client';

import { useState } from 'react';
import { Check, X, ChevronDown, ChevronUp, FileCode } from 'lucide-react';
import { Button } from '@/components/ui/button';

export interface PatchData {
  patch_id: string;
  description?: string;
  diff?: Record<string, any>;
  [key: string]: any;
}

interface PatchApprovalCardProps {
  patch: PatchData;
  onApprove: (patchId: string) => void;
  onReject: (patchId: string, reason: string) => void;
  disabled?: boolean;
}

export function PatchApprovalCard({
  patch,
  onApprove,
  onReject,
  disabled = false,
}: PatchApprovalCardProps) {
  const [showDiff, setShowDiff] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [rejectReason, setRejectReason] = useState('');

  const handleReject = () => {
    if (rejecting) {
      onReject(patch.patch_id, rejectReason);
      setRejecting(false);
      setRejectReason('');
    } else {
      setRejecting(true);
    }
  };

  return (
    <div className="border border-amber-200 dark:border-amber-800 rounded-lg p-3 my-2 bg-amber-50 dark:bg-amber-900/20">
      <div className="flex items-start gap-2">
        <FileCode className="w-4 h-4 text-amber-600 dark:text-amber-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
            Patch Proposal
          </p>
          {patch.description && (
            <p className="text-xs text-amber-700 dark:text-amber-300 mt-1">
              {patch.description}
            </p>
          )}
        </div>
      </div>

      {patch.diff && (
        <button
          onClick={() => setShowDiff(!showDiff)}
          className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400 mt-2 hover:underline"
        >
          {showDiff ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          {showDiff ? 'Hide changes' : 'Show changes'}
        </button>
      )}

      {showDiff && patch.diff && (
        <pre className="mt-2 p-2 bg-slate-900 text-slate-100 rounded text-xs overflow-x-auto max-h-[200px] overflow-y-auto">
          {JSON.stringify(patch.diff, null, 2)}
        </pre>
      )}

      {rejecting && (
        <div className="mt-2">
          <input
            type="text"
            value={rejectReason}
            onChange={(e) => setRejectReason(e.target.value)}
            placeholder="Reason for rejection (optional)"
            className="w-full text-xs px-2 py-1 rounded border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-700"
          />
        </div>
      )}

      <div className="flex gap-2 mt-3">
        <Button
          size="sm"
          onClick={() => onApprove(patch.patch_id)}
          disabled={disabled}
          className="bg-green-600 hover:bg-green-700 text-white text-xs h-7"
        >
          <Check className="w-3 h-3 mr-1" />
          Approve
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={handleReject}
          disabled={disabled}
          className="text-xs h-7 border-red-300 text-red-600 hover:bg-red-50 dark:border-red-700 dark:text-red-400 dark:hover:bg-red-900/20"
        >
          <X className="w-3 h-3 mr-1" />
          {rejecting ? 'Confirm Reject' : 'Reject'}
        </Button>
        {rejecting && (
          <Button
            size="sm"
            variant="ghost"
            onClick={() => { setRejecting(false); setRejectReason(''); }}
            className="text-xs h-7"
          >
            Cancel
          </Button>
        )}
      </div>
    </div>
  );
}
