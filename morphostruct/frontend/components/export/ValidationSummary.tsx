'use client';

import { CheckCircle, AlertTriangle, XCircle } from 'lucide-react';
import { cn } from '@/lib/utils';

interface ValidationSummaryProps {
  validation: {
    is_valid: boolean;
    is_printable: boolean;
    warnings: string[];
    errors: string[];
  };
}

export function ValidationSummary({ validation }: ValidationSummaryProps) {
  const { is_valid, is_printable, warnings, errors } = validation;

  return (
    <div className="space-y-2">
      {/* Status badges */}
      <div className="flex gap-2">
        <div
          className={cn(
            'flex items-center gap-1 px-2 py-1 rounded-full text-xs',
            is_valid
              ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
              : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
          )}
        >
          {is_valid ? (
            <CheckCircle className="w-3 h-3" />
          ) : (
            <XCircle className="w-3 h-3" />
          )}
          {is_valid ? 'Valid' : 'Invalid'}
        </div>

        <div
          className={cn(
            'flex items-center gap-1 px-2 py-1 rounded-full text-xs',
            is_printable
              ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400'
              : 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
          )}
        >
          {is_printable ? (
            <CheckCircle className="w-3 h-3" />
          ) : (
            <AlertTriangle className="w-3 h-3" />
          )}
          {is_printable ? 'Printable' : 'Review Needed'}
        </div>
      </div>

      {/* Errors */}
      {errors.length > 0 && (
        <div className="bg-red-50 dark:bg-red-900/20 rounded p-2">
          <div className="text-xs font-medium text-red-700 dark:text-red-400 mb-1">
            Errors
          </div>
          <ul className="text-xs text-red-600 dark:text-red-400 space-y-0.5">
            {errors.map((error, i) => (
              <li key={i}>• {error}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="bg-yellow-50 dark:bg-yellow-900/20 rounded p-2">
          <div className="text-xs font-medium text-yellow-700 dark:text-yellow-400 mb-1">
            Warnings
          </div>
          <ul className="text-xs text-yellow-600 dark:text-yellow-400 space-y-0.5">
            {warnings.map((warning, i) => (
              <li key={i}>• {warning}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
