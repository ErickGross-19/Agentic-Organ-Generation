'use client';

import { useState } from 'react';
import { ChevronDown, ChevronUp, FileJson, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface SpecViewerProps {
  spec: Record<string, any> | null;
  className?: string;
}

export function SpecViewer({ spec, className = '' }: SpecViewerProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  if (!spec) return null;

  const specJson = JSON.stringify(spec, null, 2);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(specJson);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      const textarea = document.createElement('textarea');
      textarea.value = specJson;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const componentCount = spec.components ? Object.keys(spec.components).length : 0;
  const domainCount = spec.domains ? Object.keys(spec.domains).length : 0;
  const policyCount = spec.policies ? Object.keys(spec.policies).length : 0;

  return (
    <div className={`border border-slate-200 dark:border-slate-700 rounded-lg bg-white dark:bg-slate-800 overflow-hidden ${className}`}>
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-slate-50 dark:hover:bg-slate-700/50"
      >
        <div className="flex items-center gap-2">
          <FileJson className="w-4 h-4 text-indigo-500" />
          <span className="text-sm font-medium">DesignSpec</span>
          <div className="flex gap-1.5 ml-2">
            {domainCount > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400">
                {domainCount} domain{domainCount !== 1 ? 's' : ''}
              </span>
            )}
            {componentCount > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-green-100 dark:bg-green-900/30 text-green-600 dark:text-green-400">
                {componentCount} component{componentCount !== 1 ? 's' : ''}
              </span>
            )}
            {policyCount > 0 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400">
                {policyCount} polic{policyCount !== 1 ? 'ies' : 'y'}
              </span>
            )}
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-4 h-4 text-slate-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-slate-400" />
        )}
      </button>

      {isExpanded && (
        <div className="border-t border-slate-200 dark:border-slate-700">
          <div className="flex justify-end p-1">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleCopy}
              className="h-6 text-xs"
            >
              {copied ? (
                <><Check className="w-3 h-3 mr-1" /> Copied</>
              ) : (
                <><Copy className="w-3 h-3 mr-1" /> Copy JSON</>
              )}
            </Button>
          </div>
          <pre className="p-3 pt-0 text-xs leading-relaxed overflow-x-auto max-h-[400px] overflow-y-auto text-slate-700 dark:text-slate-300 font-mono">
            {specJson}
          </pre>
        </div>
      )}
    </div>
  );
}
