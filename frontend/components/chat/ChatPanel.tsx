'use client';

import { useState } from 'react';
import { MessageList } from './MessageList';
import { InputBox } from './InputBox';
import { SuggestionChips } from './SuggestionChips';
import { PatchApprovalCard, PatchData } from './PatchApprovalCard';
import { PipelineProgress, PipelineProgressData } from './PipelineProgress';
import { MessageSquare, ChevronDown, ChevronUp, Workflow } from 'lucide-react';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  suggestions?: string[];
  patches?: PatchData[];
  pipelineProgress?: PipelineProgressData;
}

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
  suggestions?: string[];
  mode?: 'scaffold' | 'designspec';
  onApprovePatch?: (patchId: string) => void;
  onRejectPatch?: (patchId: string, reason: string) => void;
  pipelineProgress?: PipelineProgressData | null;
  isPipelineRunning?: boolean;
}

export function ChatPanel({
  messages,
  onSendMessage,
  isLoading = false,
  suggestions = [],
  mode = 'scaffold',
  onApprovePatch,
  onRejectPatch,
  pipelineProgress = null,
  isPipelineRunning = false,
}: ChatPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  const isDesignSpec = mode === 'designspec';

  return (
    <div className="flex flex-col bg-white dark:bg-slate-800 rounded-lg shadow-sm overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between p-4 border-b dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700/50"
      >
        <div className="flex items-center gap-2">
          {isDesignSpec ? (
            <Workflow className="w-5 h-5 text-purple-500" />
          ) : (
            <MessageSquare className="w-5 h-5 text-blue-500" />
          )}
          <h2 className="font-semibold">
            {isDesignSpec ? 'DesignSpec Agent' : 'Chat with AI'}
          </h2>
          {isDesignSpec && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-100 dark:bg-purple-900/30 text-purple-600 dark:text-purple-400 font-medium">
              DESIGNSPEC
            </span>
          )}
        </div>
        {isExpanded ? (
          <ChevronDown className="w-5 h-5 text-slate-400" />
        ) : (
          <ChevronUp className="w-5 h-5 text-slate-400" />
        )}
      </button>

      {isExpanded && (
        <>
          {/* Pipeline Progress */}
          {isDesignSpec && (pipelineProgress || isPipelineRunning) && (
            <div className="px-4 pt-2">
              <PipelineProgress
                progress={pipelineProgress}
                isRunning={isPipelineRunning}
              />
            </div>
          )}

          {/* Messages */}
          <div className="flex-1 min-h-[200px] max-h-[400px] overflow-y-auto">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center p-4">
                <p className="text-slate-500 text-sm text-center">
                  {isDesignSpec ? (
                    <>
                      Describe the organ or scaffold you want to design.<br />
                      The DesignSpec agent will propose changes to your spec.
                    </>
                  ) : (
                    <>
                      Describe the scaffold you want to create.<br />
                      For example: &quot;Create a porous disc with 200 micron pores&quot;
                    </>
                  )}
                </p>
              </div>
            ) : (
              <>
                <MessageList messages={messages} />
                {isDesignSpec && messages.length > 0 && (() => {
                  const lastMsg = messages[messages.length - 1];
                  if (lastMsg.role === 'assistant' && lastMsg.patches && lastMsg.patches.length > 0) {
                    return (
                      <div className="px-4 pb-2">
                        {lastMsg.patches.map((patch) => (
                          <PatchApprovalCard
                            key={patch.patch_id}
                            patch={patch}
                            onApprove={onApprovePatch || (() => {})}
                            onReject={onRejectPatch || (() => {})}
                            disabled={isLoading}
                          />
                        ))}
                      </div>
                    );
                  }
                  return null;
                })()}
              </>
            )}
          </div>

          {/* Suggestions */}
          {suggestions.length > 0 && (
            <SuggestionChips
              suggestions={suggestions}
              onSelect={onSendMessage}
            />
          )}

          {/* Input */}
          <InputBox
            onSend={onSendMessage}
            isLoading={isLoading}
            placeholder={
              isDesignSpec
                ? 'Describe what you want to design...'
                : 'Describe your scaffold...'
            }
          />
        </>
      )}
    </div>
  );
}
