'use client';

import { useState } from 'react';
import { MessageList } from './MessageList';
import { InputBox } from './InputBox';
import { SuggestionChips } from './SuggestionChips';
import { MessageSquare, ChevronDown, ChevronUp } from 'lucide-react';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  suggestions?: string[];
}

interface ChatPanelProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
  isLoading?: boolean;
  suggestions?: string[];
}

export function ChatPanel({
  messages,
  onSendMessage,
  isLoading = false,
  suggestions = [],
}: ChatPanelProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  return (
    <div className="flex flex-col bg-white dark:bg-slate-800 rounded-lg shadow-sm overflow-hidden">
      {/* Header */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center justify-between p-4 border-b dark:border-slate-700 hover:bg-slate-50 dark:hover:bg-slate-700/50"
      >
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-blue-500" />
          <h2 className="font-semibold">Chat with AI</h2>
        </div>
        {isExpanded ? (
          <ChevronDown className="w-5 h-5 text-slate-400" />
        ) : (
          <ChevronUp className="w-5 h-5 text-slate-400" />
        )}
      </button>

      {isExpanded && (
        <>
          {/* Messages */}
          <div className="flex-1 min-h-[200px] max-h-[400px] overflow-y-auto">
            {messages.length === 0 ? (
              <div className="h-full flex items-center justify-center p-4">
                <p className="text-slate-500 text-sm text-center">
                  Describe the scaffold you want to create.<br />
                  For example: "Create a porous disc with 200 micron pores"
                </p>
              </div>
            ) : (
              <MessageList messages={messages} />
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
          />
        </>
      )}
    </div>
  );
}
