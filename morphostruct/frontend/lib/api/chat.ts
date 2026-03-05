import { apiRequest } from './client';
import { ScaffoldType } from '../types/scaffolds';

interface ChatHistoryMessage {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatRequest {
  message: string;
  conversation_history: ChatHistoryMessage[];
  current_params?: Record<string, any>;
}

interface ChatResponse {
  message: string;
  action: 'chat' | 'generate' | 'clarify';
  suggested_params?: {
    type: ScaffoldType;
    [key: string]: any;
  };
  suggestions: string[];
}

export async function sendChatMessage(
  message: string,
  conversationHistory: ChatHistoryMessage[],
  currentParams?: Record<string, any>
): Promise<ChatResponse> {
  return apiRequest<ChatResponse>('/api/chat', {
    method: 'POST',
    body: JSON.stringify({
      message,
      conversation_history: conversationHistory,
      current_params: currentParams,
    }),
  });
}

export async function getChatStatus(): Promise<{ llm_available: boolean }> {
  return apiRequest('/api/chat/status');
}
