import { create } from 'zustand';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  suggestions?: string[];
  paramsChange?: Record<string, any>;
}

interface ChatStore {
  messages: ChatMessage[];
  addMessage: (message: Omit<ChatMessage, 'id' | 'timestamp'>) => void;
  clearMessages: () => void;

  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;

  suggestions: string[];
  setSuggestions: (suggestions: string[]) => void;

  // Conversation history for API calls
  getConversationHistory: () => { role: 'user' | 'assistant'; content: string }[];
}

export const useChatStore = create<ChatStore>((set, get) => ({
  messages: [],
  addMessage: (message) => set((state) => ({
    messages: [
      ...state.messages,
      {
        ...message,
        id: crypto.randomUUID(),
        timestamp: new Date(),
      },
    ],
  })),
  clearMessages: () => set({ messages: [], suggestions: [] }),

  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),

  suggestions: [
    'Create a porous disc scaffold',
    'Make a blood vessel network',
    'Design a bone lattice structure',
  ],
  setSuggestions: (suggestions) => set({ suggestions }),

  getConversationHistory: () => {
    const { messages } = get();
    return messages.map(m => ({
      role: m.role,
      content: m.content,
    }));
  },
}));
