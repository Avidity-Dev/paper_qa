import { writable } from 'svelte/store';

export interface ChatMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

function createChatStore() {
  const { subscribe, update } = writable<ChatMessage[]>([]);

  return {
    subscribe,
    addMessage: (type: 'user' | 'assistant', content: string) => {
      const message: ChatMessage = {
        id: crypto.randomUUID(),
        type,
        content,
        timestamp: new Date()
      };
      update(messages => [...messages, message]);
    },
    clear: () => update(() => [])
  };
}

export const chatMessages = createChatStore();