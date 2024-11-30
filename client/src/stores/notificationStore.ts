import { writable } from 'svelte/store';

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'info';
  message: string;
  timeout?: number;
}

function createNotificationStore() {
  const { subscribe, update } = writable<Notification[]>([]);

  return {
    subscribe,
    add: (notification: Omit<Notification, 'id'>) => {
      const id = Math.random().toString(36).slice(2);
      update(notifications => [...notifications, { ...notification, id }]);

      if (notification.timeout) {
        setTimeout(() => {
          update(notifications => notifications.filter(n => n.id !== id));
        }, notification.timeout);
      }
    },
    remove: (id: string) => {
      update(notifications => notifications.filter(n => n.id !== id));
    }
  };
}

export const notifications = createNotificationStore();