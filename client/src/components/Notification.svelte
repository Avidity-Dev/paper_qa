<script lang="ts">
  import { notifications } from '../stores/notificationStore';

  function getNotificationClass(type: string): string {
    switch (type) {
      case 'success': return 'bg-green-500';
      case 'error': return 'bg-red-500';
      case 'info': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  }
</script>

<div class="notifications-container">
  {#each $notifications as notification (notification.id)}
    <div class="notification {getNotificationClass(notification.type)}">
      <span>{notification.message}</span>
      <button
        class="close-button"
        on:click={() => notifications.remove(notification.id)}
      >
        ×
      </button>
    </div>
  {/each}
</div>

<style>
  .notifications-container {
    position: fixed;
    top: 1rem;
    right: 1rem;
    z-index: 1000;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-width: 400px;
  }

  .notification {
    padding: 1rem;
    border-radius: 4px;
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    animation: slideIn 0.3s ease-out;
  }

  .close-button {
    background: none;
    border: none;
    color: white;
    font-size: 1.5rem;
    cursor: pointer;
    padding: 0;
    line-height: 1;
  }

  @keyframes slideIn {
    from {
      transform: translateX(100%);
      opacity: 0;
    }
    to {
      transform: translateX(0);
      opacity: 1;
    }
  }
</style>