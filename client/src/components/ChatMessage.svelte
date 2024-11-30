<script lang="ts">
  import type { ChatMessage } from '../stores/chatStore';
  
  export let message: ChatMessage;
  
  $: formattedTime = new Intl.DateTimeFormat('en', {
    hour: '2-digit',
    minute: '2-digit'
  }).format(message.timestamp);
</script>

<div class="message {message.type}">
  <div class="message-content">
    <div class="message-text">{message.content}</div>
    <div class="message-time">{formattedTime}</div>
  </div>
</div>

<style>
  .message {
    display: flex;
    margin: 0.5rem 0;
    padding: 0.5rem;
  }

  .message.user {
    justify-content: flex-end;
  }

  .message-content {
    max-width: 80%;
    padding: 0.8rem;
    border-radius: 8px;
  }

  .user .message-content {
    background-color: #0085B2;
    color: white;
    border-radius: 8px 8px 0 8px;
  }

  .assistant .message-content {
    background-color: #f0f2f5;
    color: #1F497D;
    border-radius: 8px 8px 8px 0;
  }

  .message-time {
    font-size: 0.75rem;
    margin-top: 0.25rem;
    opacity: 0.8;
  }

  .message-text {
    line-height: 1.4;
  }
</style>