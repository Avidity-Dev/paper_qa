<script lang="ts">
  import { onMount } from 'svelte';
  import { chatMessages } from '../stores/chatStore';
  import ChatMessage from './ChatMessage.svelte';
  import { queryRepository } from '../utils/api';
  import { Circle3 } from 'svelte-loading-spinners';

  let query = '';
  let isLoading = false;
  let chatContainer: HTMLDivElement;

  onMount(() => {
    scrollToBottom();
  });

  function scrollToBottom() {
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  async function handleSubmit() {
    if (!query.trim() || isLoading) return;

    const userQuery = query.trim();
    query = '';
    isLoading = true;

    chatMessages.addMessage('user', userQuery);
    scrollToBottom();

    try {
      const response = await queryRepository(userQuery);
      const data = await response.json();
      chatMessages.addMessage('assistant', data.response || 'I processed your query.');
    } catch (error) {
      chatMessages.addMessage('assistant', 'Sorry, I encountered an error processing your query.');
    } finally {
      isLoading = false;
      scrollToBottom();
    }
  }
</script>

<div class="chat-interface">
  <div class="chat-messages" bind:this={chatContainer}>
    {#each $chatMessages as message (message.id)}
      <ChatMessage {message} />
    {/each}
    {#if isLoading}
      <div class="loading-indicator">
        <Circle3 size="24" color="#0085B2" unit="px" duration="1s" />
      </div>
    {/if}
  </div>

  <div class="chat-input">
    <textarea
      bind:value={query}
      placeholder="Ask a question..."
      rows="3"
      on:keydown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleSubmit())}
    ></textarea>
    <button 
      on:click={handleSubmit}
      disabled={!query.trim() || isLoading}
    >
      Send
    </button>
  </div>
</div>

<style>
  .chat-interface {
    display: flex;
    flex-direction: column;
    height: calc(100vh - 200px);
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  .chat-messages {
    flex-grow: 1;
    overflow-y: auto;
    padding: 1rem;
  }

  .loading-indicator {
    display: flex;
    justify-content: center;
    padding: 1rem;
  }

  .chat-input {
    border-top: 1px solid #e2e8f0;
    padding: 1rem;
    display: flex;
    gap: 1rem;
    background-color: white;
  }

  textarea {
    flex-grow: 1;
    padding: 0.5rem;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    resize: none;
    font-family: inherit;
  }

  textarea:focus {
    outline: none;
    border-color: #0085B2;
  }

  button {
    padding: 0.5rem 1.5rem;
    background-color: #0085B2;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 500;
    transition: background-color 0.2s;
  }

  button:hover:not(:disabled) {
    background-color: #1F497D;
  }

  button:disabled {
    background-color: #9BB9D3;
    cursor: not-allowed;
  }
</style>