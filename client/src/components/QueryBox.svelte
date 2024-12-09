<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { Circle3 } from 'svelte-loading-spinners';

  const dispatch = createEventDispatcher();
  
  let query = '';
  let isLoading = false;

  async function handleSubmit() {
    if (!query.trim()) return;
    
    isLoading = true;
    dispatch('query', { query });
    
    try {
      // Actual API integration will be added later
      await new Promise(resolve => setTimeout(resolve, 1000));
      dispatch('queryComplete', { query });
    } catch (error) {
      dispatch('queryError', { error });
    } finally {
      isLoading = false;
    }
  }
</script>

<div class="query-container">
  <textarea
    bind:value={query}
    placeholder="Enter your query here..."
    disabled={isLoading}
    on:keydown={e => e.key === 'Enter' && !e.shiftKey && handleSubmit()}
  ></textarea>
  
  <button 
    on:click={handleSubmit}
    disabled={!query.trim() || isLoading}
    class="query-button"
  >
    {#if isLoading}
      <Circle3 size="24" color="#ffffff" unit="px" duration="1s" />
    {:else}
      Search
    {/if}
  </button>
</div>

<style>
  .query-container {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    width: 100%;
    max-width: 600px;
    margin: 0 auto;
  }

  textarea {
    width: 100%;
    height: 120px;
    padding: 1rem;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    resize: vertical;
    font-size: 1rem;
    line-height: 1.5;
  }

  textarea:focus {
    outline: none;
    border-color: #4a90e2;
  }

  .query-button {
    padding: 0.8rem 1.5rem;
    background-color: #4a90e2;
    color: white;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 500;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 48px;
  }

  .query-button:hover:not(:disabled) {
    background-color: #357abd;
  }

  .query-button:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }
</style>