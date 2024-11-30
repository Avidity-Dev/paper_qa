<script lang="ts">
  import Navbar from './components/Navbar.svelte';
  import ChatInterface from './components/ChatInterface.svelte';
  import MetricsSummary from './components/MetricsSummary.svelte';
  import DocumentUpload from './lib/components/DocumentUpload.svelte';
  import Notification from './components/Notification.svelte';
  import { notifications } from './stores/notificationStore';

  function handleUploadComplete(event: CustomEvent) {
    notifications.add({
      type: 'success',
      message: `Successfully uploaded ${event.detail.files.length} file(s)`,
      timeout: 3000
    });
  }

  function handleProcessComplete() {
    notifications.add({
      type: 'success',
      message: 'Documents processed successfully',
      timeout: 3000
    });
  }
</script>

<Navbar />

<main>
  <Notification />
  
  <div class="layout">
    <div class="main-content">
      <ChatInterface />
    </div>
    
    <aside class="sidebar">
      <MetricsSummary />
      <DocumentUpload 
        on:uploadComplete={handleUploadComplete}
        on:processComplete={handleProcessComplete}
      />
    </aside>
  </div>
</main>

<style>
  main {
    min-height: calc(100vh - 64px);
    background-color: #f0f2f5;
    padding: 2rem;
  }

  .layout {
    max-width: 1200px;
    margin: 0 auto;
    display: grid;
    grid-template-columns: 1fr 350px;
    gap: 2rem;
  }

  .main-content {
    display: flex;
    flex-direction: column;
  }

  .sidebar {
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }

  @media (max-width: 768px) {
    .layout {
      grid-template-columns: 1fr;
    }
  }
</style>