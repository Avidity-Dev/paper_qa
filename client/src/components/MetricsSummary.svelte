<script lang="ts">
  import { repositoryMetrics } from '../stores/repositoryStore';
</script>

<div class="metrics-container">
  <h3>Repository Metrics</h3>
  
  <div class="metrics-grid">
    <div class="metric-item">
      <span class="metric-label">Total Documents</span>
      <span class="metric-value">{$repositoryMetrics.totalDocuments}</span>
    </div>
    
    <div class="metric-item">
      <span class="metric-label">Total Size</span>
      <span class="metric-value">{$repositoryMetrics.totalSize}</span>
    </div>
    
    <div class="metric-item">
      <span class="metric-label">Last Processed</span>
      <span class="metric-value">{$repositoryMetrics.lastProcessedDoc}</span>
    </div>
    
    <div class="metric-item">
      <span class="metric-label">Last Updated</span>
      <span class="metric-value">{$repositoryMetrics.lastUpdated}</span>
    </div>
  </div>
</div>

<style>
  .metrics-container {
    background-color: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  h3 {
    margin: 0 0 1rem 0;
    color: #333;
    font-size: 1.25rem;
  }

  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
  }

  .metric-item {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .metric-label {
    font-size: 0.875rem;
    color: #666;
  }

  .metric-value {
    font-size: 1.125rem;
    font-weight: 500;
    color: #333;
  }
</style>