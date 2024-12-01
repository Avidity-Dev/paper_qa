<script lang="ts">
  import { createEventDispatcher } from "svelte";

  const dispatch = createEventDispatcher();
  let files: FileList | null = null;
  let uploadStatus: string = "";
  let isUploading: boolean = false;
  let isProcessing: boolean = false;

  function handleFileSelect(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files.length > 0) {
      files = input.files;
      uploadStatus = `${files.length} file(s) selected`;
    }
  }

  async function handleUpload() {
    if (!files || files.length === 0) {
      uploadStatus = "Please select at least one file";
      return;
    }

    isUploading = true;
    uploadStatus = "Uploading files...";

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append("docs", files[i]);
    }

    try {
      // Simulate upload for now - will be replaced with actual API call
      await new Promise((resolve) => setTimeout(resolve, 1000));
      uploadStatus = "Files uploaded successfully!";
      dispatch("uploadComplete", { files });
    } catch (error) {
      uploadStatus = "Error uploading files";
      console.error("Upload error:", error);
    } finally {
      isUploading = false;
    }
  }

  async function processDocuments() {
    if (!files || files.length === 0) {
      uploadStatus = "No files to process";
      return;
    }

    isProcessing = true;
    uploadStatus = "Processing documents...";

    try {
      // Simulate processing - will be replaced with actual API call
      await new Promise((resolve) => setTimeout(resolve, 1500));
      uploadStatus = "Documents processed successfully!";
      dispatch("processComplete");
    } catch (error) {
      uploadStatus = "Error processing documents";
      console.error("Processing error:", error);
    } finally {
      isProcessing = false;
    }
  }
</script>

<div class="upload-container">
  <h2>Document Upload</h2>

  <div class="upload-box">
    <input
      type="file"
      id="fileInput"
      multiple
      accept=".pdf,.txt,.doc,.docx"
      on:change={handleFileSelect}
      disabled={isUploading || isProcessing}
    />
    <label for="fileInput" class="file-label"> Choose Files </label>
  </div>

  {#if uploadStatus}
    <p class="status-message">{uploadStatus}</p>
  {/if}

  <div class="button-group">
    <button
      on:click={handleUpload}
      disabled={!files || isUploading || isProcessing}
      class="upload-button"
    >
      {isUploading ? "Uploading..." : "Upload Files"}
    </button>

    <button
      on:click={processDocuments}
      disabled={!files || isUploading || isProcessing}
      class="process-button"
    >
      {isProcessing ? "Processing..." : "Process Documents"}
    </button>
  </div>
</div>

<style>
  .upload-container {
    max-width: 600px;
    margin: 2rem auto;
    padding: 2rem;
    border-radius: 8px;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  }

  h2 {
    text-align: center;
    color: #333;
    margin-bottom: 1.5rem;
  }

  .upload-box {
    border: 2px dashed #ccc;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1rem;
    border-radius: 4px;
  }

  input[type="file"] {
    display: none;
  }

  .file-label {
    background-color: #4a90e2;
    color: white;
    padding: 0.8rem 1.5rem;
    border-radius: 4px;
    cursor: pointer;
    display: inline-block;
    transition: background-color 0.3s;
  }

  .file-label:hover {
    background-color: #357abd;
  }

  .status-message {
    text-align: center;
    margin: 1rem 0;
    color: #666;
  }

  .button-group {
    display: flex;
    gap: 1rem;
    justify-content: center;
  }

  button {
    padding: 0.8rem 1.5rem;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 500;
    transition: background-color 0.3s;
  }

  button:disabled {
    background-color: #ccc;
    cursor: not-allowed;
  }

  .upload-button {
    background-color: #4caf50;
    color: white;
  }

  .upload-button:hover:not(:disabled) {
    background-color: #45a049;
  }

  .process-button {
    background-color: #ff9800;
    color: white;
  }

  .process-button:hover:not(:disabled) {
    background-color: #f57c00;
  }
</style>
