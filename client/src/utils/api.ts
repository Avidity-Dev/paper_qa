export async function uploadDocuments(files: FileList): Promise<Response> {
  const formData = new FormData();
  Array.from(files).forEach(file => formData.append('documents', file));
  
  return fetch('/api/upload', {
    method: 'POST',
    body: formData
  });
}

export async function processDocuments(): Promise<Response> {
  return fetch('/api/process', {
    method: 'POST'
  });
}

export async function fetchMetrics(): Promise<Response> {
  return fetch('/api/metrics');
}

export async function queryRepository(query: string): Promise<Response> {
  return fetch('/api/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ query })
  });
}