import { writable } from 'svelte/store';

export interface RepositoryMetrics {
  totalDocuments: number;
  totalSize: string;
  lastProcessedDoc: string;
  lastUpdated: string;
}

const defaultMetrics: RepositoryMetrics = {
  totalDocuments: 0,
  totalSize: '0 KB',
  lastProcessedDoc: 'None',
  lastUpdated: 'Never'
};

export const repositoryMetrics = writable<RepositoryMetrics>(defaultMetrics);