import { SearchOptions } from '../components/SearchOptionsPanel';
import { SearchResult } from '../components/ResultItem';

export interface SearchExpansion {
  query: string;
  base_terms: string[];
  synonym_terms: string[];
  feedback_terms: string[];
  embedding_terms: string[];
  expanded_terms: string[];
  suggested_queries: string[];
}

export interface SearchResponse {
  query: string;
  count: number;
  latency_ms: number;
  expansion: SearchExpansion;
  results: SearchResult[];
}

export interface SearchSuggestionsResponse {
  query: string;
  suggestions: string[];
}

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? '';

function buildSearchUrl(query: string, options: SearchOptions): string {
  const params = new URLSearchParams({
    q: query,
    top_k: String(options.topK),
    use_synonyms: String(options.useSynonyms),
    use_feedback: String(options.useFeedback),
    use_embeddings: String(options.useEmbeddings),
    embedding_backend: options.embeddingBackend,
  });

  const basePath = apiBaseUrl ? `${apiBaseUrl}/api/search` : '/api/search';
  return `${basePath}?${params.toString()}`;
}

export async function searchProducts(query: string, options: SearchOptions): Promise<SearchResponse> {
  const response = await fetch(buildSearchUrl(query, options));

  if (!response.ok) {
    let message = 'Search request failed.';
    try {
      const body = (await response.json()) as { error?: string };
      if (body.error) {
        message = body.error;
      }
    } catch {
      message = 'Search request failed.';
    }
    throw new Error(message);
  }

  return response.json() as Promise<SearchResponse>;
}

export async function searchSuggestions(query: string, limit = 8): Promise<SearchSuggestionsResponse> {
  const params = new URLSearchParams({
    q: query,
    limit: String(limit),
  });
  const basePath = apiBaseUrl ? `${apiBaseUrl}/api/suggest` : '/api/suggest';
  const response = await fetch(`${basePath}?${params.toString()}`);

  if (!response.ok) {
    throw new Error('Suggestion request failed.');
  }

  return response.json() as Promise<SearchSuggestionsResponse>;
}
