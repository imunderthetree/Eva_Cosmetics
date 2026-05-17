import { SearchResult, ResultItem } from './ResultItem';

interface SearchResultsProps {
  results: SearchResult[];
  isLoading: boolean;
  hasSearched: boolean;
  resultCount: number;
  latencyMs: number | null;
  errorMessage: string | null;
}

export function SearchResults({
  results,
  isLoading,
  hasSearched,
  resultCount,
  latencyMs,
  errorMessage,
}: SearchResultsProps) {
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center py-20">
        <div className="h-12 w-12 animate-spin rounded-full border-4 border-gray-200 border-t-[#1a5f3b] dark:border-gray-700 dark:border-t-[#2d8a54]" />
        <p className="mt-4 text-gray-600 dark:text-gray-400">Searching...</p>
      </div>
    );
  }

  if (!hasSearched) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="mb-4 text-6xl">Search</div>
        <h2 className="mb-2 text-xl text-gray-900 dark:text-white">Start Your Search</h2>
        <p className="text-gray-600 dark:text-gray-400">Enter a product query to search the Eva catalog</p>
      </div>
    );
  }

  if (errorMessage) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="mb-4 text-6xl">Error</div>
        <h2 className="mb-2 text-xl text-gray-900 dark:text-white">Search Unavailable</h2>
        <p className="max-w-xl text-gray-600 dark:text-gray-400">{errorMessage}</p>
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center">
        <div className="mb-4 text-6xl">No Results</div>
        <h2 className="mb-2 text-xl text-gray-900 dark:text-white">No Results Found</h2>
        <p className="text-gray-600 dark:text-gray-400">Try adjusting your search terms or search options</p>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-4 text-sm text-gray-600 dark:text-gray-400">
        Found {resultCount} result{resultCount !== 1 ? 's' : ''}
        {latencyMs !== null ? ` in ${latencyMs} ms` : ''}
      </div>
      <div className="space-y-4">
        {results.map((result) => (
          <ResultItem key={result.id} result={result} />
        ))}
      </div>
    </div>
  );
}
