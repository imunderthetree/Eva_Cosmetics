import { useState } from 'react';
import { ThemeProvider } from './contexts/ThemeContext';
import { Navbar } from './components/Navbar';
import { HomePage } from './components/HomePage';
import { SearchBar } from './components/SearchBar';
import {
  SearchOptionsPanel,
  defaultSearchOptions,
  SearchOptions,
} from './components/SearchOptionsPanel';
import { SearchResults } from './components/SearchResults';
import { SearchResult } from './components/ResultItem';
import { SearchExpansion, searchProducts } from './lib/searchApi';

function AppContent() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchOptions, setSearchOptions] = useState<SearchOptions>(defaultSearchOptions);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [resultCount, setResultCount] = useState(0);
  const [latencyMs, setLatencyMs] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [expansion, setExpansion] = useState<SearchExpansion | null>(null);

  const runSearch = async (nextQuery: string, nextOptions: SearchOptions) => {
    const trimmedQuery = nextQuery.trim();
    if (!trimmedQuery) return;

    setIsLoading(true);
    setHasSearched(true);
    setErrorMessage(null);

    try {
      const response = await searchProducts(trimmedQuery, nextOptions);
      setResults(response.results);
      setResultCount(response.count);
      setLatencyMs(response.latency_ms);
      setExpansion(response.expansion);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Search failed.';
      setResults([]);
      setResultCount(0);
      setLatencyMs(null);
      setErrorMessage(message);
      setExpansion(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = (nextQuery?: string) => {
    const queryToSearch = nextQuery ?? query;
    void runSearch(queryToSearch, searchOptions);
  };

  const handleOptionsChange = (nextOptions: SearchOptions) => {
    setSearchOptions(nextOptions);
    if (hasSearched && query.trim()) {
      void runSearch(query, nextOptions);
    }
  };

  const handleReset = () => {
    setHasSearched(false);
    setQuery('');
    setResults([]);
    setResultCount(0);
    setLatencyMs(null);
    setErrorMessage(null);
    setExpansion(null);
    setSearchOptions(defaultSearchOptions);
  };

  if (!hasSearched) {
    return (
      <HomePage
        query={query}
        onQueryChange={setQuery}
        onSearch={handleSearch}
      />
    );
  }

  return (
    <div className="flex min-h-screen flex-col bg-gray-50 dark:bg-gray-900">
      <Navbar onLogoClick={handleReset} />

      <main className="flex-1">
        <div className="mx-auto max-w-7xl px-6 py-8">
          <div className="mb-8">
            <SearchBar
              query={query}
              onQueryChange={setQuery}
              onSearch={handleSearch}
              expansion={expansion}
              isLoading={isLoading}
            />
          </div>

          <div className="flex flex-col gap-6 lg:flex-row">
            <SearchOptionsPanel
              options={searchOptions}
              onChange={handleOptionsChange}
              isLoading={isLoading}
            />

            <div className="flex-1">
              <SearchResults
                results={results}
                isLoading={isLoading}
                hasSearched={hasSearched}
                resultCount={resultCount}
                latencyMs={latencyMs}
                errorMessage={errorMessage}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
}
