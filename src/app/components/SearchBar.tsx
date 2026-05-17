import { SearchExpansion } from '../lib/searchApi';
import { SearchInput } from './SearchInput';

interface SearchBarProps {
  query: string;
  onQueryChange: (query: string) => void;
  onSearch: (queryOverride?: string) => void;
  expansion?: SearchExpansion | null;
  isLoading: boolean;
}

export function SearchBar({
  query,
  onQueryChange,
  onSearch,
  expansion,
  isLoading,
}: SearchBarProps) {
  const isExpansionCurrent =
    Boolean(expansion) && expansion!.query.trim().toLowerCase() === query.trim().toLowerCase();
  const hasExpansionTerms = Boolean(isExpansionCurrent && expansion && expansion.expanded_terms.length > 0);
  const hasSuggestedQueries = Boolean(isExpansionCurrent && expansion && expansion.suggested_queries.length > 0);

  const applySuggestedQuery = (nextQuery: string) => {
    onQueryChange(nextQuery);
    onSearch(nextQuery);
  };

  return (
    <div className="w-full">
      <SearchInput
        query={query}
        onQueryChange={onQueryChange}
        onSearch={onSearch}
        isLoading={isLoading}
        placeholder="Search products, ingredients, and routines..."
        inputClassName="w-full rounded-full border-2 border-gray-200 bg-white py-4 pl-6 pr-14 text-lg text-gray-900 shadow-md outline-none transition-all placeholder:text-gray-500 hover:shadow-lg focus:border-[#1a5f3b] focus:shadow-lg focus:ring-4 focus:ring-[#1a5f3b]/10 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:placeholder:text-gray-400 dark:focus:border-[#2d8a54] dark:focus:ring-[#2d8a54]/10"
        buttonClassName="absolute right-2 top-1/2 -translate-y-1/2 rounded-full bg-[#1a5f3b] p-3 text-white transition-all hover:scale-105 hover:bg-[#154b2f] disabled:opacity-50 dark:bg-[#2d8a54] dark:hover:bg-[#236b42]"
      />

      {(hasExpansionTerms || hasSuggestedQueries) && (
        <div className="mt-4 space-y-3">
          {hasExpansionTerms && expansion && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-gray-500 dark:text-gray-400">Expanded with</span>
              {expansion.expanded_terms.map((term) => (
                <button
                  key={term}
                  onClick={() => applySuggestedQuery(`${query.trim()} ${term}`.trim())}
                  disabled={isLoading}
                  className="rounded-full border border-[#1a5f3b]/20 bg-[#1a5f3b]/5 px-3 py-1.5 text-sm text-[#1a5f3b] transition hover:border-[#1a5f3b]/40 hover:bg-[#1a5f3b]/10 disabled:opacity-50 dark:border-[#2d8a54]/30 dark:bg-[#2d8a54]/10 dark:text-[#8be2ab]"
                >
                  {term}
                </button>
              ))}
            </div>
          )}

          {hasSuggestedQueries && expansion && (
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-gray-500 dark:text-gray-400">Try</span>
              {expansion.suggested_queries.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => applySuggestedQuery(suggestion)}
                  disabled={isLoading}
                  className="rounded-full border border-gray-300 bg-white px-3 py-1.5 text-sm text-gray-700 transition hover:border-gray-400 hover:bg-gray-50 disabled:opacity-50 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200 dark:hover:border-gray-500"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
