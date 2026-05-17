import { KeyboardEvent, useEffect, useRef, useState } from 'react';
import { Search } from 'lucide-react';
import { searchSuggestions } from '../lib/searchApi';

interface SearchInputProps {
  query: string;
  onQueryChange: (query: string) => void;
  onSearch: (queryOverride?: string) => void;
  isLoading: boolean;
  placeholder: string;
  inputClassName: string;
  buttonClassName: string;
  dropdownClassName?: string;
}

export function SearchInput({
  query,
  onQueryChange,
  onSearch,
  isLoading,
  placeholder,
  inputClassName,
  buttonClassName,
  dropdownClassName = '',
}: SearchInputProps) {
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [isFocused, setIsFocused] = useState(false);
  const [activeIndex, setActiveIndex] = useState(-1);
  const requestSequenceRef = useRef(0);

  useEffect(() => {
    const trimmedQuery = query.trim();
    if (trimmedQuery.length < 2) {
      setSuggestions([]);
      setActiveIndex(-1);
      return;
    }

    const requestSequence = ++requestSequenceRef.current;
    const timeoutId = window.setTimeout(async () => {
      try {
        const response = await searchSuggestions(trimmedQuery);
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        setSuggestions(response.suggestions);
        setActiveIndex(-1);
      } catch {
        if (requestSequence !== requestSequenceRef.current) {
          return;
        }
        setSuggestions([]);
        setActiveIndex(-1);
      }
    }, 180);

    return () => window.clearTimeout(timeoutId);
  }, [query]);

  const applySuggestion = (nextQuery: string) => {
    onQueryChange(nextQuery);
    setIsFocused(false);
    setActiveIndex(-1);
    onSearch(nextQuery);
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    const hasSuggestions = isFocused && suggestions.length > 0;

    if (event.key === 'ArrowDown' && hasSuggestions) {
      event.preventDefault();
      setActiveIndex((previous) => (previous + 1) % suggestions.length);
      return;
    }

    if (event.key === 'ArrowUp' && hasSuggestions) {
      event.preventDefault();
      setActiveIndex((previous) => (previous <= 0 ? suggestions.length - 1 : previous - 1));
      return;
    }

    if (event.key === 'Enter') {
      if (hasSuggestions && activeIndex >= 0 && suggestions[activeIndex]) {
        event.preventDefault();
        applySuggestion(suggestions[activeIndex]);
        return;
      }
      onSearch();
      setIsFocused(false);
      return;
    }

    if (event.key === 'Escape') {
      setIsFocused(false);
      setActiveIndex(-1);
    }
  };

  const showSuggestions = isFocused && suggestions.length > 0;

  return (
    <div className="relative w-full">
      <input
        type="text"
        value={query}
        onChange={(event) => onQueryChange(event.target.value)}
        onKeyDown={handleKeyDown}
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
        placeholder={placeholder}
        className={inputClassName}
        disabled={isLoading}
        autoComplete="off"
      />
      <button
        onClick={() => onSearch()}
        disabled={isLoading}
        className={buttonClassName}
        aria-label="Search"
      >
        {isLoading ? (
          <div className="h-5 w-5 animate-spin rounded-full border-2 border-white border-t-transparent" />
        ) : (
          <Search className="h-5 w-5" />
        )}
      </button>

      {showSuggestions && (
        <div
          className={`absolute left-0 right-0 z-40 mt-2 overflow-hidden rounded-3xl border border-gray-200 bg-white shadow-2xl dark:border-gray-700 dark:bg-gray-900 ${dropdownClassName}`.trim()}
        >
          <ul className="py-2">
            {suggestions.map((suggestion, index) => {
              const isActive = index === activeIndex;
              return (
                <li key={suggestion}>
                  <button
                    type="button"
                    onMouseDown={(event) => event.preventDefault()}
                    onClick={() => applySuggestion(suggestion)}
                    className={`flex w-full items-center gap-3 px-5 py-3 text-left text-sm transition ${
                      isActive
                        ? 'bg-[#1a5f3b]/8 text-[#1a5f3b] dark:bg-[#2d8a54]/15 dark:text-[#9be7b7]'
                        : 'text-gray-700 hover:bg-gray-50 dark:text-gray-200 dark:hover:bg-gray-800'
                    }`}
                  >
                    <Search className="h-4 w-4 shrink-0" />
                    <span className="truncate">{suggestion}</span>
                  </button>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
