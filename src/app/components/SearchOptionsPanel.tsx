export type EmbeddingBackend = 'bert' | 'elmo' | 'rnn';

export interface SearchOptions {
  topK: number;
  useSynonyms: boolean;
  useFeedback: boolean;
  useEmbeddings: boolean;
  embeddingBackend: EmbeddingBackend;
}

interface SearchOptionsPanelProps {
  options: SearchOptions;
  onChange: (options: SearchOptions) => void;
  isLoading: boolean;
}

export const defaultSearchOptions: SearchOptions = {
  topK: 25,
  useSynonyms: true,
  useFeedback: true,
  useEmbeddings: true,
  embeddingBackend: 'bert',
};

export function SearchOptionsPanel({ options, onChange, isLoading }: SearchOptionsPanelProps) {
  const updateOption = <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => {
    onChange({ ...options, [key]: value, useEmbeddings: true, embeddingBackend: 'bert' });
  };

  return (
    <aside className="w-full rounded-xl border border-gray-200 bg-white p-6 shadow-sm lg:w-72 dark:border-gray-700 dark:bg-gray-800">
      <h3 className="mb-5 text-lg font-semibold text-gray-900 dark:text-white">Search Options</h3>

      <div className="space-y-4">
        <label className="block">
          <span className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">Top results</span>
          <input
            type="number"
            min={1}
            max={50}
            value={options.topK}
            disabled={isLoading}
            onChange={(event) =>
              updateOption('topK', Math.max(1, Math.min(50, Number(event.target.value) || 1)))
            }
            className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 outline-none transition focus:border-[#1a5f3b] focus:ring-2 focus:ring-[#1a5f3b]/20 disabled:opacity-60 dark:border-gray-600 dark:bg-gray-900 dark:text-white dark:focus:border-[#2d8a54]"
          />
          <span className="mt-1 block text-xs text-gray-500 dark:text-gray-400">Retrieve up to 50 ranked products.</span>
        </label>

        <label className="flex items-center gap-3 rounded-lg border border-gray-200 px-3 py-3 dark:border-gray-700">
          <input
            type="checkbox"
            checked={options.useSynonyms}
            disabled={isLoading}
            onChange={(event) => updateOption('useSynonyms', event.target.checked)}
            className="h-4 w-4 rounded accent-[#1a5f3b] dark:accent-[#2d8a54]"
          />
          <span className="text-sm text-gray-700 dark:text-gray-300">Synonym expansion</span>
        </label>

        <label className="flex items-center gap-3 rounded-lg border border-gray-200 px-3 py-3 dark:border-gray-700">
          <input
            type="checkbox"
            checked={options.useFeedback}
            disabled={isLoading}
            onChange={(event) => updateOption('useFeedback', event.target.checked)}
            className="h-4 w-4 rounded accent-[#1a5f3b] dark:accent-[#2d8a54]"
          />
          <span className="text-sm text-gray-700 dark:text-gray-300">Relevance feedback</span>
        </label>

        <div className="flex items-center gap-3 rounded-lg border border-[#1a5f3b]/20 bg-[#1a5f3b]/5 px-3 py-3 dark:border-[#2d8a54]/30 dark:bg-[#2d8a54]/10">
          <input
            type="checkbox"
            checked
            disabled
            className="h-4 w-4 rounded accent-[#1a5f3b] dark:accent-[#2d8a54]"
          />
          <div>
            <span className="block text-sm font-medium text-[#1a5f3b] dark:text-[#8be2ab]">AI expansion enabled</span>
            <span className="block text-xs text-gray-500 dark:text-gray-400">Vector query expansion is always applied.</span>
          </div>
        </div>
      </div>
    </aside>
  );
}
