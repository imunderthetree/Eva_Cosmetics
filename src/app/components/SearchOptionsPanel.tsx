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
  topK: 10,
  useSynonyms: true,
  useFeedback: true,
  useEmbeddings: false,
  embeddingBackend: 'bert',
};

export function SearchOptionsPanel({ options, onChange, isLoading }: SearchOptionsPanelProps) {
  const updateOption = <K extends keyof SearchOptions>(key: K, value: SearchOptions[K]) => {
    onChange({ ...options, [key]: value });
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
            max={20}
            value={options.topK}
            disabled={isLoading}
            onChange={(event) =>
              updateOption('topK', Math.max(1, Math.min(20, Number(event.target.value) || 1)))
            }
            className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 outline-none transition focus:border-[#1a5f3b] focus:ring-2 focus:ring-[#1a5f3b]/20 disabled:opacity-60 dark:border-gray-600 dark:bg-gray-900 dark:text-white dark:focus:border-[#2d8a54]"
          />
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

        <label className="flex items-center gap-3 rounded-lg border border-gray-200 px-3 py-3 dark:border-gray-700">
          <input
            type="checkbox"
            checked={options.useEmbeddings}
            disabled={isLoading}
            onChange={(event) => updateOption('useEmbeddings', event.target.checked)}
            className="h-4 w-4 rounded accent-[#1a5f3b] dark:accent-[#2d8a54]"
          />
          <span className="text-sm text-gray-700 dark:text-gray-300">Embedding expansion</span>
        </label>

        {options.useEmbeddings && (
          <label className="block">
            <span className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">Embedding backend</span>
            <select
              value={options.embeddingBackend}
              disabled={isLoading}
              onChange={(event) => updateOption('embeddingBackend', event.target.value as EmbeddingBackend)}
              className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 outline-none transition focus:border-[#1a5f3b] focus:ring-2 focus:ring-[#1a5f3b]/20 disabled:opacity-60 dark:border-gray-600 dark:bg-gray-900 dark:text-white dark:focus:border-[#2d8a54]"
            >
              <option value="bert">BERT</option>
              <option value="elmo">ELMo</option>
              <option value="rnn">RNN</option>
            </select>
          </label>
        )}
      </div>
    </aside>
  );
}
