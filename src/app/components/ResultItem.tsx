import { Calendar, Tag } from 'lucide-react';

export interface SearchResult {
  id: string;
  title: string;
  category: string;
  snippet: string;
  metadata: string;
  date?: string;
  score?: number;
  price?: string;
  stock?: string;
  type?: string;
  features?: string;
}

interface ResultItemProps {
  result: SearchResult;
}

export function ResultItem({ result }: ResultItemProps) {
  const categoryLabel = result.category === 'products' ? 'Product' : result.category;

  return (
    <div className="group rounded-xl border border-gray-200 bg-white p-6 shadow-sm transition-all hover:border-[#1a5f3b]/30 hover:shadow-md dark:border-gray-700 dark:bg-gray-800 dark:hover:border-[#2d8a54]/30">
      <div className="mb-3 flex items-start justify-between gap-4">
        <h3 className="flex-1 text-lg font-medium text-gray-900 transition-colors group-hover:text-[#1a5f3b] dark:text-white dark:group-hover:text-[#2d8a54]">
          {result.title}
        </h3>
        <div className="flex items-center gap-2">
          {typeof result.score === 'number' && (
            <span className="rounded-full bg-[#1a5f3b]/10 px-3 py-1.5 text-xs font-medium text-[#1a5f3b] dark:bg-[#2d8a54]/15 dark:text-[#7ed69f]">
              Score {result.score.toFixed(3)}
            </span>
          )}
          <span className="rounded-full bg-orange-100 px-3 py-1.5 text-xs font-medium text-orange-800">
            {categoryLabel}
          </span>
        </div>
      </div>

      <p className="mb-4 text-sm leading-relaxed text-gray-600 dark:text-gray-400">{result.snippet}</p>

      <div className="flex flex-wrap items-center gap-4 text-xs text-gray-500 dark:text-gray-500">
        <div className="flex items-center gap-1.5">
          <Tag className="h-3.5 w-3.5" />
          <span>{result.metadata}</span>
        </div>
        {result.date && (
          <div className="flex items-center gap-1.5">
            <Calendar className="h-3.5 w-3.5" />
            <span>{result.date}</span>
          </div>
        )}
      </div>
    </div>
  );
}
