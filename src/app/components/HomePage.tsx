import logo from '../../assets/f61624f0391b936e3ebb19430610d556334b8ea2.png';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';
import { SearchInput } from './SearchInput';

interface HomePageProps {
  query: string;
  onQueryChange: (query: string) => void;
  onSearch: (queryOverride?: string) => void;
}

export function HomePage({ query, onQueryChange, onSearch }: HomePageProps) {
  const { theme, toggleTheme } = useTheme();

  const runPresetSearch = (value: string) => {
    onQueryChange(value);
    onSearch(value);
  };

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gradient-to-b from-white to-gray-50 px-6 dark:from-gray-900 dark:to-gray-800">
      <button
        onClick={(e) => toggleTheme(e)}
        className="fixed right-8 top-8 rounded-full p-3 text-gray-700 transition-all hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
        aria-label="Toggle theme"
      >
        {theme === 'light' ? <Moon className="h-6 w-6" /> : <Sun className="h-6 w-6" />}
      </button>

      <div className="w-full max-w-2xl text-center">
        <div className="mb-12 flex flex-col items-center justify-center gap-3">
          <img src={logo} alt="Eva Cosmetics" className="h-24" />
          <h1 className="text-5xl tracking-wide text-gray-900 dark:text-white" style={{ fontFamily: "'Playfair Display', serif" }}>
            EVA Cosmetics
          </h1>
        </div>

        <p className="mb-10 text-lg text-gray-600 dark:text-gray-400" style={{ fontFamily: "'Montserrat', sans-serif" }}>
          Search across the Eva product catalog using the notebook ranking pipeline
        </p>

        <div className="mb-8">
          <SearchInput
            query={query}
            onQueryChange={onQueryChange}
            onSearch={onSearch}
            isLoading={false}
            placeholder="Search products, ingredients, and care types..."
            inputClassName="w-full rounded-full border border-gray-300 bg-white py-5 pl-6 pr-14 text-lg text-gray-900 shadow-lg outline-none transition-all placeholder:text-gray-500 hover:shadow-xl focus:border-[#1a5f3b] focus:shadow-xl focus:ring-2 focus:ring-[#1a5f3b]/20 dark:border-gray-600 dark:bg-gray-800 dark:text-white dark:placeholder:text-gray-400 dark:focus:border-[#2d8a54]"
            buttonClassName="absolute right-3 top-1/2 -translate-y-1/2 rounded-full bg-[#1a5f3b] p-3 text-white transition-all hover:scale-105 hover:bg-[#154b2f] disabled:opacity-50 dark:bg-[#2d8a54] dark:hover:bg-[#236b42]"
          />
        </div>

        <div className="flex flex-wrap justify-center gap-3">
          <button
            onClick={() => runPresetSearch('aloe vera shampoo')}
            className="rounded-full border border-gray-300 bg-white px-5 py-2.5 text-sm text-gray-700 transition-all hover:border-gray-400 hover:shadow-md dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:border-gray-500"
          >
            Aloe Vera Shampoo
          </button>
          <button
            onClick={() => runPresetSearch('beard oil')}
            className="rounded-full border border-gray-300 bg-white px-5 py-2.5 text-sm text-gray-700 transition-all hover:border-gray-400 hover:shadow-md dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:border-gray-500"
          >
            Beard Oil
          </button>
          <button
            onClick={() => runPresetSearch('baby wipes')}
            className="rounded-full border border-gray-300 bg-white px-5 py-2.5 text-sm text-gray-700 transition-all hover:border-gray-400 hover:shadow-md dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:border-gray-500"
          >
            Baby Wipes
          </button>
          <button
            onClick={() => runPresetSearch('hair gel')}
            className="rounded-full border border-gray-300 bg-white px-5 py-2.5 text-sm text-gray-700 transition-all hover:border-gray-400 hover:shadow-md dark:border-gray-600 dark:bg-gray-800 dark:text-gray-300 dark:hover:border-gray-500"
          >
            Hair Gel
          </button>
        </div>
      </div>
    </div>
  );
}
