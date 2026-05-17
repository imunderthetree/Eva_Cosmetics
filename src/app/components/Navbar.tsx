import logo from '../../assets/f61624f0391b936e3ebb19430610d556334b8ea2.png';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

interface NavbarProps {
  onLogoClick?: () => void;
}

export function Navbar({ onLogoClick }: NavbarProps) {
  const { theme, toggleTheme } = useTheme();

  return (
    <nav className="border-b border-gray-200 bg-white px-6 py-4 dark:border-gray-700 dark:bg-gray-900">
      <div className="mx-auto flex max-w-7xl items-center justify-between">
        <button
          onClick={onLogoClick}
          className="flex items-center gap-3 rounded-lg px-2 py-1 transition-all hover:opacity-70 hover:scale-[1.02] active:scale-95"
        >
          <img src={logo} alt="Eva Cosmetics" className="h-10" />
          <span className="text-2xl tracking-wide text-gray-900 dark:text-white" style={{ fontFamily: "'Playfair Display', serif" }}>
            EVA Cosmetics
          </span>
        </button>
        
        <button
          onClick={(e) => toggleTheme(e)}
          className="rounded-full p-2.5 text-gray-700 transition-all hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800"
          aria-label="Toggle theme"
        >
          {theme === 'light' ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
        </button>
      </div>
    </nav>
  );
}