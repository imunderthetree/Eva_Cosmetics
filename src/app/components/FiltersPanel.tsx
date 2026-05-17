import { FileText, Users, Mail, Package, BarChart } from 'lucide-react';

export type FilterType = 'documents' | 'employees' | 'emails' | 'products' | 'reports';

interface FiltersPanelProps {
  activeFilters: FilterType[];
  onFilterToggle: (filter: FilterType) => void;
}

const filterOptions = [
  { id: 'documents' as FilterType, label: 'Documents', icon: FileText },
  { id: 'employees' as FilterType, label: 'Employees', icon: Users },
  { id: 'emails' as FilterType, label: 'Emails', icon: Mail },
  { id: 'products' as FilterType, label: 'Products', icon: Package },
  { id: 'reports' as FilterType, label: 'Reports', icon: BarChart },
];

export function FiltersPanel({ activeFilters, onFilterToggle }: FiltersPanelProps) {
  return (
    <div className="w-64 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-800">
      <h3 className="mb-5 text-lg font-semibold text-gray-900 dark:text-white">Filters</h3>
      <div className="space-y-3">
        {filterOptions.map((filter) => {
          const Icon = filter.icon;
          const isActive = activeFilters.includes(filter.id);
          
          return (
            <label
              key={filter.id}
              className={`flex cursor-pointer items-center gap-3 rounded-lg p-3 transition-all ${
                isActive 
                  ? 'border-2 border-[#1a5f3b]/20 bg-[#1a5f3b]/5 dark:border-[#2d8a54]/30 dark:bg-[#2d8a54]/10' 
                  : 'border-2 border-transparent hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              <input
                type="checkbox"
                checked={isActive}
                onChange={() => onFilterToggle(filter.id)}
                className="h-5 w-5 cursor-pointer rounded accent-[#1a5f3b] dark:accent-[#2d8a54]"
              />
              <Icon className={`h-5 w-5 ${isActive ? 'text-[#1a5f3b] dark:text-[#2d8a54]' : 'text-gray-500 dark:text-gray-400'}`} />
              <span className={`text-sm font-medium ${isActive ? 'text-[#1a5f3b] dark:text-[#2d8a54]' : 'text-gray-700 dark:text-gray-300'}`}>
                {filter.label}
              </span>
            </label>
          );
        })}
      </div>
    </div>
  );
}