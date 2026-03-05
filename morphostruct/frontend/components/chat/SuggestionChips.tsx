'use client';

interface SuggestionChipsProps {
  suggestions: string[];
  onSelect: (suggestion: string) => void;
}

export function SuggestionChips({ suggestions, onSelect }: SuggestionChipsProps) {
  return (
    <div className="px-4 pb-2 flex flex-wrap gap-2">
      {suggestions.map((suggestion, index) => (
        <button
          key={index}
          onClick={() => onSelect(suggestion)}
          className="text-xs px-3 py-1.5 rounded-full
                     bg-slate-100 dark:bg-slate-700
                     text-slate-700 dark:text-slate-300
                     hover:bg-blue-100 dark:hover:bg-blue-900
                     hover:text-blue-700 dark:hover:text-blue-300
                     transition-colors"
        >
          {suggestion}
        </button>
      ))}
    </div>
  );
}
