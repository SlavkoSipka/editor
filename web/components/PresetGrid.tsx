"use client";

import type { Preset } from "@/lib/api";

type Props = {
  presets: Preset[];
  selected: string | null;
  onSelect: (id: string) => void;
};

export function PresetGrid({ presets, selected, onSelect }: Props) {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
      {presets.map((p) => {
        const isActive = p.id === selected;
        return (
          <button
            key={p.id}
            type="button"
            onClick={() => onSelect(p.id)}
            className={`p-4 rounded-xl border text-left transition
              ${
                isActive
                  ? "border-indigo-400 bg-indigo-500/10"
                  : "border-neutral-800 hover:border-neutral-700 bg-neutral-900/40"
              }
            `}
          >
            <div className="font-medium text-sm mb-1">{p.name}</div>
            <div className="text-xs text-neutral-500 line-clamp-2">{p.description}</div>
          </button>
        );
      })}
    </div>
  );
}
