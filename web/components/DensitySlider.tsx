"use client";

type Props = {
  value: number; // 0.0 - 1.0
  presetDefault: number; // the preset's default, for the marker
  onChange: (v: number) => void;
};

export function DensitySlider({ value, presetDefault, onChange }: Props) {
  const pct = Math.round(value * 100);

  const label =
    value < 0.35
      ? "Minimal"
      : value < 0.55
        ? "Balanced"
        : value < 0.75
          ? "Lively"
          : "Packed";

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm text-neutral-400">Sound density</span>
        <span className="text-sm font-medium text-indigo-400">{label}</span>
      </div>

      <div className="relative">
        <input
          type="range"
          min={0}
          max={100}
          value={pct}
          onChange={(e) => onChange(Number(e.target.value) / 100)}
          className="w-full accent-indigo-500 cursor-pointer"
        />
        <div
          className="absolute -top-1 w-0.5 h-4 bg-neutral-600 pointer-events-none"
          style={{ left: `${presetDefault * 100}%` }}
          title={`Preset default: ${Math.round(presetDefault * 100)}%`}
        />
      </div>

      <div className="flex justify-between text-xs text-neutral-600">
        <span>Fewer sounds</span>
        <span>More sounds</span>
      </div>
    </div>
  );
}
