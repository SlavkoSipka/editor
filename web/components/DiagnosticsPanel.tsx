"use client";

import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle,
  ChevronDown,
  ChevronRight,
  XCircle,
} from "lucide-react";
import type { JobDiagnostics } from "@/lib/api";

type Props = { diagnostics: JobDiagnostics };

export function DiagnosticsPanel({ diagnostics: d }: Props) {
  const [open, setOpen] = useState(false);

  const hasProblems =
    d.r2_fetch_failures > 0 ||
    !d.music_found ||
    d.music_fetch_status === "failed" ||
    d.sfx_in_final_mix === 0 ||
    d.warnings.length > 0;

  return (
    <div className="border border-neutral-800 rounded-xl overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center justify-between px-4 py-3 bg-neutral-900/60 hover:bg-neutral-900 transition"
      >
        <div className="flex items-center gap-2 text-sm font-medium">
          {open ? (
            <ChevronDown className="w-4 h-4" />
          ) : (
            <ChevronRight className="w-4 h-4" />
          )}
          Diagnostics
          {hasProblems ? (
            <span className="flex items-center gap-1 text-amber-400 text-xs">
              <AlertTriangle className="w-3.5 h-3.5" /> issues detected
            </span>
          ) : (
            <span className="flex items-center gap-1 text-green-400 text-xs">
              <CheckCircle className="w-3.5 h-3.5" /> looks healthy
            </span>
          )}
        </div>
      </button>

      {open && (
        <div className="p-4 space-y-5 text-sm bg-neutral-950">
          {d.warnings.length > 0 && (
            <div className="space-y-1">
              {d.warnings.map((w, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 text-amber-300 text-xs"
                >
                  <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
                  <span className="break-words">{w}</span>
                </div>
              ))}
            </div>
          )}

          <Section title="Director">
            <Row label="Style" value={d.director_style || "—"} />
            <Row label="Vibe" value={d.director_vibe || "—"} />
            <Row
              label="Recommended preset"
              value={d.director_recommended_preset || "—"}
            />
            <Row
              label="Max SFX budget"
              value={d.director_max_sfx != null ? String(d.director_max_sfx) : "—"}
            />
            <Row label="Anchor moments" value={String(d.anchor_moments)} />
          </Section>

          <Section title="Pipeline">
            <Row label="Scenes detected" value={String(d.scenes_detected)} />
            <Row
              label="Actions suggested (Gemini)"
              value={String(d.actions_suggested)}
              warn={d.actions_suggested === 0}
            />
            <Row label="SFX matched in library" value={String(d.sfx_matched)} />
            <Row
              label="SFX after selectivity"
              value={String(d.sfx_after_selectivity)}
            />
            <Row
              label="SFX in final mix"
              value={String(d.sfx_in_final_mix)}
              warn={d.sfx_in_final_mix === 0}
            />
            <Row label="Ambient layers" value={String(d.ambient_count)} />
          </Section>

          <Section title="Music">
            <Row
              label="Music found"
              value={d.music_found ? "yes" : "NO"}
              warn={!d.music_found}
            />
            <Row label="Track" value={d.music_name || "—"} />
            <Row
              label="Fetch status"
              value={d.music_fetch_status || "—"}
              warn={
                d.music_fetch_status === "failed" ||
                d.music_fetch_status === "load_failed"
              }
            />
          </Section>

          <Section title="Storage / R2">
            <Row label="Storage mode" value={d.storage_mode} />
            <Row label="Fetch attempts" value={String(d.r2_fetch_attempts)} />
            <Row
              label="Fetch failures"
              value={String(d.r2_fetch_failures)}
              warn={d.r2_fetch_failures > 0}
            />
          </Section>

          <Section title="Speech">
            <Row label="Speech regions" value={String(d.speech_regions)} />
            <Row
              label="Coverage"
              value={`${d.speech_coverage_pct.toFixed(0)}%`}
            />
          </Section>

          <Section title={`Sounds (${d.sounds.length})`}>
            <div className="space-y-1 mt-1">
              {d.sounds.length === 0 && (
                <div className="text-neutral-500 text-xs">
                  No sounds in this job.
                </div>
              )}
              {d.sounds.map((s, i) => (
                <div
                  key={i}
                  className="flex items-center gap-2 text-xs py-1 border-b border-neutral-900 last:border-0"
                  title={s.note || undefined}
                >
                  {s.included_in_mix ? (
                    <CheckCircle className="w-3.5 h-3.5 text-green-500 shrink-0" />
                  ) : (
                    <XCircle className="w-3.5 h-3.5 text-red-500 shrink-0" />
                  )}
                  <span className="text-neutral-500 w-12 shrink-0">
                    {s.timestamp.toFixed(1)}s
                  </span>
                  <span className="text-neutral-300 w-16 shrink-0">
                    {s.layer}
                  </span>
                  <span
                    className="text-neutral-400 flex-1 truncate"
                    title={s.sound_name}
                  >
                    {s.sound_name}
                  </span>
                  {s.matched_tier && (
                    <span className="text-neutral-600 shrink-0">
                      {s.matched_tier}
                    </span>
                  )}
                  {s.match_score != null && (
                    <span className="text-neutral-600 w-10 shrink-0 text-right">
                      {s.match_score.toFixed(2)}
                    </span>
                  )}
                  <span
                    className={`shrink-0 w-16 text-right ${
                      s.fetch_status === "failed" ||
                      s.fetch_status === "load_failed"
                        ? "text-red-400"
                        : "text-neutral-600"
                    }`}
                  >
                    {s.fetch_status}
                  </span>
                </div>
              ))}
            </div>
          </Section>
        </div>
      )}
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-neutral-500 mb-2">
        {title}
      </div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}

function Row({
  label,
  value,
  warn,
}: {
  label: string;
  value: string;
  warn?: boolean;
}) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-neutral-500">{label}</span>
      <span className={warn ? "text-amber-400 font-medium" : "text-neutral-200"}>
        {value}
      </span>
    </div>
  );
}
