"use client";

import { useState } from "react";
import {
  type JobInfo,
  type SoundRating,
  type JobFeedback,
  submitFeedback,
} from "@/lib/api";
import { ThumbsUp, ThumbsDown, Ban, Play, Plus, Check } from "lucide-react";

type Props = {
  job: JobInfo;
  density: number;
  videoRef: React.RefObject<HTMLVideoElement | null>;
};

export function FeedbackPanel({ job, density, videoRef }: Props) {
  const allSounds = job.diagnostics?.sounds ?? [];
  // Keep each sound's original index so ratings + submission stay aligned
  // even though we only render the SFX rows.
  const sfxSounds = allSounds
    .map((s, idx) => ({ sound: s, idx }))
    .filter((x) => x.sound.layer === "sfx");

  const [ratings, setRatings] = useState<Record<number, SoundRating>>({});
  const [overall, setOverall] = useState<number>(0);
  const [densityFb, setDensityFb] = useState<
    "too_low" | "right" | "too_high" | ""
  >("");
  const [note, setNote] = useState("");
  const [missing, setMissing] = useState<{ timestamp: number; note?: string }[]>(
    [],
  );
  const [submitted, setSubmitted] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function seekTo(t: number) {
    const v = videoRef.current;
    if (v) {
      v.currentTime = Math.max(0, t - 0.3);
      v.play().catch(() => {});
    }
  }

  function rate(i: number, r: SoundRating) {
    setRatings((prev) => {
      const next = { ...prev };
      if (next[i] === r) delete next[i];
      else next[i] = r;
      return next;
    });
  }

  function markMissingHere() {
    const v = videoRef.current;
    const t = v ? v.currentTime : 0;
    setMissing((m) => [...m, { timestamp: Math.round(t * 10) / 10 }]);
  }

  async function handleSubmit() {
    setError(null);
    const fb: JobFeedback = {
      job_id: job.job_id,
      preset: job.preset,
      density,
      overall_rating: overall || undefined,
      density_feedback: densityFb || undefined,
      overall_note: note || undefined,
      sound_feedback: allSounds
        .map((s, i) =>
          ratings[i]
            ? {
                sound_index: i,
                sound_id: s.sound_id ?? undefined,
                sound_name: s.sound_name ?? undefined,
                action_type: s.action_type ?? undefined,
                timestamp: s.timestamp,
                matched_tier: s.matched_tier ?? undefined,
                match_score: s.match_score ?? undefined,
                rating: ratings[i],
              }
            : null,
        )
        .filter((x): x is NonNullable<typeof x> => x !== null),
      missing_sounds: missing,
    };
    try {
      await submitFeedback(job.job_id, fb);
      setSubmitted(true);
    } catch {
      setError("Failed to submit feedback — please try again.");
    }
  }

  if (submitted) {
    return (
      <div className="flex items-center gap-2 p-4 rounded-xl bg-green-500/10 border border-green-500/30 text-green-300 text-sm">
        <Check className="w-4 h-4" /> Feedback saved — thank you. This directly
        improves the system.
      </div>
    );
  }

  return (
    <div className="border border-neutral-800 rounded-xl p-4 space-y-5">
      <div className="text-sm font-medium">Rate this result</div>
      <p className="text-xs text-neutral-500">
        Click a sound to jump to it in the video, then rate it. This feedback is
        used to improve matching.
      </p>

      <div className="space-y-1.5">
        {sfxSounds.map(({ sound: s, idx }) => {
          const r = ratings[idx];
          return (
            <div
              key={idx}
              className="flex items-center gap-2 text-xs py-1.5 border-b border-neutral-900 last:border-0"
            >
              <button
                type="button"
                onClick={() => seekTo(s.timestamp)}
                className="flex items-center gap-1 text-neutral-500 hover:text-indigo-400 w-14 shrink-0"
              >
                <Play className="w-3 h-3" /> {s.timestamp.toFixed(1)}s
              </button>
              <span
                className="flex-1 truncate text-neutral-300"
                title={s.sound_name}
              >
                {s.sound_name}
              </span>
              <div className="flex gap-1 shrink-0">
                <RateBtn
                  active={r === "good"}
                  onClick={() => rate(idx, "good")}
                  color="green"
                >
                  <ThumbsUp className="w-3.5 h-3.5" />
                </RateBtn>
                <RateBtn
                  active={r === "wrong"}
                  onClick={() => rate(idx, "wrong")}
                  color="red"
                >
                  <ThumbsDown className="w-3.5 h-3.5" />
                </RateBtn>
                <RateBtn
                  active={r === "unnecessary"}
                  onClick={() => rate(idx, "unnecessary")}
                  color="amber"
                >
                  <Ban className="w-3.5 h-3.5" />
                </RateBtn>
              </div>
            </div>
          );
        })}
        {sfxSounds.length === 0 && (
          <div className="text-neutral-500 text-xs">No SFX to rate.</div>
        )}
      </div>

      <div className="flex gap-4 text-[11px] text-neutral-600">
        <span className="flex items-center gap-1">
          <ThumbsUp className="w-3 h-3 text-green-500" /> good
        </span>
        <span className="flex items-center gap-1">
          <ThumbsDown className="w-3 h-3 text-red-500" /> wrong sound
        </span>
        <span className="flex items-center gap-1">
          <Ban className="w-3 h-3 text-amber-500" /> not needed
        </span>
      </div>

      <div className="space-y-2">
        <button
          type="button"
          onClick={markMissingHere}
          className="flex items-center gap-2 text-xs text-neutral-400 hover:text-indigo-400"
        >
          <Plus className="w-3.5 h-3.5" /> Mark &quot;a sound was missing
          here&quot; (at current video time)
        </button>
        {missing.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {missing.map((m, i) => (
              <span
                key={i}
                className="text-[11px] px-2 py-0.5 rounded bg-neutral-800 text-neutral-400"
              >
                missing @ {m.timestamp}s
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="space-y-2">
        <div className="text-xs text-neutral-500">How was the amount of sound?</div>
        <div className="flex gap-2">
          {(["too_low", "right", "too_high"] as const).map((opt) => (
            <button
              key={opt}
              type="button"
              onClick={() => setDensityFb(opt)}
              className={`px-3 py-1.5 rounded-lg text-xs transition ${
                densityFb === opt
                  ? "bg-indigo-600 text-white"
                  : "bg-neutral-800 text-neutral-400 hover:bg-neutral-700"
              }`}
            >
              {opt === "too_low"
                ? "Too few"
                : opt === "right"
                  ? "Just right"
                  : "Too many"}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-2">
        <div className="text-xs text-neutral-500">Overall sound design</div>
        <div className="flex gap-1">
          {[1, 2, 3, 4, 5].map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setOverall(n)}
              className={`w-9 h-9 rounded-lg text-sm transition ${
                overall >= n
                  ? "bg-indigo-600 text-white"
                  : "bg-neutral-800 text-neutral-500 hover:bg-neutral-700"
              }`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      <textarea
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="What's the biggest problem? What worked? (free text — very useful)"
        className="w-full bg-neutral-900 border border-neutral-800 rounded-lg p-3 text-sm text-neutral-200 placeholder-neutral-600 resize-none h-20"
      />

      {error && <p className="text-sm text-red-400">{error}</p>}

      <button
        type="button"
        onClick={handleSubmit}
        className="w-full py-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-medium transition text-sm"
      >
        Submit feedback
      </button>
    </div>
  );
}

function RateBtn({
  active,
  onClick,
  color,
  children,
}: {
  active: boolean;
  onClick: () => void;
  color: "green" | "red" | "amber";
  children: React.ReactNode;
}) {
  const colors = {
    green: active ? "bg-green-600 text-white" : "text-neutral-500 hover:text-green-400",
    red: active ? "bg-red-600 text-white" : "text-neutral-500 hover:text-red-400",
    amber: active ? "bg-amber-600 text-white" : "text-neutral-500 hover:text-amber-400",
  };
  return (
    <button
      type="button"
      onClick={onClick}
      className={`p-1.5 rounded-md transition ${colors[color]}`}
    >
      {children}
    </button>
  );
}
