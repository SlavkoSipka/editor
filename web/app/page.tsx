"use client";

import { useEffect, useRef, useState } from "react";
import { getJob, getPresets, submitJob, type JobInfo, type Preset } from "@/lib/api";
import { UploadZone } from "@/components/UploadZone";
import { PresetGrid } from "@/components/PresetGrid";
import { JobProgress } from "@/components/JobProgress";
import { JobResult } from "@/components/JobResult";
import { Sparkles } from "lucide-react";

export default function Home() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    getPresets()
      .then((list) => {
        setPresets(list);
        setSelectedPreset((prev) => {
          if (prev || list.length === 0) return prev;
          const def = list.find((p) => p.id === "tiktok_viral") ?? list[0];
          return def.id;
        });
      })
      .catch((e: Error) => setError(e.message));
  }, []);

  useEffect(() => {
    if (!job || job.status === "done" || job.status === "failed") {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
      return;
    }
    pollRef.current = setInterval(async () => {
      try {
        const updated = await getJob(job.job_id);
        setJob(updated);
      } catch {
        /* transient network errors — next poll */
      }
    }, 2000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [job?.job_id, job?.status]);

  async function handleGenerate() {
    if (!file || !selectedPreset) return;
    setSubmitting(true);
    setError(null);
    setJob(null);
    try {
      const newJob = await submitJob(file, selectedPreset);
      setJob(newJob);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setSubmitting(false);
    }
  }

  function reset() {
    setJob(null);
    setFile(null);
    setError(null);
  }

  return (
    <main className="min-h-screen px-4 py-12">
      <div className="max-w-2xl mx-auto space-y-10">
        <div className="text-center space-y-3">
          <div className="inline-flex items-center gap-2 text-xs uppercase tracking-wider text-indigo-400">
            <Sparkles className="w-3.5 h-3.5" />
            AI Sound Effects
          </div>
          <h1 className="text-4xl font-bold tracking-tight">Auto SFX for your video</h1>
          <p className="text-neutral-400 max-w-md mx-auto">
            Drop a clip, pick a style, get a finished video plus editable project files.
          </p>
        </div>

        {!job && (
          <>
            <section className="space-y-3">
              <div className="text-sm text-neutral-500 font-medium">1 — Upload video</div>
              <UploadZone onFileSelected={setFile} selectedFile={file} />
            </section>

            <section className="space-y-3">
              <div className="text-sm text-neutral-500 font-medium">2 — Pick a style</div>
              <PresetGrid
                presets={presets}
                selected={selectedPreset}
                onSelect={setSelectedPreset}
              />
            </section>

            <section>
              <button
                type="button"
                onClick={handleGenerate}
                disabled={!file || !selectedPreset || submitting}
                className="w-full py-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-neutral-800 disabled:text-neutral-600 disabled:cursor-not-allowed font-medium transition"
              >
                {submitting ? "Uploading…" : "Generate"}
              </button>
              {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
            </section>
          </>
        )}

        {job && (
          <section className="space-y-6">
            <JobProgress job={job} />
            <JobResult job={job} />

            {(job.status === "done" || job.status === "failed") && (
              <button
                type="button"
                onClick={reset}
                className="w-full py-3 rounded-xl bg-neutral-800 hover:bg-neutral-700 font-medium transition"
              >
                ↺ Process another video
              </button>
            )}
          </section>
        )}

        <footer className="text-center text-xs text-neutral-600 pt-8">
          AI SFX MVP · running locally
        </footer>
      </div>
    </main>
  );
}
