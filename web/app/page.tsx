"use client";

import { useEffect, useRef, useState } from "react";
import {
  getJob,
  getPresets,
  submitJob,
  type JobInfo,
  type Preset,
  type UploadProgress,
} from "@/lib/api";
import { UploadZone } from "@/components/UploadZone";
import { PresetGrid } from "@/components/PresetGrid";
import { DensitySlider } from "@/components/DensitySlider";
import { JobProgress, type GenerationPhase } from "@/components/JobProgress";
import { JobResult } from "@/components/JobResult";
import { DiagnosticsPanel } from "@/components/DiagnosticsPanel";
import { FeedbackPanel } from "@/components/FeedbackPanel";
import { Sparkles } from "lucide-react";

export default function Home() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string | null>(null);
  const [density, setDensity] = useState<number>(0.5);
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [phase, setPhase] = useState<GenerationPhase | "idle">("idle");
  const [uploadProgress, setUploadProgress] = useState<UploadProgress | null>(null);
  const [activeFile, setActiveFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

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
    if (!selectedPreset) return;
    const p = presets.find((x) => x.id === selectedPreset);
    if (p) setDensity(p.density);
  }, [selectedPreset, presets]);

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

  useEffect(() => {
    if (!job) return;
    if (job.status === "done") {
      setPhase("done");
    } else if (job.status === "failed") {
      setPhase("failed");
    } else if (phase !== "uploading") {
      setPhase("processing");
    }
  }, [job?.status, job, phase]);

  async function handleGenerate() {
    if (!file || !selectedPreset) return;
    setError(null);
    setJob(null);
    setUploadProgress({ loadedBytes: 0, totalBytes: file.size, pct: 0 });
    setActiveFile(file);
    setPhase("uploading");

    try {
      const newJob = await submitJob(file, selectedPreset, density, (progress) => {
        setUploadProgress(progress);
      });
      setUploadProgress((prev) =>
        prev ? { ...prev, loadedBytes: prev.totalBytes, pct: 100 } : null,
      );
      setJob(newJob);
      setPhase("processing");
    } catch (e: unknown) {
      setPhase("failed");
      setError(e instanceof Error ? e.message : "Upload failed");
    }
  }

  function reset() {
    setJob(null);
    setFile(null);
    setActiveFile(null);
    setUploadProgress(null);
    setPhase("idle");
    setError(null);
  }

  const showForm = phase === "idle";

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

        {showForm && (
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

            {selectedPreset && (
              <section className="space-y-3">
                <div className="text-sm text-neutral-500 font-medium">
                  3 — Adjust density (optional)
                </div>
                <DensitySlider
                  value={density}
                  presetDefault={
                    presets.find((p) => p.id === selectedPreset)?.density ?? 0.5
                  }
                  onChange={setDensity}
                />
              </section>
            )}

            <section>
              <button
                type="button"
                onClick={handleGenerate}
                disabled={!file || !selectedPreset}
                className="w-full py-4 rounded-xl bg-indigo-600 hover:bg-indigo-500 disabled:bg-neutral-800 disabled:text-neutral-600 disabled:cursor-not-allowed font-medium transition"
              >
                Generate
              </button>
              {error && <p className="mt-3 text-sm text-red-400">{error}</p>}
            </section>
          </>
        )}

        {phase !== "idle" && (
          <section className="space-y-6">
            <JobProgress
              phase={phase}
              job={job}
              upload={uploadProgress}
              fileName={activeFile?.name}
              uploadError={phase === "failed" && !job ? error : null}
            />
            {job && <JobResult job={job} videoRef={videoRef} />}

            {job?.diagnostics &&
              (job.status === "done" || job.status === "failed") && (
                <DiagnosticsPanel diagnostics={job.diagnostics} />
              )}

            {job?.status === "done" && job.diagnostics && (
              <FeedbackPanel job={job} density={density} videoRef={videoRef} />
            )}

            {(phase === "done" || phase === "failed") && (
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
