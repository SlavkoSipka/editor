"use client";

import type { JobInfo } from "@/lib/api";
import { Download, Loader2, Music } from "lucide-react";
import { useState } from "react";

type Props = {
  job: JobInfo;
  videoRef?: React.RefObject<HTMLVideoElement | null>;
};

type DownloadKey = "mp4" | "zip";

export function JobResult({ job, videoRef }: Props) {
  const [downloading, setDownloading] = useState<DownloadKey | null>(null);

  if (job.status !== "done") return null;

  const videoUrl = job.video_url || "";
  const zipUrl = job.project_zip_url || "";

  // Cross-origin <a download> is ignored by every browser, so we fetch the
  // file as a blob and trigger a same-origin download. Works regardless of
  // whether the API sets Content-Disposition.
  async function handleDownload(
    key: DownloadKey,
    url: string,
    filename: string,
  ) {
    if (!url || downloading) return;
    setDownloading(key);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
    } catch (err) {
      window.open(url, "_blank", "noopener,noreferrer");
    } finally {
      setDownloading(null);
    }
  }

  const mp4Name = `sfx_job_${job.job_id}.mp4`;
  const zipName = `sfx_job_${job.job_id}.zip`;

  return (
    <div className="space-y-4">
      <video
        ref={videoRef}
        src={videoUrl}
        controls
        playsInline
        preload="metadata"
        className="w-full rounded-xl bg-black aspect-video"
      />

      <div className="flex flex-wrap gap-6 text-sm text-neutral-400">
        <div>
          Duration:{" "}
          <span className="text-neutral-200">
            {job.duration_sec != null ? `${job.duration_sec.toFixed(1)}s` : "—"}
          </span>
        </div>
        <div>
          SFX: <span className="text-neutral-200">{job.sfx_count ?? "—"}</span>
        </div>
        {job.music_track && (
          <div className="flex items-center gap-1 min-w-0">
            <Music className="w-4 h-4 shrink-0" />
            <span className="text-neutral-200 truncate max-w-xs">{job.music_track}</span>
          </div>
        )}
      </div>

      <div className="flex flex-col sm:flex-row gap-3 pt-2">
        <button
          type="button"
          onClick={() => handleDownload("mp4", videoUrl, mp4Name)}
          disabled={!videoUrl || downloading !== null}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-60 disabled:cursor-not-allowed rounded-lg font-medium transition"
        >
          {downloading === "mp4" ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Download className="w-4 h-4" />
          )}
          {downloading === "mp4" ? "Preparing…" : "Download MP4"}
        </button>
        <button
          type="button"
          onClick={() => handleDownload("zip", zipUrl, zipName)}
          disabled={!zipUrl || downloading !== null}
          className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-neutral-800 hover:bg-neutral-700 disabled:opacity-60 disabled:cursor-not-allowed rounded-lg font-medium transition"
        >
          {downloading === "zip" ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : (
            <Download className="w-4 h-4" />
          )}
          {downloading === "zip" ? "Preparing…" : "Editor Project (ZIP)"}
        </button>
      </div>

      <p className="text-xs text-neutral-500 text-center pt-2">
        Project ZIP includes timeline files for DaVinci Resolve, Premiere, Final Cut
      </p>
    </div>
  );
}
