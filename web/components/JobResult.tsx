"use client";

import type { JobInfo } from "@/lib/api";
import { Download, Music } from "lucide-react";

type Props = { job: JobInfo };

export function JobResult({ job }: Props) {
  if (job.status !== "done") return null;

  const videoUrl = job.video_url || "";
  const zipUrl = job.project_zip_url || "";

  return (
    <div className="space-y-4">
      <video
        src={videoUrl}
        controls
        playsInline
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
        <a
          href={videoUrl}
          download
          className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-indigo-600 hover:bg-indigo-500 rounded-lg font-medium transition"
        >
          <Download className="w-4 h-4" />
          Download MP4
        </a>
        <a
          href={zipUrl}
          download
          className="flex-1 flex items-center justify-center gap-2 px-4 py-3 bg-neutral-800 hover:bg-neutral-700 rounded-lg font-medium transition"
        >
          <Download className="w-4 h-4" />
          Editor Project (ZIP)
        </a>
      </div>

      <p className="text-xs text-neutral-500 text-center pt-2">
        Project ZIP includes timeline files for DaVinci Resolve, Premiere, Final Cut
      </p>
    </div>
  );
}
