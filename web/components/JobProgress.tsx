"use client";

import type { JobInfo, UploadProgress } from "@/lib/api";
import { AlertCircle, Check, Loader2 } from "lucide-react";

export type GenerationPhase = "uploading" | "processing" | "done" | "failed";

type Props = {
  phase: GenerationPhase;
  job?: JobInfo | null;
  upload?: UploadProgress | null;
  fileName?: string;
  uploadError?: string | null;
};

function formatMb(bytes: number): string {
  return (bytes / 1024 / 1024).toFixed(1);
}

export function JobProgress({
  phase,
  job,
  upload,
  fileName,
  uploadError,
}: Props) {
  const isUploading = phase === "uploading";
  const isProcessing = phase === "processing";
  const isDone = phase === "done";
  const isFailed = phase === "failed";

  const uploadComplete = !isUploading && (isProcessing || isDone || isFailed);
  const uploadPct = upload?.pct ?? 0;
  const pipelinePct = job?.progress_pct ?? 0;

  const showPipelineSteps = job && (isProcessing || isDone || isFailed);
  const pipelineProcessing =
    job?.status === "pending" || job?.status === "processing";

  return (
    <div className="space-y-4">
      {job && (
        <div className="flex items-center justify-between">
          <div className="text-sm text-neutral-400">
            Job <code className="text-neutral-300">{job.job_id}</code> · preset{" "}
            <span className="text-neutral-300">{job.preset}</span>
          </div>
          {!isUploading && (
            <div className="text-sm font-medium">{pipelinePct}%</div>
          )}
        </div>
      )}

      {!job && isUploading && fileName && (
        <div className="text-sm text-neutral-400 truncate">{fileName}</div>
      )}

      {isUploading && (
        <>
          <div className="flex items-center justify-between text-sm">
            <span className="text-neutral-100 font-medium">Uploading video…</span>
            <span className="text-neutral-400">{uploadPct}%</span>
          </div>
          <div className="h-2 bg-neutral-900 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 transition-all duration-300"
              style={{ width: `${uploadPct}%` }}
            />
          </div>
          {upload && upload.totalBytes > 0 && (
            <div className="text-xs text-neutral-500">
              {formatMb(upload.loadedBytes)} / {formatMb(upload.totalBytes)} MB · Sending
              your video to the server…
            </div>
          )}
        </>
      )}

      {uploadComplete && (
        <div className="flex items-center gap-3 text-sm">
          <Check className="w-4 h-4 text-green-500 shrink-0" />
          <span className="text-neutral-400">
            Upload complete
            {fileName ? ` · ${fileName}` : ""}
          </span>
        </div>
      )}

      {showPipelineSteps && job && (
        <>
          <div className="h-2 bg-neutral-900 rounded-full overflow-hidden">
            <div
              className={`h-full transition-all duration-500 ${
                isFailed ? "bg-red-500" : isDone ? "bg-green-500" : "bg-indigo-500"
              }`}
              style={{ width: `${pipelinePct}%` }}
            />
          </div>

          <div className="space-y-2 mt-4">
            {job.steps.map((step, i) => {
              const inProgress = !step.completed_at && pipelineProcessing;
              const completed = !!step.completed_at;
              return (
                <div key={`${step.name}-${i}`} className="flex items-center gap-3 text-sm">
                  {completed ? (
                    <Check className="w-4 h-4 text-green-500 shrink-0" />
                  ) : inProgress ? (
                    <Loader2 className="w-4 h-4 text-indigo-400 animate-spin shrink-0" />
                  ) : (
                    <div className="w-4 h-4 rounded-full border border-neutral-700 shrink-0" />
                  )}
                  <span className={completed ? "text-neutral-400" : "text-neutral-100"}>
                    {step.name}
                  </span>
                </div>
              );
            })}
          </div>
        </>
      )}

      {isProcessing && job && job.steps.length === 0 && (
        <div className="flex items-center gap-3 text-sm text-neutral-300">
          <Loader2 className="w-4 h-4 text-indigo-400 animate-spin shrink-0" />
          Starting pipeline…
        </div>
      )}

      {isFailed && uploadError && !job?.error && (
        <div className="flex items-start gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/30 mt-4">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-medium text-red-300">Upload failed</div>
            <div className="text-sm text-red-200/80 break-words">{uploadError}</div>
          </div>
        </div>
      )}

      {isFailed && job?.error && (
        <div className="flex items-start gap-3 p-4 rounded-lg bg-red-500/10 border border-red-500/30 mt-4">
          <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
          <div>
            <div className="font-medium text-red-300">Failed</div>
            <div className="text-sm text-red-200/80 break-words">{job.error}</div>
          </div>
        </div>
      )}
    </div>
  );
}
