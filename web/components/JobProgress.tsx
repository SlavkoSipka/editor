"use client";

import type { JobInfo } from "@/lib/api";
import { AlertCircle, Check, Loader2 } from "lucide-react";

type Props = { job: JobInfo };

export function JobProgress({ job }: Props) {
  const isProcessing = job.status === "pending" || job.status === "processing";
  const isDone = job.status === "done";
  const isFailed = job.status === "failed";

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-sm text-neutral-400">
          Job <code className="text-neutral-300">{job.job_id}</code> · preset{" "}
          <span className="text-neutral-300">{job.preset}</span>
        </div>
        <div className="text-sm font-medium">{job.progress_pct}%</div>
      </div>

      <div className="h-2 bg-neutral-900 rounded-full overflow-hidden">
        <div
          className={`h-full transition-all duration-500 ${
            isFailed ? "bg-red-500" : isDone ? "bg-green-500" : "bg-indigo-500"
          }`}
          style={{ width: `${job.progress_pct}%` }}
        />
      </div>

      <div className="space-y-2 mt-4">
        {job.steps.map((step, i) => {
          const inProgress = !step.completed_at && isProcessing;
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

      {isFailed && job.error && (
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
