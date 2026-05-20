export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export type Preset = {
  id: string;
  name: string;
  description: string;
};

export type JobStep = {
  name: string;
  started_at: string | null;
  completed_at: string | null;
  detail: string | null;
};

export type JobInfo = {
  job_id: string;
  status: "pending" | "processing" | "done" | "failed";
  preset: string;
  created_at: string;
  updated_at: string;
  progress_pct: number;
  current_step: string | null;
  steps: JobStep[];
  error: string | null;
  video_url: string | null;
  project_zip_url: string | null;
  duration_sec: number | null;
  sfx_count: number | null;
  music_track: string | null;
};

export async function getPresets(): Promise<Preset[]> {
  const res = await fetch(`${API_BASE}/presets`);
  if (!res.ok) throw new Error("Failed to load presets");
  return res.json();
}

export async function submitJob(file: File, preset: string): Promise<JobInfo> {
  const form = new FormData();
  form.append("video", file);
  form.append("preset", preset);

  const res = await fetch(`${API_BASE}/jobs`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Upload failed: ${detail}`);
  }
  return res.json();
}

export async function getJob(jobId: string): Promise<JobInfo> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error("Job not found");
  return res.json();
}
