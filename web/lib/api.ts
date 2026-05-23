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

export type SoundDiagnostic = {
  timestamp: number;
  action_type: string;
  sound_name: string;
  sound_id?: string | null;
  matched_tier?: string | null;
  match_score?: number | null;
  value_tier?: string | null;
  layer: string;
  fetch_status: string;
  included_in_mix: boolean;
  note?: string | null;
};

export type JobDiagnostics = {
  director_style?: string | null;
  director_vibe?: string | null;
  director_recommended_preset?: string | null;
  director_max_sfx?: number | null;
  anchor_moments: number;
  scenes_detected: number;
  actions_suggested: number;
  sfx_matched: number;
  sfx_after_selectivity: number;
  sfx_in_final_mix: number;
  ambient_count: number;
  music_found: boolean;
  music_name?: string | null;
  music_fetch_status?: string | null;
  storage_mode: string;
  r2_fetch_attempts: number;
  r2_fetch_failures: number;
  speech_regions: number;
  speech_coverage_pct: number;
  sounds: SoundDiagnostic[];
  warnings: string[];
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
  diagnostics?: JobDiagnostics | null;
};

export type UploadProgress = {
  loadedBytes: number;
  totalBytes: number;
  pct: number;
};

export type UploadProgressCallback = (progress: UploadProgress) => void;

export async function getPresets(): Promise<Preset[]> {
  const res = await fetch(`${API_BASE}/presets`);
  if (!res.ok) throw new Error("Failed to load presets");
  return res.json();
}

export function submitJob(
  file: File,
  preset: string,
  onProgress?: UploadProgressCallback,
): Promise<JobInfo> {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    form.append("video", file);
    form.append("preset", preset);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE}/jobs`);

    xhr.upload.onprogress = (event) => {
      if (!onProgress) return;
      const totalBytes = event.lengthComputable ? event.total : file.size;
      const loadedBytes = event.loaded;
      const pct =
        totalBytes > 0
          ? Math.min(100, Math.round((loadedBytes / totalBytes) * 100))
          : 0;
      onProgress({ loadedBytes, totalBytes, pct });
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as JobInfo);
        } catch {
          reject(new Error("Upload failed: invalid server response"));
        }
        return;
      }
      reject(new Error(`Upload failed: ${xhr.responseText || xhr.statusText}`));
    };

    xhr.onerror = () => {
      reject(new Error("Upload failed: network error"));
    };

    xhr.onabort = () => {
      reject(new Error("Upload cancelled"));
    };

    xhr.send(form);
  });
}

export async function getJob(jobId: string): Promise<JobInfo> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error("Job not found");
  return res.json();
}
