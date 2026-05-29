"use client";

import { useCallback, useEffect, useState } from "react";
import { API_BASE } from "@/lib/api";

type FeedbackRow = {
  id: string;
  job_id: string;
  reviewer?: string;
  preset?: string;
  density?: number;
  overall_rating?: number;
  density_feedback?: string;
  overall_note?: string;
  status: string;
  dev_note?: string;
  created_at: string;
};

type WorstAction = { action: string; ratings: Record<string, number> };

type Summary = {
  total_jobs?: number;
  avg_overall_rating?: number | null;
  missing_sounds_total?: number;
  by_status?: Record<string, number>;
  worst_action_types?: WorstAction[];
};

export default function AdminPage() {
  const [items, setItems] = useState<FeedbackRow[]>([]);
  const [statusFilter, setStatusFilter] = useState("");
  const [reviewerFilter, setReviewerFilter] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);

  const load = useCallback(async () => {
    const q = new URLSearchParams();
    if (statusFilter) q.set("status", statusFilter);
    if (reviewerFilter) q.set("reviewer", reviewerFilter);
    const res = await fetch(`${API_BASE}/admin/feedback?${q}`);
    const data = await res.json();
    setItems(data.items || []);
  }, [statusFilter, reviewerFilter]);

  const loadSummary = useCallback(async () => {
    const res = await fetch(`${API_BASE}/feedback/summary`);
    setSummary(await res.json());
  }, []);

  useEffect(() => {
    void (async () => {
      await load();
      await loadSummary();
    })();
  }, [load, loadSummary]);

  async function setStatus(id: string, status: string) {
    const note = window.prompt("Dev note (what did you change)? optional:") || "";
    const form = new FormData();
    form.append("status", status);
    if (note) form.append("dev_note", note);
    await fetch(`${API_BASE}/admin/feedback/${id}/status`, {
      method: "POST",
      body: form,
    });
    load();
  }

  return (
    <main className="min-h-screen px-4 py-10 bg-neutral-950 text-neutral-100">
      <div className="max-w-5xl mx-auto space-y-8">
        <h1 className="text-2xl font-bold">Feedback Tracking</h1>

        {summary && (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <Card label="Total feedback" value={summary.total_jobs ?? 0} />
            <Card label="Avg rating" value={summary.avg_overall_rating ?? "—"} />
            <Card label="Missing marks" value={summary.missing_sounds_total ?? 0} />
            <Card label="Pending" value={summary.by_status?.pending ?? 0} />
          </div>
        )}

        {summary?.worst_action_types && summary.worst_action_types.length > 0 && (
          <div className="border border-neutral-800 rounded-xl p-4">
            <div className="text-sm font-medium mb-3">
              Worst action types (most wrong/unnecessary)
            </div>
            <div className="space-y-1 text-xs">
              {summary.worst_action_types.slice(0, 10).map((a) => (
                <div key={a.action} className="flex justify-between">
                  <span className="text-neutral-400">{a.action}</span>
                  <span className="text-neutral-500">
                    {JSON.stringify(a.ratings)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="flex gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="bg-neutral-900 border border-neutral-800 rounded-lg px-3 py-2 text-sm"
          >
            <option value="">All statuses</option>
            <option value="pending">Pending</option>
            <option value="applied">Applied</option>
            <option value="ignored">Ignored</option>
          </select>
          <input
            value={reviewerFilter}
            onChange={(e) => setReviewerFilter(e.target.value)}
            placeholder="filter by reviewer"
            className="bg-neutral-900 border border-neutral-800 rounded-lg px-3 py-2 text-sm"
          />
        </div>

        <div className="space-y-3">
          {items.map((it) => (
            <div
              key={it.id}
              className="border border-neutral-800 rounded-xl p-4 space-y-2"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3 text-sm">
                  <span className="font-medium">{it.preset}</span>
                  <span className="text-neutral-500">by {it.reviewer || "?"}</span>
                  <span className="text-neutral-600">
                    rating {it.overall_rating ?? "—"}/5
                  </span>
                  <StatusBadge status={it.status} />
                </div>
                <div className="text-xs text-neutral-600">
                  {it.created_at?.slice(0, 16).replace("T", " ")}
                </div>
              </div>
              {it.overall_note && (
                <div className="text-sm text-neutral-300">
                  &quot;{it.overall_note}&quot;
                </div>
              )}
              {it.dev_note && (
                <div className="text-xs text-green-400">dev: {it.dev_note}</div>
              )}
              <div className="flex gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setStatus(it.id, "applied")}
                  className="text-xs px-3 py-1 rounded bg-green-700 hover:bg-green-600"
                >
                  Mark applied
                </button>
                <button
                  type="button"
                  onClick={() => setStatus(it.id, "ignored")}
                  className="text-xs px-3 py-1 rounded bg-neutral-800 hover:bg-neutral-700"
                >
                  Ignore
                </button>
                <button
                  type="button"
                  onClick={() => setStatus(it.id, "pending")}
                  className="text-xs px-3 py-1 rounded bg-neutral-800 hover:bg-neutral-700"
                >
                  Reset pending
                </button>
              </div>
            </div>
          ))}
          {items.length === 0 && (
            <div className="text-neutral-600 text-sm">No feedback yet.</div>
          )}
        </div>
      </div>
    </main>
  );
}

function Card({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="border border-neutral-800 rounded-xl p-4">
      <div className="text-xs text-neutral-500">{label}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: "bg-amber-500/20 text-amber-400",
    applied: "bg-green-500/20 text-green-400",
    ignored: "bg-neutral-700 text-neutral-400",
  };
  return (
    <span className={`text-xs px-2 py-0.5 rounded ${colors[status] || ""}`}>
      {status}
    </span>
  );
}
