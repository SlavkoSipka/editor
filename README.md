# AI SFX

AI-powered sound effects generator for short-form video ads. Upload a video (≤3 min), pick a mood preset, and the pipeline analyzes the footage with Gemini, matches scene events against a local CC0 sound library, and renders a finished MP4 with multi-track SFX. This repo is the **Phase 0 CLI prototype** — no UI, no auth, no billing.

## Quick start

```bash
# One-time setup (after install) — see "Library setup" below for the full pipeline
docker run -d -p 6333:6333 -v $(pwd)/data/embeddings:/qdrant/storage qdrant/qdrant
python -m src.library.downloader        # 4-6 hours, resume-safe
python -m src.library.embedder           # ~1 hour

# Process a video
python process.py path/to/video.mp4 --preset dramatic

# Available presets: dramatic, energetic, mysterious, playful, luxury
```

## Library setup (one-time)

### 1. Freesound library (~25-40k sounds)

```bash
python -m src.library.downloader
```

Takes 4-6 hours. Resume-safe — if interrupted, re-run and it picks up where it left off (already-downloaded MP3s are skipped). Use `--pages-per-query 1` for a quick smoke test (~30 min).

### 2. Sonniss professional library (optional, +~15k sounds)

Sonniss releases free professional sound packs every year for GDC, all royalty-free for commercial use. This dramatically improves quality and coverage.

1. Visit https://sonniss.com/gameaudiogdc
2. Download free yearly archives (2020-2024 recommended)
3. Place all `.zip` files in `data/sonniss_raw/`
4. Run:
   ```bash
   python -m src.library.sonniss_importer
   ```

If `data/sonniss_raw/` is missing or empty the importer prints a hint and exits cleanly — the rest of the pipeline still works without Sonniss.

### 3. UGC / Viral library (Pixabay + Mixkit) — recommended for ads

```bash
python -m src.library.pixabay_downloader
python -m src.library.mixkit_downloader
```

~30-60 min total. Resume-safe. Both scrape PUBLIC, royalty-free sound-effect pages — see the license notes in each module docstring. If a site changes its HTML, the scraper logs a warning and the rest of the pipeline keeps working.

For a quick smoke test before the full run:
```bash
python -m src.library.pixabay_downloader --limit-queries 3
python -m src.library.mixkit_downloader --limit-categories 3
```

### 4. Music library

```bash
python -m src.library.music_downloader
```

### 5. Build vector index

```bash
docker run -d -p 6333:6333 -v $(pwd)/data/embeddings:/qdrant/storage qdrant/qdrant
python -m src.library.embedder           # SFX collection
python -m src.library.music_embedder     # Music collection
```

Useful flags:
- `--use-cached-analysis` — skip Gemini and reuse `data/temp/{video_name}/analysis.json` if it exists.
- `--output path/to/final.mp4` — write the final MP4 elsewhere instead of `data/temp/{video_name}/final.mp4`.

## Prerequisites

- Python 3.11+
- FFmpeg available on `PATH`
- Docker (for Qdrant)

## Setup

1. Clone the repo.
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate          # macOS/Linux
   # venv\Scripts\activate           # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the example env file and fill in your keys:
   ```bash
   cp .env.example .env
   # then edit .env and set GEMINI_API_KEY and FREESOUND_API_KEY
   ```
5. Verify FFmpeg is detected:
   ```bash
   python -m src.utils.ffmpeg_check
   ```

## Run the API locally

Make sure Qdrant is running:

```bash
docker run -d -p 6333:6333 -v "$(pwd)/data/embeddings:/qdrant/storage" qdrant/qdrant
```

Then start the API (from the repo root, with your venv activated):

```bash
source venv/bin/activate
python -m api.run
```

- API base URL: http://localhost:8000  
- OpenAPI docs (Swagger): http://localhost:8000/docs  
- Health check: http://localhost:8000/health  

Submit a job:

```bash
curl -X POST http://localhost:8000/jobs \
  -F "video=@tests/sample_videos/test.mp4" \
  -F "preset=tiktok_viral"
```

Poll until `status` is `done`, then download `video_url` / `project_zip_url` from the JSON or use:

`GET /jobs/{job_id}/download/final.mp4` and `GET /jobs/{job_id}/download/project.zip`.

### Environment variables (API)

| Variable | Purpose |
|----------|---------|
| `DATA_DIR` | Root data path; use `/data` on Railway when a volume is mounted. |
| `GEMINI_API_KEY` | Required — Gemini video analysis. |
| `FREESOUND_API_KEY` | Required by `config.py` at import (set a placeholder if unused). |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins; use `*` locally, your Vercel URL in production. |
| `PORT` | Uvicorn listen port (Railway often injects this). |

## Cloud deployment

Single Docker image: **FastAPI + Qdrant binary** in one container; Qdrant storage lives under `$DATA_DIR/embeddings` (same layout as local `docker run … -v ./data/embeddings:/qdrant/storage`). A Railway volume mounted at `/data` keeps uploads, library copies, embeddings, and Qdrant files across deploys.

### Backend (Railway)

1. Push the repo to GitHub.
2. [Railway](https://railway.app) → **New Project** → **Deploy from GitHub repo** → select this repo.
3. Railway should detect the root `Dockerfile` (see `railway.json`).
4. **Variables** (example):
   - `DATA_DIR=/data`
   - `GEMINI_API_KEY=…`
   - `FREESOUND_API_KEY=…`
   - `ALLOWED_ORIGINS=https://your-app.vercel.app` (comma-separated if multiple; omit or use `*` only for testing)
5. **Volumes** → add a volume mounted at **`/data`** (e.g. 8–32 GB depending on library size).
6. Deploy. First boot may take a while: the entrypoint seeds `/data` from baked-in `bootstrap/` copies when empty, starts Qdrant, then runs SFX/music embedders only if collections have zero points (fast path when `embeddings.json` + Qdrant raft files were baked in).
7. Copy the public **HTTPS URL** for the service (e.g. `https://your-api.up.railway.app`).

### Frontend (Vercel)

1. [Vercel](https://vercel.com) → **Import** → same GitHub repo.
2. Set **Root Directory** to `web`.
3. **Environment variables**: `NEXT_PUBLIC_API_BASE=https://your-api.up.railway.app` (your Railway URL, no trailing slash).
4. Optionally edit `web/.env.production` as a template only — Vercel env vars override at build time.
5. Deploy and open the Vercel URL.

### Shipping the sound library (Strategy 1 — bake into image)

For an internal MVP, this repo’s **`Dockerfile` copies `data/library/` and `data/embeddings/` into the image** as `/app/bootstrap/…`. The runtime entrypoint copies them onto the Railway volume when `/data` is still empty. That makes the Docker **build** large and slow (expect **10–30+ minutes** and several GB of context) but keeps cold starts predictable.

**Before `docker build` / Railway build:** ensure your machine has `data/library/manifest.json` and a populated `data/embeddings/` (including Qdrant files if you already indexed locally).

**Later:** move the library to object storage (e.g. R2) and sync on boot to shrink the image — not part of this step.

### Library bootstrap (without baking)

If you **exclude** library/embeddings from the image (custom Dockerfile), populate `/data/library` and `/data/embeddings` via Railway shell, rsync, or a one-off sync — then restart so Qdrant sees the files.

### Verify Docker locally

From the repo root (optional; requires Docker and enough disk/RAM):

```bash
docker build --platform linux/amd64 -t ai-sfx-api:test .
```

### Railway CLI (optional)

```bash
brew install railwayapp/railway/railway   # macOS
railway login
railway link   # select project / service
railway shell  # debug inside a running deployment
```

## Project structure

```
ai-sfx/
├── Dockerfile                          # API + Qdrant (Railway)
├── docker-entrypoint.sh
├── railway.json
├── api/                                # FastAPI HTTP layer
│   ├── main.py                         # App + routes
│   ├── jobs.py                         # In-memory jobs + pipeline thread
│   ├── models.py                       # Pydantic schemas
│   ├── storage.py                      # Upload + result paths under data/
│   └── run.py                          # uvicorn entrypoint
├── web/                                # Next.js frontend (Vercel)
│   └── vercel.json
├── process.py                          # End-to-end pipeline entry point
├── src/
│   ├── config.py                       # Paths, constants, env loading
│   ├── presets.py                      # Mood preset definitions
│   ├── preprocessing/
│   │   ├── frame_extractor.py          # FFmpeg → JPEG frames
│   │   ├── audio_extractor.py          # FFmpeg → mono WAV
│   │   ├── scene_detector.py           # PySceneDetect scene cuts
│   │   └── onset_detector.py           # librosa transients
│   ├── analysis/
│   │   ├── prompts.py                  # Gemini system + per-scene prompt
│   │   └── gemini_client.py            # Per-scene parallel Gemini calls
│   ├── library/
│   │   ├── freesound_client.py         # Freesound API wrapper
│   │   ├── downloader.py               # Bulk CC0 SFX download → manifest.json
│   │   ├── embedder.py                 # sentence-transformers + Qdrant index
│   │   └── vector_store.py             # Qdrant client wrapper
│   ├── matching/
│   │   ├── sfx_matcher.py              # Action → top sound resolver
│   │   ├── preset_reranker.py          # Re-rank candidates by preset keywords
│   │   └── timing_adjuster.py          # Snap timestamps to onsets
│   ├── mixing/
│   │   └── mixer.py                    # pydub overlay + LUFS normalization
│   ├── export/
│   │   └── renderer.py                 # FFmpeg mux video + mixed audio → MP4
│   └── utils/
│       ├── logger.py
│       └── ffmpeg_check.py
├── data/
│   ├── library/                        # Downloaded SFX (gitignored)
│   ├── embeddings/                     # Qdrant storage + embeddings.json (gitignored)
│   ├── temp/                           # Per-video working dirs (gitignored)
│   ├── api_uploads/                    # API uploaded videos (gitignored)
│   ├── api_results/                    # API job outputs (gitignored)
│   └── logs/                           # Runtime logs (gitignored; under DATA_DIR)
├── tests/
│   ├── sample_videos/                  # Test inputs (gitignored)
│   └── fixtures/                       # JSON fixtures (e.g., dry-run scene response)
```

## Pipeline overview

`process.py` runs six stages, each cached so re-runs are cheap:

1. Gemini scene analysis → `analysis.json`
2. Audio onset detection → `onsets.json`
3. Match each detected action against the Qdrant SFX library → `match_plan.json`
4. Snap timestamps to onsets within ±200ms → updated `match_plan.json`
5. Mix SFX + ambient + original audio with preset gains, normalize to −14 LUFS → `mixed_audio.wav`
6. Mux original video + mixed audio into final MP4 → `final.mp4`

## Test video

Place a short clip (5–60 seconds) at `tests/sample_videos/test.mp4`. The folder is gitignored so videos won't be committed.

## Status

Phase 0 complete — end-to-end CLI prototype working. **FastAPI backend** (`api/`) is available for local HTTP jobs; Phase 1 continues with a Next.js frontend (22b) and deployment (22c).
