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
| `R2_ENDPOINT_URL` | Optional — Cloudflare R2 S3 API endpoint (`https://….r2.cloudflarestorage.com`). |
| `R2_ACCESS_KEY_ID` | R2 API token access key. |
| `R2_SECRET_ACCESS_KEY` | R2 API token secret. |
| `R2_BUCKET` | Bucket name (objects under prefixes `library/` and `embeddings/`). |
| `R2_SYNC_FORCE` | Set `1` to re-download library + embeddings even if `manifest.json` already exists on the volume. |
| `STORAGE_MODE` | `local` (default) keeps the full library on disk; `r2` fetches sounds on demand into a small LRU cache (use on Railway Hobby's 5 GB volume). |
| `SOUND_CACHE_MAX_MB` | LRU cap for the on-demand sound cache (default `1500`). |

## Evaluation

The eval framework measures sound design quality objectively, so you can prove
a change made things better (or worse).

### Setup
1. Place 10–15 representative test videos in `eval/videos/`.
2. Register each in `eval/test_set.py` with expected SFX count ranges and a recommended preset.

### Run

```bash
python -m eval.run                       # objective metrics only (fast, free)
python -m eval.run --ai-judge            # also run the Gemini AI judge (uses tokens)
python -m eval.run --compare \
    eval/results/eval_A.json \
    eval/results/eval_B.json             # diff two runs
```

Each run writes a timestamped JSON under `eval/results/` (gitignored). Run the
suite before *and* after any significant pipeline change to confirm the change
improved the average objective score.

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
   - `ALLOWED_ORIGINS=https://your-app.netlify.app` (your frontend URL; comma-separated if multiple)
   - **R2 (recommended for deploy from GitHub):** `R2_ENDPOINT_URL`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` — same prefixes as `aws s3 sync`: `library/`, `embeddings/`
5. **Volumes** → add a volume mounted at **`/data`**. Two profiles:
   - **Full library on volume** (any plan ≥ 32 GB): 8–32 GB volume, leave `STORAGE_MODE` unset (defaults to `local`). First boot syncs the whole library from R2.
   - **Small volume / on-demand R2 fetch** (Railway Hobby, 5 GB cap): add `STORAGE_MODE=r2`. Only the Qdrant index + a ~1.5 GB LRU sound cache occupy the volume; individual sounds are pulled from R2 on demand and cached. Total footprint stays around 3–4 GB.
6. Deploy. The entrypoint starts FastAPI immediately and runs bootstrap (R2 sync + Qdrant indexing) in the background, so `/health` returns 200 within seconds and Railway's healthcheck passes even on first boot. Indexing progress logs to `/tmp/bootstrap.log`.
7. Copy the public **HTTPS URL** for the service (e.g. `https://your-api.up.railway.app`).

### Frontend (Vercel)

1. [Vercel](https://vercel.com) → **Import** → same GitHub repo.
2. Set **Root Directory** to `web`.
3. **Environment variables**: `NEXT_PUBLIC_API_BASE=https://your-api.up.railway.app` (your Railway URL, no trailing slash).
4. Optionally edit `web/.env.production` as a template only — Vercel env vars override at build time.
5. Deploy and open the Vercel URL.

### Sound library: R2 (default for Railway + GitHub)

The Docker image **does not** copy `data/library` or `data/embeddings` from the repo (they are not on GitHub). On first boot, if all `R2_*` variables are set, `docker-entrypoint.sh` pulls the data from R2:

- `STORAGE_MODE=local` (default) — `python -m src.utils.r2_sync` mirrors **`s3://$R2_BUCKET/library/`** → `$DATA_DIR/library/` and **`embeddings/`** → `$DATA_DIR/embeddings/`. Suitable when the volume is big enough for the whole library.
- `STORAGE_MODE=r2` — only `embeddings/` and `library/manifest.json` are pulled (small). Individual sound files are fetched on demand at job time via `src/library/r2_fetch.py` into `$DATA_DIR/sound_cache/` and evicted LRU. Suitable for Railway Hobby's 5 GB volume.

Upload your library to R2 with `aws s3 sync` using the same prefixes (`library/`, `embeddings/`).

### Optional: bake into the image

Uncomment the two `COPY data/...` lines in the `Dockerfile` and build where those folders exist; the entrypoint still copies from `/app/bootstrap/` to `$DATA_DIR` when the volume is empty.

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
