# AI Sound Effects Generator — Project Plan

## What we're building

A web application that allows video editors to upload short-form videos (Instagram ads, TikTok ads, Reels — max 3 minutes) and automatically receive the same video with AI-placed sound effects. The user picks a mood preset (Dramatic, Energetic, Mysterious, Playful, Luxury) and the system analyzes the video, detects events (door slams, footsteps, transitions, impacts, whooshes), matches them with sound effects from a local library, and outputs a finished MP4 plus an optional XML/OTIO file that can be imported into Premiere Pro or DaVinci Resolve with all SFX on separate tracks.

## Target user

Professional editors making short-form ads for Instagram, TikTok, Reels. Not feature films. Sweet spot is 15-90 second videos.

## Core value proposition

The differentiator is **automatic placement of multiple separate SFX layers on a timeline**, exported as separate tracks editors can fine-tune. Existing tools either generate one merged audio track (MMAudio, ElevenLabs, Pika) or only suggest sounds without placing them (Epidemic Sound, Artlist). We do both — automatic placement AND editor-friendly multi-track export.

---

## How the system works (high-level pipeline)

1. **Upload** — User uploads MP4 (max 3min), picks a preset.
2. **Preprocessing** — FFmpeg extracts frames at 2 fps + original WAV audio.
3. **Scene detection** — PySceneDetect splits video into 3-15 logical scenes.
4. **Audio onset detection** — librosa analyzes original audio and finds timestamps of transients. Used as timing hints when audio is informative; skipped when audio is silent or just music.
5. **Vision analysis (parallel per scene)** — Gemini 2.5 Flash receives 3-5 frames per scene + onset hints + scene timing. Returns structured JSON: list of actions with timestamps, mood, environment, suggested SFX categories.
6. **SFX plan generation** — Combine all scenes into a single timeline plan as JSON (the "blueprint" before any rendering).
7. **SFX matching** — For each planned action, generate text embedding, search local vector database (Qdrant) for top 3 matching sounds.
8. **Style filter (preset)** — Re-rank candidates based on chosen preset (Dramatic prefers deeper/longer reverb; Energetic prefers sharp/percussive; etc).
9. **Timing adjustment** — When onset detection found a nearby transient (within 200ms of Gemini's estimated timestamp), snap to it. Otherwise keep Gemini's estimate.
10. **Mix engineering (algorithmic, not AI)** — Normalize loudness (-14 LUFS), apply ducking, panning, reverb tail per preset.
11. **Render** — FFmpeg combines original video + original audio + SFX layers → final MP4. Also generate OTIO file → convert to FCPXML and Premiere XML for editor import.

---

## Tech stack

- **Language:** Python 3.11+ for backend pipeline
- **Frontend (later):** Next.js 14 + React + Tailwind
- **Backend API (later):** FastAPI
- **Video/Audio:** FFmpeg, PySceneDetect, librosa, pydub, pyloudnorm
- **AI Vision:** Google Gemini 2.5 Flash (model id: `gemini-2.5-flash`)
- **Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`, runs locally, free)
- **Vector DB:** Qdrant (self-hosted in Docker, free)
- **SFX library:** Freesound.org API (CC0 filter only) — primary source for prototype
- **Timeline export:** OpenTimelineIO (Python library)
- **Storage (later):** Cloudflare R2

## API keys needed

- `GEMINI_API_KEY` — Google AI Studio
- `FREESOUND_API_KEY` — freesound.org/apiv2

Stored in `.env` file, never committed.

---

## Project structure

```
ai-sfx/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── process.py                    # Main CLI entry point
├── src/
│   ├── __init__.py
│   ├── config.py                 # Constants, paths, env loading
│   ├── presets.py                # Preset definitions (one source of truth)
│   ├── preprocessing/
│   ├── analysis/
│   ├── library/
│   ├── matching/
│   ├── mixing/
│   ├── export/
│   └── utils/
├── data/
│   ├── library/                  # Downloaded SFX (gitignored)
│   ├── embeddings/               # Cached embeddings (gitignored)
│   └── temp/                     # Working directory (gitignored)
├── tests/
│   ├── sample_videos/            # 5 test videos with expected output
│   └── fixtures/                 # Expected JSON outputs for regression
└── logs/                         # Debug logs (gitignored)
```

---

## Mood presets (preliminary)

Each preset is a Python dataclass with re-ranking keywords and mix parameters.

- **Dramatic** — deep impacts, long reverb tails, low-end emphasis, sparse and impactful
- **Energetic** — sharp percussive transitions, whooshes, fast pace, bright frequencies, dense layering
- **Mysterious** — ambient drones, reverse swells, subtle textures, moderate reverb, sparse SFX
- **Playful** — bouncy cartoon-ish hits, plucks, whistles, light reverb, comedic timing
- **Luxury** — clean minimal SFX, soft impacts, subtle ambient, polished and restrained

---

## Development phases

### Phase 0 — Validation prototype (CLI only)
Goal: prove the pipeline works before building UI. Python CLI script that takes a video file and preset, outputs MP4 with SFX. Tested on 20-30 sample ads. No frontend, no auth, no billing.

### Phase 1 — MVP web app
After Phase 0 validates with real editors. Next.js frontend + FastAPI backend wrapping the Phase 0 pipeline.

### Phase 2 — Premium features
OTIO/XML export polish, Pro preset packs, Epidemic Sound API integration, team accounts.

---

## Build order (Phase 0) — STRICTLY incremental

Build ONE step at a time. After each step, the user manually tests, confirms it works, and only THEN we move to the next step. Do not build ahead.

1. **Project setup** — venv, requirements.txt, folder structure, .env, .gitignore, FFmpeg detection helper
2. **Frame + audio extraction** — given video.mp4, output frames/ folder and audio.wav
3. **Scene detection** — list of scene boundaries with timestamps
4. **Onset detection** — list of audio event timestamps (best-effort; may be empty for silent/music-only audio)
5. **Gemini integration** — send frames per scene, get JSON of actions/mood/SFX categories. Includes retry logic, JSON validation, and `--dry-run` mode that uses a fixture instead of calling the API.
6. **VALIDATION CHECKPOINT** — print JSON for 5 test videos, manually check quality. **Iterate prompts until output is consistently good. Do not proceed past this point until the user confirms quality is acceptable.**
7. **Freesound bulk download script** — fetch ~5,000 top-rated CC0 sounds with metadata, save locally with manifest.json
8. **Embedding pipeline** — embed all sound descriptions with sentence-transformers, store in local Qdrant
9. **SFX matcher** — given action description, return top 3 matching sounds from Qdrant
10. **Preset re-ranker** — apply preset rules to SFX candidates
11. **Timing adjustment** — snap to nearest onset within 200ms window
12. **Mixer** — layer SFX over original audio with pydub, apply ducking and LUFS normalization
13. **Renderer** — FFmpeg final MP4 output
14. **End-to-end test** — `python process.py video.mp4 --preset dramatic` → working output
15. **Timeline export (OTIO/FCPXML/Premiere XML)** — last because it's complex and not needed for first demo

---

## Critical rules for development

- **One step at a time.** Don't build later steps until earlier ones work.
- **Test each module standalone** before integrating. Each module has a `if __name__ == "__main__"` test block.
- **Always implement `--dry-run` for steps that hit external APIs.** Use a saved fixture so we can test pipeline logic without spending tokens.
- **Validate Gemini output quality early.** If it doesn't recognize actions well, the whole project fails. Be ready to iterate prompts heavily.
- **Onset sync is best-effort.** Use it when available, gracefully fall back to Gemini timestamps when audio is uninformative.
- **Library quality > quantity.** 5,000 well-curated CC0 sounds beat 100,000 noisy ones. Filter by rating, downloads, duration.
- **Keep mixing rules deterministic.** Mix engineering is algorithmic, not AI. Predictable rules = predictable output.
- **Cache aggressively.** Embeddings, downloaded sounds, scene detection results — never recompute what we already have.
- **Log everything to `logs/`.** Every stage writes its output for debugging.
- **Generate the SFX plan as JSON BEFORE rendering.** Plan is cheap, render is expensive. User can inspect/iterate plan before committing to render.
- **Pin dependency versions** in requirements.txt to avoid librosa/numpy/scipy version conflicts.

---

## What we are NOT building (yet)

- Frontend / UI
- User accounts / authentication
- Billing / subscriptions
- Plugin for Premiere or any editor
- Mobile app
- Generative audio fallback (ElevenLabs)
- Real-time processing
- Collaboration features

These come later. Phase 0 is CLI only.