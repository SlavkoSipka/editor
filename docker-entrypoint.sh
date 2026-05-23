#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
STORAGE_MODE="${STORAGE_MODE:-local}"
export DATA_DIR STORAGE_MODE

mkdir -p "$DATA_DIR/embeddings"
mkdir -p "$DATA_DIR/library"
mkdir -p "$DATA_DIR/api_uploads"
mkdir -p "$DATA_DIR/api_results"
mkdir -p "$DATA_DIR/temp"
mkdir -p "$DATA_DIR/logs"
mkdir -p "$DATA_DIR/sound_cache"

# QDRANT_RESET=1 wipes incompatible/corrupt collection files before Qdrant
# starts. Useful one-shot recovery via Railway env vars when the Qdrant binary
# crashes at startup (e.g., version mismatch from a prior R2 sync). Toggle it
# off again after a successful boot.
if [ "${QDRANT_RESET:-}" = "1" ]; then
  echo "QDRANT_RESET=1 — clearing \$DATA_DIR/embeddings/collections and .indexed"
  rm -rf "$DATA_DIR/embeddings/collections"
  rm -f  "$DATA_DIR/embeddings/.indexed"
fi

echo "Starting Qdrant (storage = \$DATA_DIR/embeddings, STORAGE_MODE=$STORAGE_MODE)..."
QDRANT__STORAGE__STORAGE_PATH="$DATA_DIR/embeddings" \
QDRANT__SERVICE__HTTP_PORT=6333 \
QDRANT__SERVICE__GRPC_PORT=6334 \
qdrant >/tmp/qdrant.log 2>&1 &

# Wait briefly for Qdrant, then bootstrap (R2 sync / embedder) in background
# so /health is reachable immediately and Railway's healthcheck doesn't time
# out while we (re-)build the index.
bootstrap() {
  echo "Waiting for Qdrant..."
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:6333/healthz" >/dev/null 2>&1 || curl -sf "http://127.0.0.1:6333/" >/dev/null 2>&1; then
      echo "Qdrant is ready."
      break
    fi
    sleep 1
  done

  R2_READY=0
  if [ -n "${R2_BUCKET:-}" ] && [ -n "${R2_ENDPOINT_URL:-}" ] && \
     [ -n "${R2_ACCESS_KEY_ID:-}" ] && [ -n "${R2_SECRET_ACCESS_KEY:-}" ]; then
    R2_READY=1
  fi

  if [ "$STORAGE_MODE" = "r2" ]; then
    if [ "$R2_READY" != "1" ]; then
      echo "ERROR: STORAGE_MODE=r2 but R2_* env vars are missing — aborting bootstrap."
      return 1
    fi

    # In R2 mode the full library never lives on disk: pull only the
    # embeddings.json + manifest (small) so we can rebuild the Qdrant index.
    if [ ! -f "$DATA_DIR/embeddings/embeddings.json" ] || [ "${R2_SYNC_FORCE:-}" = "1" ]; then
      echo "R2: syncing embeddings prefix (small, vectors only)..."
      python -m src.utils.r2_sync --prefix embeddings --dest "$DATA_DIR/embeddings" \
        || echo "WARNING: R2 embeddings sync failed"
    else
      echo "R2: embeddings.json already present — skipping."
    fi

    if [ ! -f "$DATA_DIR/library/manifest.json" ] || [ "${R2_SYNC_FORCE:-}" = "1" ]; then
      echo "R2: fetching library/manifest.json..."
      python -c "
from pathlib import Path
from src.library.r2_fetch import fetch_object
dest = Path('$DATA_DIR/library/manifest.json')
ok = fetch_object('library/manifest.json', dest)
print('manifest:', 'ok' if ok else 'FAILED')
" || echo "WARNING: manifest fetch failed"
    fi
  else
    # Local / baked-image mode — keep the existing full-sync behavior.
    if [ "$R2_READY" = "1" ]; then
      if [ ! -f "$DATA_DIR/library/manifest.json" ] || [ "${R2_SYNC_FORCE:-}" = "1" ]; then
        echo "R2: syncing library/ and embeddings/ into \$DATA_DIR (full library)..."
        python -m src.utils.r2_sync --prefix library --dest "$DATA_DIR/library" \
          || echo "WARNING: R2 library sync failed"
        python -m src.utils.r2_sync --prefix embeddings --dest "$DATA_DIR/embeddings" \
          || echo "WARNING: R2 embeddings sync failed"
      else
        echo "R2: credentials set; library already on volume — skipping sync."
      fi
    else
      echo "R2: missing R2_* env — skipping object sync (use image bootstrap or mount data)."
    fi

    if [ ! -f "$DATA_DIR/library/manifest.json" ] && [ -f "/app/bootstrap/library/manifest.json" ]; then
      echo "Seeding library from image bootstrap..."
      mkdir -p "$DATA_DIR/library"
      cp -a /app/bootstrap/library/. "$DATA_DIR/library/"
    fi
    if [ ! -f "$DATA_DIR/embeddings/embeddings.json" ] && [ -f "/app/bootstrap/embeddings/embeddings.json" ]; then
      echo "Seeding embeddings cache from image bootstrap..."
      cp -a /app/bootstrap/embeddings/. "$DATA_DIR/embeddings/"
    fi
  fi

  collection_points() {
    python3 -c "
import json, sys, urllib.request
name = sys.argv[1]
try:
    req = urllib.request.urlopen(
        f'http://127.0.0.1:6333/collections/{name}', timeout=5
    )
    data = json.load(req)
    print(int(data.get('result', {}).get('points_count', 0)))
except Exception:
    print(0)
" "$1"
  }

  EMBED_ARGS=""
  if [ "$STORAGE_MODE" = "r2" ]; then
    EMBED_ARGS="--from-cache"
  fi

  if [ -f "$DATA_DIR/embeddings/embeddings.json" ] || [ -f "$DATA_DIR/library/manifest.json" ]; then
    SFX_PTS=$(collection_points sfx_library)
    if [ "${SFX_PTS:-0}" -eq 0 ]; then
      echo "Indexing SFX collection (mode=$STORAGE_MODE)..."
      if python -m src.library.embedder $EMBED_ARGS; then
        touch "$DATA_DIR/embeddings/.indexed"
      else
        echo "WARNING: SFX embedder failed — check logs"
      fi
    else
      echo "SFX collection already has $SFX_PTS points — skipping embedder."
    fi

    # Music collection is optional and only ever built from local manifest;
    # skip in pure R2 mode unless someone explicitly seeded the music manifest.
    if [ -f "$DATA_DIR/library/music/manifest.json" ]; then
      MUSIC_PTS=$(collection_points music_library)
      if [ "${MUSIC_PTS:-0}" -eq 0 ]; then
        echo "Indexing music collection..."
        python -m src.library.music_embedder || echo "WARNING: music embedder failed"
      else
        echo "Music collection already has $MUSIC_PTS points — skipping music embedder."
      fi
    fi
  else
    echo "WARNING: No embeddings.json or manifest.json found — Qdrant will be empty."
  fi

  echo "Bootstrap complete."
}

bootstrap >/tmp/bootstrap.log 2>&1 &
echo "Bootstrap running in background (tail /tmp/bootstrap.log for progress)."

PORT="${PORT:-8000}"
echo "Starting FastAPI on port $PORT..."
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
