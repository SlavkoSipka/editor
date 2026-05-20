#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
export DATA_DIR

mkdir -p "$DATA_DIR/embeddings"
mkdir -p "$DATA_DIR/library"
mkdir -p "$DATA_DIR/api_uploads"
mkdir -p "$DATA_DIR/api_results"
mkdir -p "$DATA_DIR/temp"
mkdir -p "$DATA_DIR/logs"

# Seed persistent volume from image bootstrap (first boot on Railway).
if [ ! -f "$DATA_DIR/library/manifest.json" ] && [ -f "/app/bootstrap/library/manifest.json" ]; then
  echo "Seeding library from image bootstrap..."
  mkdir -p "$DATA_DIR/library"
  cp -a /app/bootstrap/library/. "$DATA_DIR/library/"
fi

if [ ! -f "$DATA_DIR/embeddings/embeddings.json" ] && [ -f "/app/bootstrap/embeddings/embeddings.json" ]; then
  echo "Seeding embeddings cache from image bootstrap..."
  cp -a /app/bootstrap/embeddings/. "$DATA_DIR/embeddings/"
fi

echo "Starting Qdrant (storage = \$DATA_DIR/embeddings, same layout as local Docker volume)..."
QDRANT__STORAGE__STORAGE_PATH="$DATA_DIR/embeddings" \
QDRANT__SERVICE__HTTP_PORT=6333 \
QDRANT__SERVICE__GRPC_PORT=6334 \
qdrant >/tmp/qdrant.log 2>&1 &

echo "Waiting for Qdrant..."
for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:6333/healthz" >/dev/null 2>&1 || curl -sf "http://127.0.0.1:6333/" >/dev/null 2>&1; then
    echo "Qdrant is ready."
    break
  fi
  sleep 1
done

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

if [ -f "$DATA_DIR/library/manifest.json" ]; then
  SFX_PTS=$(collection_points sfx_library)
  if [ "${SFX_PTS:-0}" -eq 0 ]; then
    echo "Indexing SFX collection (uses embeddings.json cache when present)..."
    if python -m src.library.embedder; then
      touch "$DATA_DIR/embeddings/.indexed"
    else
      echo "WARNING: SFX embedder failed — check logs"
    fi
  else
    echo "SFX collection already has $SFX_PTS points — skipping embedder."
  fi

  if [ -f "$DATA_DIR/library/music/manifest.json" ]; then
    MUSIC_PTS=$(collection_points music_library)
    if [ "${MUSIC_PTS:-0}" -eq 0 ]; then
      echo "Indexing music collection..."
      python -m src.library.music_embedder || echo "WARNING: music embedder failed — pipeline may run without music bed"
    else
      echo "Music collection already has $MUSIC_PTS points — skipping music embedder."
    fi
  fi
else
  echo "WARNING: No library manifest at \$DATA_DIR/library/manifest.json — API health will show library_loaded=false until you add one."
fi

PORT="${PORT:-8000}"
echo "Starting FastAPI on port $PORT..."
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"
