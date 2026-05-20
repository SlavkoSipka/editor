"""Sonniss GameAudioGDC importer.

User must manually download archives from:
  https://sonniss.com/gameaudiogdc

Place .zip/.tar archives in: data/sonniss_raw/

This script then:
1. Extracts all archives
2. Walks every .wav/.mp3/.flac file
3. Converts to MP3 in data/library/sonniss/{stem}.mp3
4. Builds metadata from filename + folder path (acts as tags)
5. Merges into data/library/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import DATA_DIR, LIBRARY_DIR
from src.utils.logger import get_logger

logger = get_logger("sonniss_importer")

SONNISS_RAW_DIR: Path = DATA_DIR / "sonniss_raw"
SONNISS_LIBRARY_DIR: Path = LIBRARY_DIR / "sonniss"

_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".aif", ".aiff", ".ogg", ".m4a"}
_MIN_DURATION_SEC = 0.1
_MAX_DURATION_SEC = 60.0
_MAX_WORKERS = 4
_TOKEN_RE = re.compile(r"[^A-Za-z0-9]+")

_PREFLIGHT_HINT = (
    "Sonniss archives not found in {raw_dir}.\n\n"
    "To use Sonniss sounds (~15k professional SFX):\n"
    "1. Go to https://sonniss.com/gameaudiogdc\n"
    "2. Download free yearly archives (2020-2024 recommended)\n"
    "3. Place .zip files in {raw_dir}\n"
    "4. Re-run this command.\n\n"
    "Skipping Sonniss import."
)


def _stable_id(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:6], "big") % (10**9)


def _clean_text(value: str) -> str:
    tokens = [t.lower() for t in _TOKEN_RE.split(value) if t]
    return " ".join(tokens)


def _tags_from(relative_path: Path) -> list[str]:
    pieces: list[str] = []
    for part in list(relative_path.parts[:-1]) + [relative_path.stem]:
        pieces.extend(t for t in _TOKEN_RE.split(part) if t)
    seen: set[str] = set()
    out: list[str] = []
    for tok in pieces:
        low = tok.lower()
        if low in seen or len(low) < 2:
            continue
        seen.add(low)
        out.append(low)
    return out


def _probe_duration(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None
    out = (result.stdout or "").strip()
    if not out:
        return None
    try:
        return float(out)
    except ValueError:
        return None


def _convert_to_mp3(src: Path, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        return True
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", str(src),
                "-ac", "2", "-ar", "44100",
                "-codec:a", "libmp3lame", "-b:a", "192k",
                str(tmp),
            ],
            check=True, capture_output=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return False
    tmp.replace(dest)
    return True


def _process_one(args: tuple[str, str, str]) -> dict[str, Any] | None:
    """Worker: convert a single audio file and return a manifest entry, or None.

    args = (src_path, rel_path_posix, target_dir_str)
    """
    src_str, rel_str, target_dir_str = args
    src = Path(src_str)
    rel = Path(rel_str)
    target_dir = Path(target_dir_str)

    duration = _probe_duration(src)
    if duration is None or duration < _MIN_DURATION_SEC or duration > _MAX_DURATION_SEC:
        return None

    stem = rel.stem
    sid = _stable_id(rel.as_posix())
    mp3_name = f"{sid}_{_TOKEN_RE.sub('_', stem)[:60].strip('_') or 'sonniss'}.mp3"
    dest = target_dir / mp3_name

    if not _convert_to_mp3(src, dest):
        return None

    folder_text = _clean_text(rel.parent.as_posix())
    name_text = _clean_text(stem)
    description = folder_text or name_text

    return {
        "id": sid,
        "name": stem.replace("_", " ").strip() or "sonniss sound",
        "description": description,
        "tags": _tags_from(rel),
        "duration": duration,
        "license": "Sonniss GDC Royalty-Free",
        "local_path": f"data/library/sonniss/{mp3_name}",
        "source": "sonniss",
        "original_path": rel.as_posix(),
    }


def _extract_archive(archive: Path, dest: Path) -> bool:
    try:
        if archive.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(dest)
            return True
        # Fall back to shutil for .tar/.tar.gz and similar.
        shutil.unpack_archive(str(archive), str(dest))
        return True
    except (zipfile.BadZipFile, shutil.ReadError, OSError) as exc:
        logger.error("Failed to extract %s: %s", archive.name, exc)
        return False


def _gather_audio_files(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _AUDIO_EXTS:
            continue
        pairs.append((path, path.relative_to(root)))
    return pairs


def _merge_into_main_manifest(new_entries: list[dict[str, Any]]) -> Path:
    manifest_path = LIBRARY_DIR / "manifest.json"
    existing: list[dict[str, Any]] = []
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except json.JSONDecodeError:
            logger.warning("Existing manifest unreadable; overwriting.")
            existing = []

    kept = [e for e in existing if not (isinstance(e, dict) and e.get("source") == "sonniss")]
    merged = kept + new_entries
    manifest_path.write_text(json.dumps(merged, indent=2))
    logger.info(
        "Merged manifest: %d non-sonniss + %d sonniss = %d total → %s",
        len(kept), len(new_entries), len(merged), manifest_path,
    )
    return manifest_path


def _find_archives(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        return []
    archives: list[Path] = []
    for ext in (".zip", ".tar", ".tar.gz", ".tgz"):
        archives.extend(sorted(raw_dir.glob(f"*{ext}")))
    return archives


def import_sonniss_archives(
    sonniss_raw_dir: Path = SONNISS_RAW_DIR,
    target_dir: Path = SONNISS_LIBRARY_DIR,
) -> int:
    """Extract all archives in sonniss_raw_dir, convert files, generate metadata.
    Returns count of imported sounds.
    """
    archives = _find_archives(sonniss_raw_dir)
    if not archives:
        print(_PREFLIGHT_HINT.format(raw_dir=sonniss_raw_dir))
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Found %d Sonniss archive(s) in %s", len(archives), sonniss_raw_dir)

    all_entries: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="sonniss_") as tmp_root_str:
        tmp_root = Path(tmp_root_str)

        for archive in archives:
            extract_dir = tmp_root / archive.stem
            extract_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Extracting %s ...", archive.name)
            if not _extract_archive(archive, extract_dir):
                continue

            audio_pairs = _gather_audio_files(extract_dir)
            logger.info("  %d audio files in %s", len(audio_pairs), archive.name)
            if not audio_pairs:
                continue

            tasks = [
                (str(src), rel.as_posix(), str(target_dir))
                for src, rel in audio_pairs
            ]

            with ProcessPoolExecutor(max_workers=_MAX_WORKERS) as pool:
                futures = [pool.submit(_process_one, task) for task in tasks]
                for fut in tqdm(
                    as_completed(futures), total=len(futures),
                    desc=f"Converting {archive.name}", unit="snd",
                ):
                    try:
                        entry = fut.result()
                    except Exception as exc:
                        logger.error("Worker error: %s", exc)
                        continue
                    if entry is not None:
                        all_entries.append(entry)

    seen_ids: set[int] = set()
    deduped: list[dict[str, Any]] = []
    for entry in all_entries:
        if entry["id"] in seen_ids:
            continue
        seen_ids.add(entry["id"])
        deduped.append(entry)

    logger.info(
        "Imported %d Sonniss sounds (%d duplicates removed)",
        len(deduped), len(all_entries) - len(deduped),
    )

    if deduped:
        _merge_into_main_manifest(deduped)

    return len(deduped)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a locally-downloaded Sonniss GameAudioGDC archive set.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=SONNISS_RAW_DIR,
        help="Folder containing the downloaded Sonniss .zip archives.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=SONNISS_LIBRARY_DIR,
        help="Where to write the converted MP3 files.",
    )
    args = parser.parse_args()

    try:
        subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        print(
            "ffmpeg not found. Install ffmpeg first (e.g. `brew install ffmpeg`).",
            file=sys.stderr,
        )
        sys.exit(1)

    count = import_sonniss_archives(args.raw_dir, args.target_dir)
    print(f"Imported {count} Sonniss sounds.")


if __name__ == "__main__":
    _main()
