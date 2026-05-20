from __future__ import annotations

import subprocess


_INSTALL_HINT = (
    "FFmpeg not found. Install: "
    "macOS `brew install ffmpeg`, "
    "Ubuntu `apt install ffmpeg`, "
    "Windows: download from ffmpeg.org"
)


def check_ffmpeg() -> str:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(_INSTALL_HINT) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"FFmpeg returned non-zero exit code: {exc.returncode}. {_INSTALL_HINT}"
        ) from exc

    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    return first_line.strip()


if __name__ == "__main__":
    print(check_ffmpeg())
