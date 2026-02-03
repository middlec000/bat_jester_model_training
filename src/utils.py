import logging
from pathlib import Path
import shutil
import subprocess
from fractions import Fraction
from datetime import datetime


LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_VIDEO_LOGGERS: dict[str, logging.Logger] = {}
_RUN_TS: str | None = None


def _ensure_run_ts() -> None:
    """Ensure a per-process run timestamp is set (used to create per-run log filenames)."""
    global _RUN_TS
    if _RUN_TS is None:
        _RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def get_processing_logger(
    output_dir: Path, run_name: str = "processing"
) -> logging.Logger:
    """Return a logger that writes to a timestamped per-run file in output_dir.

    Example filename: 20260202_150309_processing.log
    """
    _ensure_run_ts()
    logger_name = f"processing.{_RUN_TS}"
    logger = logging.getLogger(logger_name)
    if not logger.handlers:
        log_path = output_dir / f"{_RUN_TS}_{run_name}.log"
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_unprocessed_files(
    input_dir: Path, input_filetype: str, output_dir: Path, output_filetype: str
) -> list[Path]:
    """Return a list of input files that do not have a corresponding output file.

    To support timestamped run log files (and other output naming conventions), this
    checks whether *any* file in output_dir contains the input file stem and has the
    requested extension. This makes it robust to per-run timestamped filenames.
    """
    input_files = sorted(input_dir.glob(f"*.{input_filetype}"))
    unprocessed: list[Path] = []
    for f in input_files:
        stem = f.stem
        # Consider processed if any file in output_dir contains the stem and has the desired extension
        processed = any(output_dir.glob(f"*{stem}*.{output_filetype}"))
        if not processed:
            unprocessed.append(f)
    return unprocessed


def get_video_fps(video_path: Path) -> float | None:
    """Return FPS (as float) for video using ffprobe, or None if it cannot be determined."""
    if not shutil.which("ffprobe"):
        return None

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return None

    out = proc.stdout.strip()
    if not out:
        return None

    try:
        return float(Fraction(out))
    except Exception:
        return None


def video_has_audio(video_path: Path) -> bool:
    if not shutil.which("ffprobe"):
        return False
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        return False
    return bool(proc.stdout.strip())
