import logging
from pathlib import Path


LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
_VIDEO_LOGGERS: dict[str, logging.Logger] = {}


def get_processing_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("processing")
    if not logger.handlers:
        handler = logging.FileHandler(output_dir / "processing.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_video_logger(video_name: str, output_dir: Path) -> logging.Logger:
    if video_name not in _VIDEO_LOGGERS:
        logger = logging.getLogger(f"video.{video_name}")
        handler = logging.FileHandler(
            output_dir / f"{video_name}.log", encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        _VIDEO_LOGGERS[video_name] = logger
    return _VIDEO_LOGGERS[video_name]


def get_unprocessed_files(
    input_path: Path, input_filetype: str, logging_dir: Path
) -> list[Path]:
    """
    Get list of files in input_path that have not yet been processed,
    based on the presence of log files in logging_dir.
    """
    all_files = list(input_path.glob(f"*.{input_filetype}"))
    processed_files = {
        log_file.stem.replace(".log", "") for log_file in logging_dir.glob("*.log")
    }
    unprocessed_files = [file for file in all_files if file.name not in processed_files]
    return unprocessed_files
