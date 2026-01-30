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


def get_file_logger(file_name: str, output_dir: Path) -> logging.Logger:
    if file_name not in _VIDEO_LOGGERS:
        logger = logging.getLogger(f"file.{file_name}")
        handler = logging.FileHandler(output_dir / f"{file_name}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT))
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        _VIDEO_LOGGERS[file_name] = logger
    return _VIDEO_LOGGERS[file_name]


def get_unprocessed_files(
    input_dir: Path, input_filetype: str, output_dir: Path, output_filetype: str
) -> list[Path]:
    """
    Get list of files in input_dir that have not yet been processed,
    based on the presence of log files in logging_dir.
    """
    input_files = {f.stem: f for f in input_dir.glob(f"*.{input_filetype}")}
    output_files = set(f.stem for f in output_dir.glob(f"*.{output_filetype}"))
    unprocessed_files = [
        input_files[stem] for stem in input_files if stem not in output_files
    ]
    return unprocessed_files
