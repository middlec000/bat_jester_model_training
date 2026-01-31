from pathlib import Path
from time import time
import sys
import os

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import logging_setup

INPUT_DIR = Path(
    "~/data/bat_jester_model_training/D_completely_xy_labeled_clips"
).expanduser()
OUTPUT_DIR = Path(
    "~/data/bat_jester_model_training/F_audio_extracted_from_videos"
).expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/5_logs").expanduser()


def main():
    # Parse command-line arguments
    run_all = "--run-all" in sys.argv

    print(f"Separate audio from video (--run-all: {run_all})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging_setup.get_processing_logger(LOGGING_DIR)

    input_files = [
        x for x in list(INPUT_DIR.glob("*.mp4")) if "_annotated_" not in x.stem
    ]
    if run_all:
        unprocessed_files = input_files
        logger.info(
            f"Processing all files (--run-all flag set): {len(unprocessed_files)} files"
        )
    else:
        unprocessed_files = logging_setup.get_unprocessed_files(
            input_dir=INPUT_DIR,
            input_filetype="mp4",
            output_dir=OUTPUT_DIR,
            output_filetype="wav",
        )
        logger.info(f"Processing unprocessed files: {len(unprocessed_files)} files")

    for video_file in unprocessed_files:
        output_file = OUTPUT_DIR / video_file.with_suffix(".wav").name

        start_time = time()
        # Use ffmpeg to extract audio
        command = (
            f'ffmpeg -i "{video_file}" -q:a 0 -map a "{output_file}" -y -loglevel panic'
        )
        return_code = os.system(command)
        if return_code != 0:
            logger.error(f"Failed to extract audio from {video_file.name}")
            continue

        elapsed_time = time() - start_time
        logger.info(
            f"Extracted audio from {video_file.name} in {elapsed_time:.2f} seconds and saved to {output_file.name}"
        )


if __name__ == "__main__":
    main()
