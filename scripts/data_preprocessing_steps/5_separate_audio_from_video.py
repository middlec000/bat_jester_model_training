from pathlib import Path
from time import time
import sys
import os
import argparse

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import utils

INPUT_DIR = Path(
    "~/data/bat_jester_model_training/D_completely_xy_labeled_clips"
).expanduser()
OUTPUT_DIR = Path(
    "~/data/bat_jester_model_training/F_audio_extracted_from_videos"
).expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/5_logs").expanduser()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Extract audio from videos")
    parser.add_argument(
        "--run",
        nargs="+",
        default=["new"],
        help='Run mode: "all" to process all files, "new" to process only unprocessed files (default), or provide one or more substrings to process all files whose names contain any substring',
    )
    args = parser.parse_args()

    run_arg = args.run
    if len(run_arg) == 1 and run_arg[0] in ("all", "new"):
        run_mode = run_arg[0]
        substrings = None
    else:
        run_mode = "substr"
        substrings = run_arg

    print(f"Separate audio from video (run_mode: {run_mode}, substrings: {substrings})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    logger = utils.get_processing_logger(LOGGING_DIR)

    input_files = [
        x for x in list(INPUT_DIR.glob("*.mp4")) if "_annotated_" not in x.stem
    ]
    if run_mode == "all":
        unprocessed_files = input_files
        logger.info(
            f"Processing all files (run_mode 'all'): {len(unprocessed_files)} files"
        )
    elif run_mode == "new":
        unprocessed_files = utils.get_unprocessed_files(
            input_dir=INPUT_DIR,
            input_filetype="mp4",
            output_dir=OUTPUT_DIR,
            output_filetype="wav",
        )
        logger.info(f"Processing unprocessed files: {len(unprocessed_files)} files")
    else:
        candidate_files = input_files
        unprocessed_files = [
            f for f in candidate_files if any(s in f.name for s in substrings)
        ]
        logger.info(
            f"Filtering with substrings=%s: {len(candidate_files)} -> {len(unprocessed_files)} files",
            substrings,
        )

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
