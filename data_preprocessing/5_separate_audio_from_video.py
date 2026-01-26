from pathlib import Path
from time import time
import sys
import os

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent))

from bat_logging import logging_setup

INPUT_DIR = Path("data/D_completely_xy_labeled_clips")
OUTPUT_DIR = Path("data/F_audio_extracted_from_videos")
LOGGING_DIR = Path("data/5_logs")

logger = logging_setup.get_processing_logger(LOGGING_DIR)

input_files = [x for x in list(INPUT_DIR.glob("*.mp4")) if "_annotated_" not in x.stem]
processed_files = list(OUTPUT_DIR.glob("*.wav"))
unprocessed_files = [
    f
    for f in input_files
    if (OUTPUT_DIR / f.with_suffix(".wav").name) not in processed_files
]

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
