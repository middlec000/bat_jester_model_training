from pathlib import Path
from time import time
import sys
import polars as pl

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent))

from bat_logging import logging_setup

INPUT_DIR = Path("data/D_completely_xy_labeled_clips")
OUTPUT_DIR = Path("data/E_juggle_labels")
LOGGING_DIR = Path("data/4_logs")


FRAMES_PER_SECOND = 30.0  # 29.78150102817087

logger = logging_setup.get_processing_logger(LOGGING_DIR)

input_files = list(INPUT_DIR.glob("*.parquet"))
processed_files = list(OUTPUT_DIR.glob("*.parquet"))
unprocessed_files = [
    f for f in input_files if (OUTPUT_DIR / f.name) not in processed_files
]

for xy_label_file in unprocessed_files:
    output_file = OUTPUT_DIR / xy_label_file.name
    output_txt = OUTPUT_DIR / f"{xy_label_file.stem}_juggle_timestamps.txt"

    start_time = time()
    df = pl.read_parquet(xy_label_file)

    # Calculate vertical ball velocity (change in y position)
    df = df.with_columns(
        pl.col("y").diff().alias("vertical_velocity"),
        pl.int_range(0, pl.len()).alias("frame_index"),
        (pl.int_range(0, pl.len()) / FRAMES_PER_SECOND).alias("time_seconds"),
    )

    # Detect juggle timestamps: when ball transitions from downward to upward
    # Downward velocity is positive (y increases), upward is negative (y decreases)
    df = df.with_columns(
        pl.when(pl.col("vertical_velocity") < 0)
        .then(pl.lit("upward"))
        .when(pl.col("vertical_velocity") > 0)
        .then(pl.lit("downward"))
        .otherwise(pl.lit(None))
        .alias("velocity_direction")
    )

    # Detect transitions from downward to upward
    df = df.with_columns(
        (
            (pl.col("velocity_direction") == "upward")
            & (pl.col("velocity_direction").shift(1) == "downward")
        ).alias("is_juggle_timestamp")
    )

    df.write_parquet(output_file)
    with open(output_txt, "w") as f:
        juggle_timestamps = df.filter(pl.col("is_juggle_timestamp")).select(
            pl.col("time_seconds")
        )
        for ts in juggle_timestamps["time_seconds"]:
            f.write(f"{ts:.3f}\n")
    elapsed_time = time() - start_time
    logger.info(
        f"Processed {xy_label_file.name} in {elapsed_time:.2f} seconds and saved to {output_file.name} and {output_txt.name}"
    )
