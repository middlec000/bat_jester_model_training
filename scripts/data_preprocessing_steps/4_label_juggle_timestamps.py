from pathlib import Path
from time import time
import sys
import polars as pl

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import utils

INPUT_DIR = Path(
    "~/data/bat_jester_model_training/D_completely_xy_labeled_clips"
).expanduser()
OUTPUT_DIR = Path("~/data/bat_jester_model_training/E_juggle_labels").expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/4_logs").expanduser()


FALLBACK_FRAMES_PER_SECOND = 30.0


def main():
    # Parse command-line arguments
    run_all = "--run-all" in sys.argv

    print(f"Label juggle timestamps (--run-all: {run_all})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    logger = utils.get_processing_logger(LOGGING_DIR)

    input_files = list(INPUT_DIR.glob("*.parquet"))

    if run_all:
        unprocessed_files = input_files
        logger.info(
            f"Processing all files (--run-all flag set): {len(unprocessed_files)} files"
        )
    else:
        unprocessed_files = utils.get_unprocessed_files(
            input_dir=INPUT_DIR,
            input_filetype="parquet",
            output_dir=OUTPUT_DIR,
            output_filetype="parquet",
        )
        logger.info(f"Processing unprocessed files: {len(unprocessed_files)} files")

    for xy_label_file in unprocessed_files:
        video_file = INPUT_DIR / f"{xy_label_file.stem}.mp4"
        output_file = OUTPUT_DIR / xy_label_file.name
        output_txt = OUTPUT_DIR / f"{xy_label_file.stem}.txt"

        start_time = time()
        df = pl.read_parquet(xy_label_file)

        fps = utils.get_video_fps(video_file) or FALLBACK_FRAMES_PER_SECOND

        # Calculate vertical ball velocity (change in y position)
        df = df.with_columns(
            pl.col("y").diff().alias("vertical_velocity"),
            pl.int_range(0, pl.len()).alias("frame_index"),
            (pl.int_range(0, pl.len()) / fps).alias("time_seconds"),
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

        # Correct velocity_direction: switch isolated values that don't persist for at least 2 frames
        # Create a group identifier for consecutive identical values
        df = df.with_columns(
            (pl.col("velocity_direction") != pl.col("velocity_direction").shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("direction_group")
        )

        # Count frames in each group
        df = df.with_columns(
            pl.col("velocity_direction")
            .count()
            .over("direction_group")
            .alias("group_frame_count")
        )

        # Replace isolated values (groups with < 2 frames) with the other direction
        df = df.with_columns(
            pl.when(pl.col("group_frame_count") < 2)
            .then(
                pl.when(pl.col("velocity_direction") == "upward")
                .then(pl.lit("downward"))
                .when(pl.col("velocity_direction") == "downward")
                .then(pl.lit("upward"))
                .otherwise(pl.col("velocity_direction"))
            )
            .otherwise(pl.col("velocity_direction"))
            .alias("velocity_direction_corrected")
        )

        # Detect transitions from downward to upward
        df = df.with_columns(
            (
                (pl.col("velocity_direction_corrected") == "upward")
                & (pl.col("velocity_direction_corrected").shift(1) == "downward")
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


if __name__ == "__main__":
    main()
