"""
Script to remove frames from videos and corresponding parquet files where x and y ball positions are unlabeled.
Splits videos and parquet files into segments of continuous labeled frames, discarding segments with fewer than MIN_FRAMES frames.
Preserves audio tracks in the output videos.
"""

from pathlib import Path
import polars as pl
from typing import List
import sys
import subprocess
import shutil
import logging
import argparse

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import utils

VIDEO_DIR = Path("~/data/bat_jester_model_training/B_clipped_videos").expanduser()
BALL_XY_POSITIONS_DIR = Path(
    "~/data/bat_jester_model_training/C_ball_xy_positions"
).expanduser()
OUTPUT_DIR = Path(
    "~/data/bat_jester_model_training/D_completely_xy_labeled_clips"
).expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/3_logs").expanduser()

MIN_FRAMES = 30
CONFIDENCE_THRESHOLD = 0.03


def threshold_xy_on_confidence(
    xy_labels_df: pl.DataFrame, confidence_threshold: float
) -> pl.DataFrame:
    """
    Set x and y to null if confidence is below the threshold.

    Args:
        xy_labels_df: DataFrame with Frame, x, y, and confidence columns
        confidence_threshold: Minimum confidence to keep x and y values
    Returns:
        DataFrame with x and y set to null where confidence < threshold
    """
    df = xy_labels_df.clone()

    df = df.with_columns(
        pl.when(pl.col("confidence") < confidence_threshold)
        .then(None)
        .otherwise(pl.col("x"))
        .alias("x_thresh"),
        pl.when(pl.col("confidence") < confidence_threshold)
        .then(None)
        .otherwise(pl.col("y"))
        .alias("y_thresh"),
    )

    return df


def impute_nulls_with_gap1(xy_labels_df: pl.DataFrame) -> pl.DataFrame:
    """
    Impute x and y values for frames with nulls if they have nonnull values
    in the frames immediately before and after (null gap of 1).

    Args:
        xy_labels_df: DataFrame with Frame, x, and y columns (Frame acts as index)

    Returns:
        DataFrame with imputed values
    """
    df = xy_labels_df.clone()

    # For each column that might have nulls (x and y)
    for col in ["x_thresh", "y_thresh"]:
        if col in df.columns:
            # Create previous frame values by shifting Frame up by 1
            prev_df = df.select(
                [(pl.col("Frame") + 1).alias("Frame"), pl.col(col).alias(f"{col}_prev")]
            )

            # Create next frame values by shifting Frame down by 1
            next_df = df.select(
                [(pl.col("Frame") - 1).alias("Frame"), pl.col(col).alias(f"{col}_next")]
            )

            # Join to get neighboring values
            df = df.join(prev_df, on="Frame", how="left")
            df = df.join(next_df, on="Frame", how="left")

            # Impute null values using linear interpolation
            df = df.with_columns(
                pl.when(
                    (pl.col(col).is_null())
                    & (pl.col(f"{col}_prev").is_not_null())
                    & (pl.col(f"{col}_next").is_not_null())
                )
                .then((pl.col(f"{col}_prev") + pl.col(f"{col}_next")) / 2)
                .otherwise(pl.col(col))
                .alias(f"{col}_imputed"),
            )

            # Replace original column and drop temporary columns
            df = df.drop([f"{col}_prev", f"{col}_next"])

    return df


def find_nonnull_segments(xy_labels_df: pl.DataFrame) -> List[tuple]:
    """
    Find continuous segments where both x and y are non-null.

    Returns a list of (start_frame_num, end_frame_num) tuples representing
    continuous segments of non-null x and y values.
    For example, if frames 1-5 are complete and 10-15 are complete, returns
    [(1, 5), (10, 15)].
    """
    # Create a boolean mask for rows where both x and y are non-null
    valid_mask = (
        xy_labels_df["x_thresh_imputed"].is_not_null()
        & xy_labels_df["y_thresh_imputed"].is_not_null()
    )

    # Add a helper column to identify segments
    df_with_valid = xy_labels_df.with_columns(valid_mask.alias("is_valid"))

    # Get frames that are valid
    valid_frames = sorted(
        df_with_valid.filter(pl.col("is_valid")).select("Frame")["Frame"].to_list()
    )

    if not valid_frames:
        return []

    # Find continuous segments
    segments = []
    segment_start = valid_frames[0]
    prev_frame = valid_frames[0]

    for frame in valid_frames[1:]:
        if frame != prev_frame + 1:
            # Gap detected, save previous segment
            segments.append((segment_start, prev_frame))
            segment_start = frame
        prev_frame = frame

    # Don't forget the last segment
    segments.append((segment_start, prev_frame))

    return segments


def copy_video_with_audio(
    input_path: Path, output_path: Path, logger: logging.Logger
) -> bool:
    """
    Copy a video file preserving both video and audio streams using ffmpeg subprocess.

    Args:
        input_path: Path to input video
        output_path: Path to output video

    Returns:
        True if successful, False otherwise
    """
    try:
        if not shutil.which("ffmpeg"):
            logger.error("ffmpeg not found in PATH")
            return False

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c",
            "copy",
            str(output_path),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error(f"ffmpeg copy failed for {input_path}: {proc.stderr.strip()}")
            return False

        return True

    except Exception as e:
        logger.error(f"Error copying video from {input_path} to {output_path}: {e}")
        return False


def split_video_at_frames(
    video_path: Path,
    nonnull_segments: List[tuple],
    output_prefix: Path,
    logger: logging.Logger,
    annotated: bool = False,
) -> int:
    """
    Split a video into segments based on continuous non-null regions using ffmpeg.

    If FPS is available via ffprobe, performs frame-accurate trimming using -ss (after -i) and -t.
    If FPS cannot be determined, falls back to using the frame-selection filter (select='between(n,...)')
    which is exact in frames but will drop audio.

    Returns the number of segments saved.
    """
    segments_saved = 0

    fps = utils.get_video_fps(video_path)
    if fps is None:
        logger.debug(f"ffprobe failed to probe FPS for {video_path}")
    has_audio = utils.video_has_audio(video_path)

    for segment_num, (start_frame, end_frame) in enumerate(nonnull_segments, 1):
        frame_count = end_frame - start_frame + 1

        if frame_count < MIN_FRAMES:
            logger.debug(
                f"Discarded segment {segment_num}: {frame_count} frames (< {MIN_FRAMES})"
            )
            continue

        if annotated:
            output_path = (
                output_prefix.parent
                / f"{output_prefix.stem.replace('_annotated', '')}_seg{segment_num}_annotated{output_prefix.suffix}"
            )
        else:
            output_path = (
                output_prefix.parent
                / f"{output_prefix.stem}_seg{segment_num}{output_prefix.suffix}"
            )

        try:
            if fps:
                start_time = start_frame / fps
                duration = frame_count / fps

                # Use -ss after -i for frame-accurate seeking
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-ss",
                    f"{start_time:.6f}",
                    "-t",
                    f"{duration:.6f}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                ]

                if has_audio:
                    cmd += ["-c:a", "aac", "-b:a", "128k"]
                else:
                    cmd += ["-an"]

                cmd += [str(output_path)]

            else:
                # Fallback: select frames by index (exact in frames) but audio will be dropped
                vf = f"select='between(n,{start_frame},{end_frame})'"
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    vf,
                    "-vsync",
                    "0",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "23",
                    "-an",
                    str(output_path),
                ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                logger.error(
                    f"ffmpeg failed to create segment {segment_num} for {video_path}: {proc.stderr.strip()}"
                )
                continue

            logger.info(
                f"Saved segment {segment_num}: {output_path} ({frame_count} frames)"
            )
            segments_saved += 1

        except Exception as e:
            logger.error(f"Error processing segment {segment_num}: {e}")

    return segments_saved


def split_parquet_at_frames(
    df: pl.DataFrame,
    nonnull_segments: List[tuple],
    output_prefix: Path,
    video_stem: str,
    logger: logging.Logger,
) -> int:
    """
    Split parquet data into segments with continuous non-null x,y values.
    Uses the segment boundaries from find_nonnull_segments().

    Returns the number of segments saved.
    """
    segments_saved = 0

    for segment_num, (start_frame, end_frame) in enumerate(nonnull_segments, 1):
        # Filter DataFrame to only include frames in this segment
        segment_df = df.filter(
            (pl.col("Frame") >= start_frame) & (pl.col("Frame") <= end_frame)
        )

        if len(segment_df) >= MIN_FRAMES:
            output_path = (
                output_prefix.parent / f"{video_stem}_seg{segment_num}.parquet"
            )
            segment_df.write_parquet(output_path)
            logger.info(
                f"Saved parquet segment {segment_num}: {output_path} ({len(segment_df)} frames)"
            )
            segments_saved += 1
        elif len(segment_df) > 0:
            logger.debug(
                f"Discarded parquet segment {segment_num}: {len(segment_df)} frames (< {MIN_FRAMES})"
            )

    return segments_saved


def main():
    parser = argparse.ArgumentParser(
        description="Remove frames with unlabeled xy positions"
    )
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
    print(
        f"Remove xy unlabeled frames (run_mode: {run_mode}, substrings: {substrings})"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    logger = utils.get_processing_logger(LOGGING_DIR)

    logger.info(f"Starting video processing from {VIDEO_DIR}")
    logger.info("Run mode: %s", run_mode)
    if substrings:
        logger.info("Substring filters: %s", substrings)

    mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))

    if run_mode == "all":
        unprocessed_videos = mp4_files
    elif run_mode == "new":
        processed_stems = [
            file.stem.split("_seg")[0] for file in OUTPUT_DIR.glob("*.parquet")
        ]
        unprocessed_videos = [f for f in mp4_files if f.stem not in processed_stems]
    else:
        unprocessed_videos = [
            f for f in mp4_files if any(s in f.name for s in substrings)
        ]
        logger.info(
            f"Applying substring filter {substrings}: {len(mp4_files)} -> {len(unprocessed_videos)} videos"
        )

    logger.info(
        f"Found {len(mp4_files)} .mp4 files and {len(unprocessed_videos)} unprocessed videos"
    )

    total_segments = 0
    processed_count = 0

    for video_path in unprocessed_videos:
        video_stem = video_path.stem

        # Find corresponding parquet file
        parquet_path = BALL_XY_POSITIONS_DIR / f"{video_stem}.parquet"

        # Find corresponding annotated video
        annotated_video_path = BALL_XY_POSITIONS_DIR / f"{video_stem}_annotated.mp4"

        if not parquet_path.exists():
            logger.warning(f"No parquet file found for {video_stem}, skipping")
            continue

        if not annotated_video_path.exists():
            logger.warning(f"No annotated video found for {video_stem}, skipping")
            continue

        try:
            # Load xy labels
            xy_labels_df = pl.read_parquet(parquet_path)

            # Threshold x,y on confidence
            xy_labels_df = threshold_xy_on_confidence(
                xy_labels_df, CONFIDENCE_THRESHOLD
            )

            # Impute nulls with gap 1 before finding non-null segments
            xy_labels_df = impute_nulls_with_gap1(xy_labels_df)

            # Find segments with continuous non-null x, y values
            nonnull_segments = find_nonnull_segments(xy_labels_df)

            if len(nonnull_segments) == 0:
                logger.info(f"No non-null segments in {video_stem}, skipping")
                continue

            if len(nonnull_segments) == 1:
                start_frame, end_frame = nonnull_segments[0]
                logger.info(
                    f"Single continuous segment in {video_stem}: frames {start_frame}-{end_frame}"
                )
                output_path = OUTPUT_DIR / video_path.name
                annotated_output_path = OUTPUT_DIR / f"{video_stem}_annotated.mp4"

                # Copy original video with audio preserved
                if copy_video_with_audio(
                    input_path=video_path, output_path=output_path, logger=logger
                ):
                    logger.info(f"Copied {video_stem} to {output_path}")

                # Copy the annotated video with audio preserved
                if copy_video_with_audio(
                    input_path=annotated_video_path,
                    output_path=annotated_output_path,
                    logger=logger,
                ):
                    logger.info(
                        f"Copied annotated {video_stem} to {annotated_output_path}"
                    )

                # Copy parquet file as-is
                output_parquet_path = OUTPUT_DIR / f"{video_stem}.parquet"
                xy_labels_df.write_parquet(output_parquet_path)
                logger.info(f"Copied parquet for {video_stem} to {output_parquet_path}")

                total_segments += 1
            else:
                logger.info(
                    f"Found {len(nonnull_segments)} non-null segments in {video_stem}"
                )

                # Split original video into non-null segments
                output_prefix = OUTPUT_DIR / f"{video_stem}.mp4"
                segments_saved = split_video_at_frames(
                    video_path, nonnull_segments, output_prefix, logger=logger
                )

                # Split annotated video into non-null segments
                annotated_output_prefix = OUTPUT_DIR / f"{video_stem}_annotated.mp4"
                split_video_at_frames(
                    annotated_video_path,
                    nonnull_segments,
                    annotated_output_prefix,
                    annotated=True,
                    logger=logger,
                )

                # Split parquet file into non-null segments
                split_parquet_at_frames(
                    xy_labels_df,
                    nonnull_segments,
                    output_prefix,
                    video_stem,
                    logger=logger,
                )

                total_segments += segments_saved

            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing {video_stem}: {e}")

    logger.info(
        f"Processing complete. Processed {processed_count}/{len(mp4_files)} videos, saved {total_segments} segments"
    )


if __name__ == "__main__":
    main()

"""
# Examples:
# uv run python scripts/data_preprocessing_steps/3_remove_xy_unlabeled_frames.py --run all
# uv run python scripts/data_preprocessing_steps/3_remove_xy_unlabeled_frames.py --run new
# uv run python scripts/data_preprocessing_steps/3_remove_xy_unlabeled_frames.py --run substring1 substring2
"""
