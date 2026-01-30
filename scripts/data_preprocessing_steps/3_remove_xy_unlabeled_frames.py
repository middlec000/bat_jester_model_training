from pathlib import Path
import cv2
import pandas as pd
from typing import List
import sys
import subprocess

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import logging_setup

VIDEO_DIR = Path("data/B_clipped_videos")
BALL_XY_POSITIONS_DIR = Path("data/C_ball_xy_positions")
OUTPUT_DIR = Path("data/D_completely_xy_labeled_clips")
LOGGING_DIR = Path("data/3_logs")

MIN_FRAMES = 30

logger = logging_setup.get_processing_logger(LOGGING_DIR)


def collect_list_processed_videos(output_dir: Path) -> List[str]:
    """Collect list of processed video stems in the output directory.

    Returns the set of video stems that have already been processed.
    Maps names from file names like "PXL_20251124_223727362.TS_seg5.parquet" to "PXL_20251124_223727362.TS".

    """
    processed_videos = set()
    for file in output_dir.glob("*.parquet"):
        stem = file.stem.split("_seg")[0]
        processed_videos.add(stem)
    return list(processed_videos)


def find_null_frames(xy_labels_df: pd.DataFrame) -> List[int]:
    """Find frames where x or y is null in the xy labels."""
    null_mask = xy_labels_df["x"].isna() | xy_labels_df["y"].isna()
    null_frames = xy_labels_df[null_mask].index.tolist()
    return null_frames


def split_video_at_frames(
    video_path: Path,
    null_frames: List[int],
    output_prefix: Path,
    annotated: bool = False,
) -> int:
    """
    Split a video at specified frames and save segments, preserving audio.

    Uses ffmpeg to cut the video at the specified frame times while preserving
    both video and audio streams.

    Args:
        video_path: Path to the input video
        null_frames: List of frame indices where to split
        output_prefix: Prefix for output files
        annotated: If True, uses {stem}_seg{N}_annotated.mp4 naming, else {stem}_seg{N}.mp4

    Returns the number of segments saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        logger.warning(f"Could not determine FPS for {video_path.name}, using 30 fps")
        fps = 30

    segments_saved = 0
    segment_num = 1

    # Convert frame numbers to timestamps in seconds
    split_times = sorted([frame / fps for frame in null_frames]) + [float("inf")]
    split_times_with_frames = sorted(null_frames) + [float("inf")]

    segment_start = 0
    segment_start_time = 0.0

    for split_idx, (split_frame, split_time) in enumerate(
        zip(split_times_with_frames, split_times)
    ):
        segment_end = int(split_frame) if split_frame != float("inf") else total_frames
        segment_end_time = split_time
        segment_duration = segment_end_time - segment_start_time

        # Convert to frame count
        segment_frame_count = segment_end - segment_start

        if segment_frame_count >= MIN_FRAMES:
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

            # Use ffmpeg to cut the video while preserving audio
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-ss",
                str(segment_start_time),
                "-to",
                str(segment_end_time),
                "-c:v",
                "copy",  # Copy video codec without re-encoding
                "-c:a",
                "aac",  # Re-encode audio to AAC (compatible with mp4)
                "-y",
                "-loglevel",
                "panic",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(
                    f"ffmpeg failed for segment {segment_num}: {result.stderr}"
                )
            else:
                logger.info(
                    f"Saved segment {segment_num}: {output_path} ({segment_frame_count} frames, {segment_duration:.2f}s)"
                )
                segments_saved += 1
        elif segment_frame_count > 0:
            logger.debug(
                f"Discarded segment {segment_num}: {segment_frame_count} frames (< {MIN_FRAMES})"
            )

        segment_start = segment_end
        segment_start_time = segment_end_time
        segment_num += 1

    return segments_saved


def split_parquet_at_frames(
    df: pd.DataFrame, null_frames: List[int], output_prefix: Path, video_stem: str
) -> int:
    """
    Split parquet data at specified frames and save segments.

    Returns the number of segments saved.
    """
    segments_saved = 0
    segment_num = 1
    start_frame = 0

    # Add sentinel values for easier processing
    split_points = sorted(null_frames) + [float("inf")]

    for split_idx, split_point in enumerate(split_points):
        end_frame = int(split_point) if split_point != float("inf") else len(df)
        segment_df = df.iloc[start_frame:end_frame].copy()

        if len(segment_df) >= MIN_FRAMES:
            output_path = (
                output_prefix.parent / f"{video_stem}_seg{segment_num}.parquet"
            )
            segment_df.to_parquet(output_path)
            logger.info(
                f"Saved parquet segment {segment_num}: {output_path} ({len(segment_df)} frames)"
            )
            segments_saved += 1
        elif len(segment_df) > 0:
            logger.debug(
                f"Discarded parquet segment {segment_num}: {len(segment_df)} frames (< {MIN_FRAMES})"
            )

        start_frame = end_frame
        segment_num += 1

    return segments_saved


def process_videos():
    """Process all videos in VIDEO_DIR."""
    logger.info(f"Starting video processing from {VIDEO_DIR}")

    processed_videos = collect_list_processed_videos(OUTPUT_DIR)
    mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))
    unprocessed_videos = [f for f in mp4_files if f.stem not in processed_videos]
    # unprocessed_videos = mp4_files
    logger.info(
        f"Found {len(mp4_files)} .mp4 files and {len(unprocessed_videos)} unprocessed videos"
    )

    total_segments = 0
    processed_count = 0

    for video_path in unprocessed_videos:
        video_name = video_path.stem

        # Find corresponding parquet file
        parquet_path = BALL_XY_POSITIONS_DIR / f"{video_name}.parquet"

        # Find corresponding annotated video
        annotated_video_path = BALL_XY_POSITIONS_DIR / f"{video_name}_annotated.mp4"

        if not parquet_path.exists():
            logger.warning(f"No parquet file found for {video_name}, skipping")
            continue

        if not annotated_video_path.exists():
            logger.warning(f"No annotated video found for {video_name}, skipping")
            continue

        try:
            # Load xy labels
            xy_labels_df = pd.read_parquet(parquet_path)

            # Find null frames
            null_frames = find_null_frames(xy_labels_df)

            if len(null_frames) == 0:
                logger.info(f"No null values in {video_name}, copying as-is")
                output_path = OUTPUT_DIR / video_path.name
                annotated_output_path = OUTPUT_DIR / f"{video_name}_annotated.mp4"

                # Use ffmpeg to copy videos with audio preserved
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(video_path),
                    "-c:v",
                    "copy",  # Copy video codec without re-encoding
                    "-c:a",
                    "aac",  # Re-encode audio to AAC (compatible with mp4)
                    "-y",
                    "-loglevel",
                    "panic",
                    str(output_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(
                        f"ffmpeg failed copying {video_path.name}: {result.stderr}"
                    )
                else:
                    logger.info(f"Copied {video_name} to {output_path}")

                # Copy the annotated video with audio preserved
                cmd = [
                    "ffmpeg",
                    "-i",
                    str(annotated_video_path),
                    "-c:v",
                    "copy",  # Copy video codec without re-encoding
                    "-c:a",
                    "aac",  # Re-encode audio to AAC (compatible with mp4)
                    "-y",
                    "-loglevel",
                    "panic",
                    str(annotated_output_path),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(
                        f"ffmpeg failed copying annotated video: {result.stderr}"
                    )
                else:
                    logger.info(
                        f"Copied annotated {video_name} to {annotated_output_path}"
                    )

                # Copy parquet file as-is
                output_parquet_path = OUTPUT_DIR / f"{video_name}.parquet"
                xy_labels_df.to_parquet(output_parquet_path)
                logger.info(f"Copied parquet for {video_name} to {output_parquet_path}")

                total_segments += 1
            else:
                logger.info(
                    f"Found {len(null_frames)} null frames in {video_name}: {null_frames}"
                )

                # Split original video at null frames
                output_prefix = OUTPUT_DIR / f"{video_name}.mp4"
                segments_saved = split_video_at_frames(
                    video_path, null_frames, output_prefix
                )

                # Split annotated video at null frames
                annotated_output_prefix = OUTPUT_DIR / f"{video_name}_annotated.mp4"
                split_video_at_frames(
                    annotated_video_path,
                    null_frames,
                    annotated_output_prefix,
                    annotated=True,
                )

                # Split parquet file at null frames
                split_parquet_at_frames(
                    xy_labels_df, null_frames, output_prefix, video_name
                )

                total_segments += segments_saved

            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing {video_name}: {e}")

    logger.info(
        f"Processing complete. Processed {processed_count}/{len(mp4_files)} videos, saved {total_segments} segments"
    )


if __name__ == "__main__":
    process_videos()
