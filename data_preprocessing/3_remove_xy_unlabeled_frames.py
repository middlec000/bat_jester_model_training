from pathlib import Path
import cv2
import pandas as pd
from typing import List
import logging

VIDEO_DIR = Path("data/1_clipped_videos")
BALL_XY_POSITIONS_DIR = Path("data/2_ball_xy_positions")
OUTPUT_DIR = Path("data/3_completely_xy_labeled_clips")
LOGGING_DIR = Path("data/2_to_3_logs")

MIN_FRAMES = 30

# Setup logging
# LOGGING_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOGGING_DIR / "processing.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_null_frames(xy_labels_df: pd.DataFrame) -> List[int]:
    """Find frames where x or y is null in the xy labels."""
    null_mask = xy_labels_df["x"].isna() | xy_labels_df["y"].isna()
    null_frames = xy_labels_df[null_mask].index.tolist()
    return null_frames


def split_video_at_frames(
    video_path: Path, null_frames: List[int], output_prefix: Path
) -> int:
    """
    Split a video at specified frames and save segments.

    Returns the number of segments saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    segments_saved = 0
    frame_count = 0
    segment_frames = []
    segment_num = 1

    # Add sentinel values for easier processing
    split_points = sorted(null_frames) + [float("inf")]
    next_split_idx = 0
    next_split = split_points[next_split_idx]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we need to save the current segment and start a new one
        if frame_count >= next_split:
            # Save current segment if it has enough frames
            if len(segment_frames) >= MIN_FRAMES:
                output_path = (
                    output_prefix.parent
                    / f"{output_prefix.stem}_seg{segment_num}{output_prefix.suffix}"
                )
                out = cv2.VideoWriter(
                    str(output_path), fourcc, fps, (frame_width, frame_height)
                )
                for seg_frame in segment_frames:
                    out.write(seg_frame)
                out.release()
                logger.info(
                    f"Saved segment {segment_num}: {output_path} ({len(segment_frames)} frames)"
                )
                segments_saved += 1
            elif len(segment_frames) > 0:
                logger.debug(
                    f"Discarded segment {segment_num}: {len(segment_frames)} frames (< {MIN_FRAMES})"
                )

            # Move to next split point
            next_split_idx += 1
            next_split = split_points[next_split_idx]
            segment_frames = []
            segment_num += 1

        segment_frames.append(frame)
        frame_count += 1

    # Handle the last segment
    if len(segment_frames) >= MIN_FRAMES:
        output_path = (
            output_prefix.parent
            / f"{output_prefix.stem}_seg{segment_num}{output_prefix.suffix}"
        )
        out = cv2.VideoWriter(
            str(output_path), fourcc, fps, (frame_width, frame_height)
        )
        for seg_frame in segment_frames:
            out.write(seg_frame)
        out.release()
        logger.info(
            f"Saved segment {segment_num}: {output_path} ({len(segment_frames)} frames)"
        )
        segments_saved += 1
    elif len(segment_frames) > 0:
        logger.debug(
            f"Discarded segment {segment_num}: {len(segment_frames)} frames (< {MIN_FRAMES})"
        )

    cap.release()
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
        end_frame = int(split_point)
        segment_df = df.iloc[start_frame:end_frame].copy()

        if len(segment_df) >= MIN_FRAMES:
            output_path = (
                output_prefix.parent
                / f"{video_stem}_seg{segment_num}_xy_frame_labels.parquet"
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

    mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))
    logger.info(f"Found {len(mp4_files)} .mp4 files")

    total_segments = 0
    processed_count = 0

    for video_path in mp4_files:
        video_name = video_path.stem

        # Find corresponding parquet file
        parquet_path = BALL_XY_POSITIONS_DIR / f"{video_name}_xy_frame_labels.parquet"

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

                # Copy the original video without splitting
                cap = cv2.VideoCapture(str(video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(
                    str(output_path), fourcc, fps, (frame_width, frame_height)
                )
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                out.release()
                cap.release()
                logger.info(f"Copied {video_name} to {output_path}")

                # Copy the annotated video without splitting
                cap = cv2.VideoCapture(str(annotated_video_path))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                out = cv2.VideoWriter(
                    str(annotated_output_path), fourcc, fps, (frame_width, frame_height)
                )
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                out.release()
                cap.release()
                logger.info(f"Copied annotated {video_name} to {annotated_output_path}")

                # Copy parquet file as-is
                output_parquet_path = (
                    OUTPUT_DIR / f"{video_name}_xy_frame_labels.parquet"
                )
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
                    annotated_video_path, null_frames, annotated_output_prefix
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
