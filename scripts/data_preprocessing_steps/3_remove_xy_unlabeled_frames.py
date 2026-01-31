"""
Script to remove frames from videos and corresponding parquet files where x and y ball positions are unlabeled.
Splits videos and parquet files into segments of continuous labeled frames, discarding segments with fewer than MIN_FRAMES frames.
Preserves audio tracks in the output videos.
"""

from pathlib import Path
import polars as pl
from typing import List
from fractions import Fraction
import sys
import av
import logging

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import logging_setup

VIDEO_DIR = Path("~/data/bat_jester_model_training/B_clipped_videos").expanduser()
BALL_XY_POSITIONS_DIR = Path(
    "~/data/bat_jester_model_training/C_ball_xy_positions"
).expanduser()
OUTPUT_DIR = Path(
    "~/data/bat_jester_model_training/D_completely_xy_labeled_clips"
).expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/3_logs").expanduser()

MIN_FRAMES = 30


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
    for col in ["x", "y"]:
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
                .alias(f"{col}_new")
            )

            # Replace original column and drop temporary columns
            df = (
                df.drop(col)
                .rename({f"{col}_new": col})
                .drop([f"{col}_prev", f"{col}_next"])
            )

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
    valid_mask = xy_labels_df["x"].is_not_null() & xy_labels_df["y"].is_not_null()

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
    Copy a video file preserving both video and audio streams using PyAV.

    Args:
        input_path: Path to input video
        output_path: Path to output video

    Returns:
        True if successful, False otherwise
    """
    try:
        input_container = av.open(str(input_path))
        output_container = av.open(str(output_path), mode="w")

        # Copy video stream
        if input_container.streams.video:
            input_video = input_container.streams.video[0]
            output_video = output_container.add_stream(
                "libx264", rate=input_video.average_rate
            )
            output_video.width = input_video.width
            output_video.height = input_video.height
            output_video.pix_fmt = "yuv420p"
            output_video.options = {"crf": "23", "preset": "fast"}
        else:
            logger.warning(f"No video stream found in {input_path}")
            input_container.close()
            output_container.close()
            return False

        # Copy audio stream
        output_audio = None
        if input_container.streams.audio:
            input_audio = input_container.streams.audio[0]
            # Copy audio stream using the same codec as input
            output_audio = output_container.add_stream(
                input_audio.codec_context.name, rate=input_audio.sample_rate
            )

        # Process and write all packets
        for packet in input_container.demux():
            if packet.stream.type == "video":
                for frame in packet.decode():
                    for encoded_packet in output_video.encode(frame):
                        output_container.mux(encoded_packet)
            elif packet.stream.type == "audio" and output_audio:
                for frame in packet.decode():
                    for encoded_packet in output_audio.encode(frame):
                        output_container.mux(encoded_packet)

        # Flush remaining packets
        for packet in output_video.encode():
            output_container.mux(packet)
        if output_audio:
            for packet in output_audio.encode():
                output_container.mux(packet)

        input_container.close()
        output_container.close()
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
    Split a video into segments based on continuous non-null regions.

    For each segment, reloads the video and extracts only the frames in that range.

    Args:
        video_path: Path to the input video
        nonnull_segments: List of (start_frame_num, end_frame_num) tuples for valid regions
        output_prefix: Prefix for output files
        annotated: If True, uses {stem}_seg{N}_annotated.mp4 naming, else {stem}_seg{N}.mp4

    Returns the number of segments saved.
    """
    segments_saved = 0

    # Process each segment independently
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
            # Open input container fresh for this segment
            input_container = av.open(str(video_path))
            input_video = input_container.streams.video[0]
            input_audio = (
                input_container.streams.audio[0]
                if input_container.streams.audio
                else None
            )

            # Create output container
            output_container = av.open(str(output_path), mode="w")
            output_video = output_container.add_stream(
                "libx264", rate=input_video.average_rate
            )
            output_video.width = input_video.width
            output_video.height = input_video.height
            output_video.pix_fmt = "yuv420p"
            output_video.options = {"crf": "23", "preset": "fast"}

            output_audio = None
            if input_audio:
                output_audio = output_container.add_stream(
                    input_audio.codec_context.name, rate=input_audio.sample_rate
                )
                # Ensure audio time base is set so we can compute pts offsets
                if output_audio.time_base is None:
                    output_audio.time_base = input_audio.time_base
                if output_audio.time_base is None:
                    output_audio.time_base = Fraction(1, int(input_audio.sample_rate))

            # Compute fps and segment times
            fps = None
            if input_video.average_rate is not None:
                try:
                    fps = float(input_video.average_rate)
                except Exception:
                    fps = None
            elif getattr(input_video, "base_rate", None) is not None:
                try:
                    fps = float(input_video.base_rate)
                except Exception:
                    fps = None

            if output_video.time_base is None and fps:
                output_video.time_base = Fraction(1, int(round(fps)))
            if output_video.time_base is None:
                output_video.time_base = input_video.time_base

            if fps:
                start_time = start_frame / fps
                end_time = (end_frame + 1) / fps
            else:
                start_time = None
                end_time = None

            # Iterate packets and write only frames/samples that overlap the segment
            video_frame_index = 0
            output_frame_num = 0
            audio_sample_index = 0
            frames_written = 0
            done_video = False
            done_audio = False

            demux_streams = (
                (input_video, input_audio) if input_audio else (input_video,)
            )

            for packet in input_container.demux(demux_streams):
                if packet.stream.type == "video":
                    for frame in packet.decode():
                        # Use decoded order index as a fallback if frame timestamps are missing
                        frame_index = video_frame_index

                        if frame.pts is not None and frame.time_base is not None:
                            try:
                                frame_time = float(frame.pts * frame.time_base)
                            except Exception:
                                frame_time = frame_index / fps if fps else None
                        else:
                            frame_time = frame_index / fps if fps else None

                        # Skip frames before segment
                        if frame_time is not None and start_time is not None:
                            if frame_time < start_time:
                                video_frame_index += 1
                                continue
                            if frame_time >= end_time:
                                done_video = True
                                break

                            rel_pts = int(
                                round(
                                    (frame_time - start_time)
                                    / float(output_video.time_base)
                                )
                            )
                        else:
                            # Fallback to frame index counts
                            if frame_index < start_frame:
                                video_frame_index += 1
                                continue
                            if frame_index > end_frame:
                                done_video = True
                                break
                            rel_pts = output_frame_num

                        # Assign proper pts/time_base and encode
                        frame.pts = max(rel_pts, 0)
                        frame.time_base = output_video.time_base

                        for encoded_packet in output_video.encode(frame):
                            output_container.mux(encoded_packet)

                        output_frame_num += 1
                        frames_written += 1
                        video_frame_index += 1

                    if done_video and (not output_audio or done_audio):
                        break

                elif packet.stream.type == "audio" and output_audio:
                    for frame in packet.decode():
                        # compute frame start time
                        if frame.pts is not None and frame.time_base is not None:
                            try:
                                frame_start_time = float(frame.pts * frame.time_base)
                            except Exception:
                                frame_start_time = (
                                    audio_sample_index / input_audio.sample_rate
                                )
                        else:
                            frame_start_time = (
                                audio_sample_index / input_audio.sample_rate
                            )

                        frame_duration = frame.samples / input_audio.sample_rate
                        frame_end_time = frame_start_time + frame_duration

                        # Skip if entirely before the segment
                        if start_time is not None and frame_end_time <= start_time:
                            audio_sample_index += frame.samples
                            continue

                        # Stop if we've passed the segment
                        if end_time is not None and frame_start_time >= end_time:
                            done_audio = True
                            break

                        # Compute relative pts for this audio frame
                        if start_time is not None:
                            rel_audio_pts = int(
                                round(
                                    (max(frame_start_time, start_time) - start_time)
                                    / float(output_audio.time_base)
                                )
                            )
                        else:
                            rel_audio_pts = (
                                frame.pts
                                if frame.pts is not None
                                else audio_sample_index
                            )

                        frame.pts = max(rel_audio_pts, 0)
                        frame.time_base = output_audio.time_base

                        for encoded_packet in output_audio.encode(frame):
                            output_container.mux(encoded_packet)

                        audio_sample_index += frame.samples

                    if done_audio and done_video:
                        break

            # Flush remaining packets
            for packet in output_video.encode():
                output_container.mux(packet)
            if output_audio:
                for packet in output_audio.encode():
                    output_container.mux(packet)

            output_container.close()
            input_container.close()

            logger.info(
                f"Saved segment {segment_num}: {output_path} ({frames_written} frames)"
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
    run_all = "--run-all" in sys.argv
    print(f"Remove xy unlabeled frames (--run-all: {run_all})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging_setup.get_processing_logger(LOGGING_DIR)

    logger.info(f"Starting video processing from {VIDEO_DIR}")
    logger.info(f"Run all videos: {'Yes' if run_all else 'No (unprocessed only)'}")

    mp4_files = sorted(VIDEO_DIR.glob("*.mp4"))

    if run_all:
        unprocessed_videos = mp4_files
    else:
        processed_stems = [
            file.stem.split("_seg")[0] for file in OUTPUT_DIR.glob("*.parquet")
        ]
        unprocessed_videos = [f for f in mp4_files if f.stem not in processed_stems]

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
uv run python scripts/data_preprocessing_steps/3_remove_xy_unlabeled_frames.py --run-all
"""
