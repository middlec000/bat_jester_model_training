#!/usr/bin/env python3
"""
Script to trim videos based on voice commands "start" and "stop" detected by Whisper.

For each MP4 video in input_dir:
1. Use Whisper to transcribe the audio and detect timestamps
2. Find when "start" and "stop" are said
3. If both words are detected, trim the video (from start + START_BUFFER_SECONDS to stop - STOP_BUFFER_SECONDS)
4. Save the trimmed video to output_dir

Whisper model: https://github.com/openai/whisper/blob/main/model-card.md
"""

import whisper
import os
from pathlib import Path
from moviepy import VideoFileClip
import tempfile
import subprocess
from time import time
import sys

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import logging_setup

INPUT_DIR = Path("data/A_raw_videos")
OUTPUT_DIR = Path("~/data/bat_jester_model_training/B_clipped_videos").expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/1_logs").expanduser()

MODEL_NAME = "medium.en"
START_BUFFER_SECONDS = 2.0
STOP_BUFFER_SECONDS = 3.0
MIN_PROBABILITY = 0.05
VOLUME_BOOST = 2.0


def find_all_word_timestamps(
    segments, target_word, min_probability=0.0, min_timestamp=0.0
) -> list[tuple[float, float]]:
    """Return a sorted list of (timestamp, probability) for all occurrences of target_word."""
    target_word_lower = target_word.lower()
    matches: list[tuple[float, float]] = []

    for segment in segments:
        if "words" in segment:
            for word_info in segment["words"]:
                word = word_info["word"].strip().lower()
                word_clean = word.strip(".,!?;:")
                probability = word_info.get("probability", 1.0)
                timestamp = word_info["start"]

                if (
                    target_word_lower == word_clean
                    and probability >= min_probability
                    and timestamp >= min_timestamp
                ):
                    matches.append((timestamp, probability))
        else:
            # fallback to segment-level match
            text = segment.get("text", "").lower()
            if target_word_lower in text:
                matches.append((segment["start"], 1.0))

    matches.sort(key=lambda x: x[0])
    return matches


def preprocess_audio(
    video_path: str,
    output_path: str,
    volume_boost: float = 1.0,
    denoise: bool = True,
    normalize: bool = True,
) -> bool:
    """
    Preprocess audio with multiple enhancement filters.

    Args:
        video_path: Input video file
        output_path: Output video file with enhanced audio
        volume_boost: Volume multiplier
        denoise: Apply noise reduction
        normalize: Apply audio normalization

    Returns:
        True if successful, False otherwise
    """
    filters = []

    # High-pass filter to remove low-frequency noise (rumble)
    filters.append("highpass=f=80")

    # Noise reduction (FFmpeg's afftdn filter)
    if denoise:
        filters.append("afftdn=nf=-25")

    # Dynamic range compression to make quiet speech louder
    filters.append("acompressor=threshold=0.089:ratio=9:attack=200:release=1000")

    # Volume boost
    if volume_boost > 1.0:
        filters.append(f"volume={volume_boost}")

    # Normalization to ensure consistent levels
    if normalize:
        filters.append("loudnorm")

    audio_filter = ",".join(filters)

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-af",
        audio_filter,
        "-c:v",
        "copy",  # Copy video without re-encoding
        "-movflags",
        "+faststart",  # Move moov atom to beginning for better compatibility
        "-y",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except Exception:
        return False


def process_video(
    video_path,
    output_dir,
    model,
    processing_logger,
    logging_dir: Path,
    min_probability=0.0,
    volume_boost=1.0,
    enable_preprocessing=True,  # New parameter
) -> bool:
    """
    Process a single video: detect start/stop words and trim.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save trimmed video
        model: Loaded Whisper model
        processing_logger: Logger for high-level processing messages
        logging_dir: Directory where log files should be written
        min_probability: Minimum probability threshold for word detection (0.0 to 1.0)
        volume_boost: Audio volume multiplier (e.g., 2.0 = double volume, 5.0 = 5x volume)

    Returns:
        True if video was successfully processed, False otherwise
    """
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]
    video_logger = logging_setup.get_file_logger(video_stem, logging_dir)
    processing_logger.info("Processing %s", video_name)
    video_logger.info("Processing %s", video_name)

    temp_file = None
    transcribe_path = video_path

    if enable_preprocessing:
        video_logger.info("Applying audio preprocessing (denoise, normalize, compress)")
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        if not preprocess_audio(
            str(video_path),
            temp_path,
            volume_boost=volume_boost,
            denoise=True,
            normalize=True,
        ):
            video_logger.warning("Audio preprocessing failed; using original video")
        else:
            transcribe_path = temp_path

    result = model.transcribe(str(transcribe_path), word_timestamps=True, fp16=False)
    segments = result["segments"]

    # Debug: Show full transcript
    video_logger.debug("Full transcript: '%s'", result["text"])
    video_logger.debug("")

    # TO DO: Remove this debug block later
    # Debug: Print all transcribed words with timestamps
    video_logger.debug("All transcribed words:")
    for segment in segments:
        if "words" in segment:
            for word_info in segment["words"]:
                word = word_info["word"].strip()
                timestamp = word_info["start"]
                probability = word_info.get("probability", 1.0)
                video_logger.debug(
                    "%0.2fs: '%s' (prob: %.3f)", timestamp, word, probability
                )
        else:
            video_logger.debug("Segment text: %s", segment["text"])
    video_logger.debug("")
    # End debug block

    # Find "start" and "stop" timestamps
    # Ignore "start" commands within 0.25 seconds of video beginning
    starts = find_all_word_timestamps(
        segments, "start", min_probability, min_timestamp=0.25
    )
    stops = find_all_word_timestamps(
        segments, "stop", min_probability, min_timestamp=0.25
    )

    if len(starts) == 0 and len(stops) == 0:
        video_logger.warning("Could not detect 'start' or 'stop'")
        video_logger.info("Tip: Speak both words clearly in the video.")
        processing_logger.warning(
            "Skipping %s because words could not be detected", video_name
        )
        return False
    # If no explicit 'start' was found, but exactly two 'stop's are detected and both meet the
    # probability threshold, treat the first 'stop' as the 'start'. This helps when users
    # accidentally say 'stop' twice but mean the first to mark the start.
    elif len(starts) == 0:
        if len(stops) == 2:
            start_time, start_prob = stops[0]
            stop_time, stop_prob = stops[1]
            video_logger.info(
                "No explicit 'start' found; treating first 'stop' at %0.2fs as 'start' (prob: %.3f)",
                start_time,
                start_prob,
            )
            video_logger.info("Interpreting first 'stop' as 'start' for %s", video_name)
        else:
            video_logger.warning("Could not detect 'start'")
            processing_logger.warning(
                "Skipping %s because 'start' could not be detected", video_name
            )
            return False
    # If no explicit 'stop' found, but exactly two 'start's are detected and both meet the
    # probability threshold, treat the second 'start' as the 'stop'. This helps when users
    # accidentally say 'start' twice but mean the second one to mark the end.
    elif len(stops) == 0:
        if len(starts) == 2:
            start_time, start_prob = starts[0]
            stop_time, stop_prob = starts[1]
            video_logger.info(
                "No explicit 'stop' found; treating second 'start' at %0.2fs as 'stop' (prob: %.3f)",
                stop_time,
                stop_prob,
            )
            video_logger.info(
                "Interpreting second 'start' as 'stop' for %s", video_name
            )
        else:
            video_logger.warning("Could not detect 'stop'")
            processing_logger.warning(
                "Skipping %s because 'stop' could not be detected", video_name
            )
            return False
    else:
        # Both 'start' and 'stop' detected; use the highest probability occurrences
        start_time, start_prob = max(starts, key=lambda x: x[1])
        stop_time, stop_prob = max(stops, key=lambda x: x[1])

    video_logger.info("Detected 'start' at %0.2fs (prob: %.3f)", start_time, start_prob)
    video_logger.info("Detected 'stop' at %0.2fs (prob: %.3f)", stop_time, stop_prob)

    # Calculate trim points using configured buffers
    # Start buffer: seconds after the 'start' word to begin the clip
    # Stop buffer: seconds before the 'stop' word to end the clip
    trim_start = start_time + START_BUFFER_SECONDS
    trim_end = stop_time - STOP_BUFFER_SECONDS

    # Ensure trim_start is non-negative
    trim_start = max(0.0, trim_start)

    video_logger.info("Trimming video from %0.2fs to %0.2fs", trim_start, trim_end)

    # Get video duration to validate trim range
    video = VideoFileClip(str(video_path))
    video_duration = video.duration
    video.close()

    # Ensure trim_end doesn't exceed video duration
    trim_end = min(trim_end, video_duration)

    if trim_start >= trim_end:
        video_logger.warning(
            "Skipping: Invalid trim range (%0.2fs to %0.2fs)",
            trim_start,
            trim_end,
        )
        return False

    # Use ffmpeg with stream copy for fast clipping (no re-encoding)
    output_path = os.path.join(output_dir, video_name)
    video_logger.info("Saving to: %s", output_path)

    duration = trim_end - trim_start
    cmd = [
        "ffmpeg",
        "-ss",
        str(trim_start),  # Start time
        "-i",
        str(video_path),  # Input file
        "-t",
        str(duration),  # Duration
        "-c",
        "copy",  # Copy streams without re-encoding
        "-movflags",
        "+faststart",  # Move moov atom to beginning
        "-y",  # Overwrite output
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            video_logger.error("Error clipping video: %s", result.stderr)
            processing_logger.error("Error clipping %s: %s", video_name, result.stderr)
            return False
    except Exception as e:
        video_logger.error("Error running ffmpeg: %s", e)
        processing_logger.error("Error running ffmpeg for %s: %s", video_name, e)
        return False

    # Clean up temporary boosted audio file
    if temp_file is not None:
        try:
            os.unlink(temp_file.name)
        except Exception:
            pass

    video_logger.info("✓ Successfully processed %s", video_name)
    processing_logger.info("Completed %s", video_name)
    return True


def main():
    start_time = time()

    # Parse command-line arguments
    run_all = "--run-all" in sys.argv

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    processing_logger = logging_setup.get_processing_logger(LOGGING_DIR)
    processing_logger.info("Configuration:")
    processing_logger.info("  Model: %s", MODEL_NAME)
    processing_logger.info("  Min probability threshold: %s", MIN_PROBABILITY)
    processing_logger.info("  Audio volume boost: %sx", VOLUME_BOOST)
    processing_logger.info("  Start buffer seconds: %s", START_BUFFER_SECONDS)
    processing_logger.info("  Stop buffer seconds: %s", STOP_BUFFER_SECONDS)
    processing_logger.info("  Input directory: %s", INPUT_DIR)
    processing_logger.info("  Output directory: %s", OUTPUT_DIR)
    processing_logger.info("  Logging directory: %s", LOGGING_DIR)
    processing_logger.info(
        "  Run all videos: %s", "Yes" if run_all else "No (unprocessed only)"
    )

    # Load Whisper model
    processing_logger.info("Loading Whisper model...")
    model = whisper.load_model(MODEL_NAME, device="cpu")

    # Filter out videos that have already been processed
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    if run_all:
        unprocessed_videos = sorted(INPUT_DIR.glob("*.mp4"))
    else:
        unprocessed_videos = logging_setup.get_unprocessed_files(
            input_dir=INPUT_DIR,
            input_filetype="mp4",
            output_dir=LOGGING_DIR,
            output_filetype="log",
        )

    if not unprocessed_videos:
        processing_logger.info("No videos found to process")
        return

    processing_logger.info("Found %s video(s) to process", len(unprocessed_videos))

    # Process each video
    successful = 0
    skipped = 0

    for video_path in unprocessed_videos:
        try:
            if process_video(
                video_path,
                OUTPUT_DIR,
                model,
                processing_logger,
                LOGGING_DIR,
                MIN_PROBABILITY,
                VOLUME_BOOST,
            ):
                successful += 1
            else:
                skipped += 1
        except Exception as e:
            processing_logger.error("Error processing %s: %s", video_path.name, e)
            skipped += 1

    # Summary
    processing_logger.info("%s", "=" * 60)
    processing_logger.info("Processing complete!")
    processing_logger.info("  Successfully processed: %s", successful)
    processing_logger.info("  Skipped: %s", skipped)
    processing_logger.info("  Total unprocessed videos: %s", len(unprocessed_videos))
    processing_logger.info("%s", "=" * 60)
    end_time = time()
    elapsed = end_time - start_time
    processing_logger.info("Total elapsed time: %0.2f seconds", elapsed)


if __name__ == "__main__":
    main()

"""
uv run python scripts/data_preprocessing_steps/1_clip_beginning_and_end.py --run-all
"""
