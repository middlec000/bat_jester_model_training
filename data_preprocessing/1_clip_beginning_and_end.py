#!/usr/bin/env python3
"""
Script to trim videos based on voice commands "start" and "stop" detected by Whisper.

For each MP4 video in input_dir:
1. Use Whisper to transcribe the audio and detect timestamps
2. Find when "start" and "stop" are said
3. If both words are detected, trim the video (start - 3s to stop + 3s)
4. Save the trimmed video to output_dir
"""

import whisper
import os
from pathlib import Path
from moviepy import VideoFileClip
import tempfile
import subprocess


def find_word_timestamp(segments, target_word, min_probability=0.0, min_timestamp=0.0):
    """
    Find the timestamp of a target word in Whisper segments.

    Args:
        segments: List of segment dictionaries from Whisper
        target_word: Word to search for (case-insensitive)
        min_probability: Minimum probability threshold for word detection (0.0 to 1.0)
        min_timestamp: Minimum timestamp threshold (ignore words before this time)

    Returns:
        Tuple of (timestamp, probability) when the word starts, or (None, None) if not found
    """
    target_word_lower = target_word.lower()
    best_match = None
    best_probability = 0.0

    for segment in segments:
        # Check if segment has word-level timestamps
        if "words" in segment:
            for word_info in segment["words"]:
                # Strip punctuation and whitespace for comparison
                word = word_info["word"].strip().lower()
                word_clean = word.strip(".,!?;:")
                probability = word_info.get("probability", 1.0)
                timestamp = word_info["start"]

                # Check if this word matches and meets the probability and timestamp thresholds
                if (
                    target_word_lower == word_clean
                    and probability >= min_probability
                    and timestamp >= min_timestamp
                ):
                    if probability > best_probability:
                        best_match = timestamp
                        best_probability = probability
        else:
            # Fallback to segment-level text search (no probability available)
            text = segment["text"].lower()
            if target_word_lower in text:
                return segment[
                    "start"
                ], 1.0  # Assume high probability for segment-level match

    if best_match is not None:
        return best_match, best_probability

    return None, None


def process_video(video_path, output_dir, model, min_probability=0.0, volume_boost=1.0):
    """
    Process a single video: detect start/stop words and trim.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save trimmed video
        model: Loaded Whisper model
        min_probability: Minimum probability threshold for word detection (0.0 to 1.0)
        volume_boost: Audio volume multiplier (e.g., 2.0 = double volume, 5.0 = 5x volume)

    Returns:
        True if video was successfully processed, False otherwise
    """
    video_name = os.path.basename(video_path)
    print(f"\nProcessing: {video_name}")

    # If volume boost is needed, create a temporary file with amplified audio
    temp_file = None
    transcribe_path = video_path

    if volume_boost > 1.0:
        print(f"  Boosting audio volume by {volume_boost}x...")
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Use ffmpeg to amplify audio while keeping video unchanged
        # volume filter multiplies the audio amplitude
        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-af",
            f"volume={volume_boost}",
            "-c:v",
            "copy",  # Copy video stream without re-encoding
            "-y",  # Overwrite output file
            temp_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("  Warning: Failed to boost audio. Using original video.")
                print(f"  Error: {result.stderr}")
            else:
                transcribe_path = temp_path
        except Exception as e:
            print(f"  Warning: Failed to boost audio: {e}. Using original video.")

    # Transcribe audio with Whisper (enable word-level timestamps)
    print("  Transcribing audio...")
    result = model.transcribe(str(transcribe_path), word_timestamps=True, fp16=False)
    segments = result["segments"]

    # Debug: Show full transcript
    print(f"  DEBUG - Full transcript: '{result['text']}'")
    print()

    # TO DO: Remove this debug block later
    # Debug: Print all transcribed words with timestamps
    print("  DEBUG - All transcribed words:")
    for segment in segments:
        if "words" in segment:
            for word_info in segment["words"]:
                word = word_info["word"].strip()
                timestamp = word_info["start"]
                probability = word_info.get("probability", 1.0)
                print(f"    {timestamp:.2f}s: '{word}' (prob: {probability:.3f})")
        else:
            print(f"    Segment text: {segment['text']}")
    print()
    # End debug block

    # Find "start" and "stop" timestamps
    # Ignore "start" commands within 0.25 seconds of video beginning
    start_time, start_prob = find_word_timestamp(
        segments, "start", min_probability, min_timestamp=0.25
    )
    stop_time, stop_prob = find_word_timestamp(segments, "stop", min_probability)

    if start_time is None or stop_time is None:
        print("  Skipping: Could not detect both 'start' and 'stop'")
        print(f"    start: {start_time} (prob: {start_prob if start_prob else 'N/A'})")
        print(f"    stop: {stop_time} (prob: {stop_prob if stop_prob else 'N/A'})")
        print(f"    (min_probability threshold: {min_probability})")
        print("  Tip: Check if you actually said both words clearly in the video.")
        return False

    print(f"  Detected 'start' at {start_time:.2f}s (prob: {start_prob:.3f})")
    print(f"  Detected 'stop' at {stop_time:.2f}s (prob: {stop_prob:.3f})")

    # Calculate trim points (with 3-second buffer)
    trim_start = start_time
    trim_end = stop_time - 2

    print(f"  Trimming video from {trim_start:.2f}s to {trim_end:.2f}s")

    # Get video duration to validate trim range
    video = VideoFileClip(str(video_path))
    video_duration = video.duration
    video.close()

    # Ensure trim_end doesn't exceed video duration
    trim_end = min(trim_end, video_duration)

    if trim_start >= trim_end:
        print(f"  Skipping: Invalid trim range ({trim_start:.2f}s to {trim_end:.2f}s)")
        return False

    # Use ffmpeg with stream copy for fast clipping (no re-encoding)
    output_path = os.path.join(output_dir, video_name)
    print(f"  Saving to: {output_path}")

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
        "-y",  # Overwrite output
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error clipping video: {result.stderr}")
            return False
    except Exception as e:
        print(f"  Error running ffmpeg: {e}")
        return False

    # Clean up temporary boosted audio file
    if temp_file is not None:
        try:
            os.unlink(temp_file.name)
        except Exception:
            pass

    print(f"  ✓ Successfully processed {video_name}")
    return True


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    # data symlink is in the parent directory
    data_dir = (script_dir.parent / "data").resolve()

    input_dir = data_dir / "0_raw_videos"
    output_dir = data_dir / "1_clipped_videos"
    model_name = "small.en"

    # MANUAL THRESHOLD: Set minimum probability for detecting "start" and "stop" words
    # Range: 0.0 (accept all) to 1.0 (only perfect confidence)
    # Lower values (e.g., 0.3-0.5) will catch more words but may include false positives
    MIN_PROBABILITY = 0.0  # Adjust this value as needed

    # AUDIO VOLUME BOOST: Multiply audio volume to help Whisper detect quiet speech
    # Range: 1.0 (no change) to 10.0 (very loud)
    # Recommended: Start with 2.0-5.0 for quiet videos
    VOLUME_BOOST = 5.0  # Adjust this value as needed

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Min probability threshold: {MIN_PROBABILITY}")
    print(f"  Audio volume boost: {VOLUME_BOOST}x")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")

    # Load Whisper model
    print("\nLoading Whisper model...")
    model = whisper.load_model(model_name, device="cpu")

    # Find all MP4 videos
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.MP4"))
    # video_files = video_files[1:2]  # To DO: remove this line to process all videos
    # video_files = [input_dir / "PXL_20251215_232210350.mp4"]

    if not video_files:
        print(f"No MP4 files found in {input_dir}")
        return

    print(f"\nFound {len(video_files)} video(s)")

    # Process each video
    successful = 0
    skipped = 0

    for video_path in video_files:
        try:
            if process_video(
                video_path, output_dir, model, MIN_PROBABILITY, VOLUME_BOOST
            ):
                successful += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  Error processing {video_path.name}: {e}")
            skipped += 1

    # Summary
    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"  Successfully processed: {successful}")
    print(f"  Skipped: {skipped}")
    print(f"  Total: {len(video_files)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
