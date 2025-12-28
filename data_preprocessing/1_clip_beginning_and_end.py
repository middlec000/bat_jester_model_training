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
from moviepy.video.io.VideoFileClip import VideoFileClip


def find_word_timestamp(segments, target_word, min_probability=0.0):
    """
    Find the timestamp of a target word in Whisper segments.

    Args:
        segments: List of segment dictionaries from Whisper
        target_word: Word to search for (case-insensitive)
        min_probability: Minimum probability threshold for word detection (0.0 to 1.0)

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

                # Check if this word matches and meets the probability threshold
                if target_word_lower == word_clean and probability >= min_probability:
                    if probability > best_probability:
                        best_match = word_info["start"]
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


def process_video(video_path, output_dir, model, min_probability=0.0):
    """
    Process a single video: detect start/stop words and trim.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save trimmed video
        model: Loaded Whisper model
        min_probability: Minimum probability threshold for word detection (0.0 to 1.0)

    Returns:
        True if video was successfully processed, False otherwise
    """
    video_name = os.path.basename(video_path)
    print(f"\nProcessing: {video_name}")

    # Transcribe audio with Whisper (enable word-level timestamps)
    print("  Transcribing audio...")
    result = model.transcribe(str(video_path), word_timestamps=True, fp16=False)
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
    start_time, start_prob = find_word_timestamp(segments, "start", min_probability)
    stop_time, stop_prob = find_word_timestamp(segments, "stop", min_probability)

    if start_time is None or stop_time is None:
        print(f"  Skipping: Could not detect both 'start' and 'stop'")
        print(f"    start: {start_time} (prob: {start_prob if start_prob else 'N/A'})")
        print(f"    stop: {stop_time} (prob: {stop_prob if stop_prob else 'N/A'})")
        print(f"    (min_probability threshold: {min_probability})")
        print(f"  Tip: Check if you actually said both words clearly in the video.")
        return False

    print(f"  Detected 'start' at {start_time:.2f}s (prob: {start_prob:.3f})")
    print(f"  Detected 'stop' at {stop_time:.2f}s (prob: {stop_prob:.3f})")

    # Calculate trim points (with 3-second buffer)
    trim_start = start_time
    trim_end = stop_time - 2

    print(f"  Trimming video from {trim_start:.2f}s to {trim_end:.2f}s")

    # Load and trim video
    video = VideoFileClip(str(video_path))

    # Ensure trim_end doesn't exceed video duration
    trim_end = min(trim_end, video.duration)

    if trim_start >= trim_end:
        print(f"  Skipping: Invalid trim range ({trim_start:.2f}s to {trim_end:.2f}s)")
        video.close()
        return False

    trimmed_video = video.subclip(trim_start, trim_end)

    # Save trimmed video
    output_path = os.path.join(output_dir, video_name)
    print(f"  Saving to: {output_path}")
    trimmed_video.write_videofile(
        output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None
    )

    # Clean up
    trimmed_video.close()
    video.close()

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

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Min probability threshold: {MIN_PROBABILITY}")
    print(f"  Input directory: {input_dir}")
    print(f"  Output directory: {output_dir}")

    # Load Whisper model
    print(f"\nLoading Whisper model...")
    model = whisper.load_model(model_name, device="cpu")

    # Find all MP4 videos
    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.MP4"))
    video_files = video_files[1:2]  # To DO: remove this line to process all videos
    video_files = [input_dir / "PXL_20251210_230116248.mp4"]

    if not video_files:
        print(f"No MP4 files found in {input_dir}")
        return

    print(f"\nFound {len(video_files)} video(s)")

    # Process each video
    successful = 0
    skipped = 0

    for video_path in video_files:
        try:
            if process_video(video_path, output_dir, model, MIN_PROBABILITY):
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
