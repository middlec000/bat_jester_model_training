import sys
import logging
from pathlib import Path
from time import time
import pandas as pd
import cv2

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import logging_setup, video_labeler

INPUT_PATH = Path("data/B_clipped_videos")
OUTPUT_PATH = Path("data/C_ball_xy_positions")
LOGGING_DIR = Path("data/2_logs")


def label_xy_positions(
    input_video_file_path: Path,
    output_dir: Path,
    output_annotated: bool = True,
    confidence_threshold: float = 0.001,
    video_logger: logging.Logger = None,
):
    """
    Process a single video and save:
    - Annotated video with ball positions overlaid
    - Parquet file with frame-by-frame ball positions (Frame, x, y)
    """
    if video_logger:
        video_logger.info("Processing video: %s", input_video_file_path.name)

    # Initialize labeler with low confidence threshold for ball detection
    labeler = video_labeler.SoccerJuggleVideoLabeler(
        video_path=str(input_video_file_path), confidence_threshold=confidence_threshold
    )

    if output_annotated:
        # Process video and save annotated version
        annotated_output_path = output_dir / (
            input_video_file_path.stem + "_annotated.mp4"
        )
        labeler.process_video(visualize=False, output_path=str(annotated_output_path))
    else:
        # Just process without saving annotated video
        annotated_output_path = None
        labeler.process_video(visualize=False, output_path=None)

    # Get total number of frames in the video
    cap = cv2.VideoCapture(str(input_video_file_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Create a complete DataFrame with all frames
    # ball_positions format: [(frame_num, x, y, confidence), ...]
    all_frames = pd.DataFrame({"Frame": range(total_frames)})

    if labeler.ball_positions:
        detected_df = pd.DataFrame(
            labeler.ball_positions, columns=["Frame", "x", "y", "confidence"]
        )
        # Merge to get all frames, with nulls where ball wasn't detected
        df = all_frames.merge(detected_df[["Frame", "x", "y"]], on="Frame", how="left")
    else:
        # No detections - all x, y values will be null
        df = all_frames
        df["x"] = None
        df["y"] = None

    # Save as parquet
    parquet_output_path = output_dir / (input_video_file_path.stem + ".parquet")
    df[["Frame", "x", "y"]].to_parquet(parquet_output_path, index=False)

    # Calculate detection statistics
    detected_count = df[["x", "y"]].notna().all(axis=1).sum()
    total_count = len(df)

    if video_logger:
        video_logger.info("Summary:")
        video_logger.info("  - Total frames: %s", total_count)
        video_logger.info("  - Frames with ball detected: %s", detected_count)
        video_logger.info(
            "  - Detection rate: %.1f%%", detected_count / total_count * 100
        )
        if output_annotated:
            video_logger.info("  - Annotated video saved to: %s", annotated_output_path)
        video_logger.info("  - Labels saved to: %s", parquet_output_path)


if __name__ == "__main__":
    start_time = time()

    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)

    processing_logger = logging_setup.get_processing_logger(LOGGING_DIR)

    unprocessed_videos = logging_setup.get_unprocessed_files(
        INPUT_PATH, "mp4", OUTPUT_PATH, "parquet"
    )

    if not unprocessed_videos:
        processing_logger.info("No new video files found in %s", INPUT_PATH)
    else:
        processing_logger.info("Found %s video(s) to process", len(unprocessed_videos))

        for input_video_file_path in unprocessed_videos:
            video_logger = logging_setup.get_file_logger(
                input_video_file_path.name, LOGGING_DIR
            )
            processing_logger.info("Processing %s", input_video_file_path.name)

            label_xy_positions(
                input_video_file_path,
                OUTPUT_PATH,
                confidence_threshold=0.001,
                video_logger=video_logger,
            )

            video_logger.info("✓ Successfully processed %s", input_video_file_path.name)

        end_time = time()

        # Log final summary
        processing_logger.info("%s", "=" * 60)
        processing_logger.info(
            "Processing completed in %.2f seconds", end_time - start_time
        )
        processing_logger.info("%s", "=" * 60)
