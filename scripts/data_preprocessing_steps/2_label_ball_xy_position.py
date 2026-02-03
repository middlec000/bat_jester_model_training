import sys
import logging
from pathlib import Path
from time import time
import cv2

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import utils, video_labeler

INPUT_PATH = Path("~/data/bat_jester_model_training/B_clipped_videos").expanduser()
OUTPUT_PATH = Path("~/data/bat_jester_model_training/C_ball_xy_positions").expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/2_logs").expanduser()
CONFIDENCE_THRESHOLD = 0.02


def label_xy_positions(
    input_video_file_path: Path,
    output_dir: Path,
    output_annotated: bool = True,
    confidence_threshold: float = 0.001,
    processing_logger: logging.Logger = None,
):
    """
    Process a single video and save:
    - Annotated video with ball positions overlaid
    - Parquet file with frame-by-frame ball positions (Frame, x, y)

    Logs are written to `processing_logger` (per-run timestamped logger).
    """
    if processing_logger:
        processing_logger.info("Processing video: %s", input_video_file_path.name)

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
    cap.release()

    parquet_output_path = output_dir / (input_video_file_path.stem + ".parquet")
    labeler.ball_positions.write_parquet(parquet_output_path)

    # Calculate detection statistics
    detected_count = labeler.ball_positions[["x", "y"]].drop_nulls().height
    total_count = labeler.ball_positions.height

    if processing_logger:
        processing_logger.info("Summary for %s:", input_video_file_path.name)
        processing_logger.info("  - Total frames: %s", total_count)
        processing_logger.info("  - Frames with ball detected: %s", detected_count)
        processing_logger.info(
            "  - Detection rate: %.1f%%", detected_count / total_count * 100
        )
        if output_annotated:
            processing_logger.info(
                "  - Annotated video saved to: %s", annotated_output_path
            )
        processing_logger.info("  - Labels saved to: %s", parquet_output_path)


def main():
    start_time = time()

    # Parse command-line arguments
    run_all = "--run-all" in sys.argv

    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)

    processing_logger = utils.get_processing_logger(LOGGING_DIR)
    processing_logger.info("Configuration:")
    processing_logger.info("  Input directory: %s", INPUT_PATH)
    processing_logger.info("  Output directory: %s", OUTPUT_PATH)
    processing_logger.info(
        "  Run all videos: %s", "Yes" if run_all else "No (unprocessed only)"
    )

    if run_all:
        unprocessed_videos = sorted(INPUT_PATH.glob("*.mp4"))
    else:
        unprocessed_videos = utils.get_unprocessed_files(
            INPUT_PATH, "mp4", OUTPUT_PATH, "parquet"
        )

    if not unprocessed_videos:
        processing_logger.info("No new video files found in %s", INPUT_PATH)
    else:
        processing_logger.info("Found %s video(s) to process", len(unprocessed_videos))

        for input_video_file_path in unprocessed_videos:
            processing_logger.info("Processing %s", input_video_file_path.name)

            label_xy_positions(
                input_video_file_path,
                OUTPUT_PATH,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                processing_logger=processing_logger,
            )

            processing_logger.info(
                "✓ Successfully processed %s", input_video_file_path.name
            )

        end_time = time()

        # Log final summary
        processing_logger.info("%s", "=" * 60)
        processing_logger.info(
            "Processing completed in %.2f seconds", end_time - start_time
        )
        processing_logger.info("%s", "=" * 60)


if __name__ == "__main__":
    main()

"""
uv run python scripts/data_preprocessing_steps/2_label_ball_xy_position.py --run-all
"""
