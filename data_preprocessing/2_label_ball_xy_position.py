from pathlib import Path
from time import time
import pandas as pd
from video_labeler import SoccerJuggleVideoLabeler
import cv2

INPUT_PATH = Path("data/1_clipped_videos")
OUTPUT_PATH = Path("data/2_ball_xy_positions")
LOGGING_DIR = Path("data/1_to2_logs")


def label_xy_positions(
    input_video_file_path: Path,
    output_dir: Path,
    output_annotated: bool = True,
    confidence_threshold: float = 0.001,
):
    """
    Process a single video and save:
    - Annotated video with ball positions overlaid
    - Parquet file with frame-by-frame ball positions (Frame, x, y)
    """
    print(f"\n{'=' * 60}")
    print(f"Processing video: {input_video_file_path.name}")
    print(f"{'=' * 60}")

    # Initialize labeler with low confidence threshold for ball detection
    labeler = SoccerJuggleVideoLabeler(
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
    parquet_output_path = output_dir / (
        input_video_file_path.stem + "_xy_frame_labels.parquet"
    )
    df[["Frame", "x", "y"]].to_parquet(parquet_output_path, index=False)

    # Calculate detection statistics
    detected_count = df[["x", "y"]].notna().all(axis=1).sum()
    total_count = len(df)

    print("\nSummary:")
    print(f"  - Total frames: {total_count}")
    print(f"  - Frames with ball detected: {detected_count}")
    print(f"  - Detection rate: {detected_count / total_count * 100:.1f}%")
    if output_annotated:
        print(f"  - Annotated video saved to: {annotated_output_path}")
    print(f"  - Labels saved to: {parquet_output_path}")


if __name__ == "__main__":
    start_time = time()

    # Create output directory if it doesn't exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Get all MP4 files from input directory
    input_video_files = list(INPUT_PATH.glob("*.mp4"))
    input_video_files = [
        f for f in input_video_files if f.stem == "PXL_20251215_232210350"
    ]

    if not input_video_files:
        print(f"No video files found in {INPUT_PATH}")
    else:
        print(f"Found {len(input_video_files)} video(s) to process")

        for input_video_file_path in input_video_files:
            label_xy_positions(
                input_video_file_path, OUTPUT_PATH, confidence_threshold=0.001
            )

        end_time = time()

        # Print final summary
        print(f"\n{'=' * 60}")
        print(f"Processing completed in {end_time - start_time:.2f} seconds")
        print(f"{'=' * 60}")
