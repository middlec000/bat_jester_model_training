from pathlib import Path
from time import time
import sys
import argparse
import polars as pl
import numpy as np
import librosa

# Add parent directory to path so we can import bat_logging / utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import utils

LABELS_DIR = Path("~/data/bat_jester_model_training/E_juggle_labels").expanduser()
AUDIO_DIR = Path(
    "~/data/bat_jester_model_training/F_audio_extracted_from_videos"
).expanduser()
OUTPUT_DIR = Path("~/data/bat_jester_model_training/G_synced_audio_labels").expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/6_logs").expanduser()

# Search neighborhood specified in seconds (converted to samples at runtime after loading audio)
# Legacy default matched ~2500 samples at 22050 Hz
AUDIO_NEIGHBORHOOD_SECONDS = 0.1
SAMPLE_RATE = 48000  # Use librosa load default if None


def transform_labels_timestamp_to_vector(
    labels: list, sample_rate: int, frame_rate: int, data_size: int
) -> np.ndarray:
    """Create a boolean vector (0/1) of length data_size with 1s at label indices."""
    labels_vector = np.zeros(data_size, dtype=np.int8)
    for seconds in labels:
        index = int(seconds * sample_rate)
        if 0 <= index < data_size:
            labels_vector[index] = 1
    return labels_vector


def shift_labels_to_local_max(
    boolean_array: np.ndarray, real_array: np.ndarray, search_step: int
) -> np.ndarray:
    """For each 1 in boolean_array, move it to index within +/- search_step that has the largest absolute value in real_array.

    If two 1s are closer than search_step this will raise a ValueError to avoid ambiguity.
    """
    if len(boolean_array) != len(real_array):
        raise ValueError("Boolean and real arrays must have the same length.")
    ones_indices = np.where(boolean_array == 1)[0]

    # Ensure labels aren't too close
    for i in range(len(ones_indices)):
        for j in range(i + 1, len(ones_indices)):
            if abs(ones_indices[i] - ones_indices[j]) < search_step:
                raise ValueError(
                    "Labels are too close together for unambiguous shifting."
                )

    labels_corrected = np.zeros_like(boolean_array)
    for index in ones_indices:
        start = max(0, index - search_step)
        end = min(len(real_array), index + search_step + 1)
        search_window = np.abs(real_array[start:end])
        max_index_relative = int(np.argmax(search_window))
        max_index_absolute = start + max_index_relative
        labels_corrected[max_index_absolute] = 1
    return labels_corrected


def save_parquet(
    audio_values: np.ndarray,
    original_labels_vector: np.ndarray,
    corrected_labels_vector: np.ndarray,
    output_path: Path,
):
    """Save the audio and label arrays to a parquet file with columns 'audio' and 'label'."""
    df = pl.DataFrame(
        {
            "audio": audio_values.astype(float),
            "original_label": original_labels_vector.astype(int),
            "corrected_label": corrected_labels_vector.astype(int),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Sync labels to local audio maxima and save parquet files"
    )
    parser.add_argument(
        "--run",
        nargs="+",
        default=["new"],
        help='Run mode: "all" to process all files, "new" to process only unprocessed files (default), or provide one or more substrings to process files whose names contain any substring',
    )
    parser.add_argument(
        "--frame-rate", type=int, default=30, help="Frame rate used in timestamps"
    )
    parser.add_argument(
        "--neighborhood",
        type=float,
        default=AUDIO_NEIGHBORHOOD_SECONDS,
        help="Search neighborhood in seconds (will be converted to samples using the audio file's sr)",
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
        f"Sync labels to local maxima (run_mode: {run_mode}, substrings: {substrings})"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    logger = utils.get_processing_logger(LOGGING_DIR)

    input_files = list(AUDIO_DIR.glob("*.wav"))

    if run_mode == "all":
        candidates = input_files
        logger.info(f"Processing all files (--run flag 'all'): {len(candidates)} files")
    elif run_mode == "new":
        candidates = utils.get_unprocessed_files(
            input_dir=AUDIO_DIR,
            input_filetype="wav",
            output_dir=OUTPUT_DIR,
            output_filetype="parquet",
        )
        logger.info(f"Processing unprocessed files: {len(candidates)} files")
    else:
        candidates = input_files
        candidates = [f for f in candidates if any(s in f.name for s in substrings)]
        logger.info(
            f"Filtering with substrings=%s: {len(input_files)} -> {len(candidates)} files",
            substrings,
        )

    for audio_filename in candidates:
        label_file = LABELS_DIR / (audio_filename.stem + ".txt")
        output_file = OUTPUT_DIR / audio_filename.with_suffix(".parquet").name
        start_time = time()
        try:
            # Load audio
            audio_data, sr = librosa.load(str(audio_filename), sr=SAMPLE_RATE)

            # Load timestamps
            if label_file.exists():
                with open(label_file, "r") as f:
                    timestamps = [
                        float(line.strip()) for line in f.readlines() if line.strip()
                    ]
            else:
                timestamps = []

            # Build label vector
            labels_vector = transform_labels_timestamp_to_vector(
                labels=timestamps,
                sample_rate=sr,
                frame_rate=args.frame_rate,
                data_size=len(audio_data),
            )

            # Convert neighborhood (seconds) to samples and shift labels
            neighborhood_samples = int(args.neighborhood * sr)
            labels_shifted = shift_labels_to_local_max(
                labels_vector, audio_data, neighborhood_samples
            )

            # Save parquet
            save_parquet(
                audio_values=audio_data,
                original_labels_vector=labels_vector,
                corrected_labels_vector=labels_shifted,
                output_path=output_file,
            )

            elapsed_time = time() - start_time
            logger.info(
                f"Synced labels for {audio_filename.name} with sample rate {sr} in {elapsed_time:.2f}s and saved to {output_file}"
            )

        except Exception as e:
            logger.error(f"Failed to sync labels for {audio_filename.name}: {str(e)}")


if __name__ == "__main__":
    main()

"""
uv run python scripts/data_preprocessing_steps/6_sync_labels_to_max_audio.py --run all
"""
