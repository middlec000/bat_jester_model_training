from pathlib import Path
from time import time
import sys
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import utils

INPUT_DIR = Path("~/data/bat_jester_model_training/G_synced_audio_labels").expanduser()
OUTPUT_DIR = Path(
    "~/data/bat_jester_model_training/H_audio_plots_with_labels"
).expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/7_logs").expanduser()
AUDIO_NEIGHBORHOOD_SECONDS = 0.1
SAMPLE_RATE = 48000


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate labeled audio plots")
    parser.add_argument(
        "--run",
        nargs="+",
        default=["new"],
        help='Run mode: "all" to process all files, "new" to process only unprocessed files (default), or provide one or more substrings to process all files whose names contain any substring',
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
        f"Generate labeled audio plots (run_mode: {run_mode}, substrings: {substrings})"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGING_DIR.mkdir(parents=True, exist_ok=True)
    logger = utils.get_processing_logger(LOGGING_DIR)

    input_files = list(INPUT_DIR.glob("*.parquet"))

    if run_mode == "all":
        files = input_files
        logger.info(f"Processing all files (--run flag 'all'): {len(files)} files")
    elif run_mode == "new":
        files = utils.get_unprocessed_files(
            input_dir=INPUT_DIR,
            input_filetype="parquet",
            output_dir=OUTPUT_DIR,
            output_filetype="png",
        )
        logger.info(f"Processing unprocessed files: {len(files)} files")
    else:
        candidate_files = input_files
        files = [f for f in candidate_files if any(s in f.name for s in substrings)]
        logger.info(
            f"Filtering with substrings=%s: {len(candidate_files)} -> {len(files)} files",
            substrings,
        )

    for parquet_filename in files:
        start_time = time()
        try:
            # Read synced audio + labels parquet
            df = pl.read_parquet(parquet_filename)
            audio_data = df["audio"].to_numpy()
            original_labels = df["original_label"].to_numpy()
            corrected_labels = df["corrected_label"].to_numpy()

            # Use configured sample rate
            sr = SAMPLE_RATE
            if sr is None:
                raise ValueError(
                    "SAMPLE_RATE must be set to convert sample indices to time."
                )
            duration = len(audio_data) / sr

            orig_indices = np.where(original_labels == 1)[0]
            corr_indices = np.where(corrected_labels == 1)[0]

            print(
                f"File: {parquet_filename.name}, sr={sr}, samples={len(audio_data)}, duration={duration:.3f}s"
            )

            # Plot waveform and overlay original (red) and corrected (green) labels
            fig, ax = plt.subplots(figsize=(12, 4), dpi=100)
            time_axis = np.linspace(0, duration, len(audio_data))
            ax.plot(
                time_axis,
                audio_data,
                linewidth=0.5,
                color="blue",
                alpha=0.7,
                label="Audio",
            )
            for oi in orig_indices:
                ax.axvline(
                    x=oi / sr, color="red", linewidth=1.0, alpha=0.6, linestyle="--"
                )
            for ci in corr_indices:
                ax.axvline(
                    x=ci / sr, color="green", linewidth=1.0, alpha=0.6, linestyle="-"
                )
            if orig_indices.size > 0:
                ax.axvline(
                    x=orig_indices[0] / sr,
                    color="red",
                    linewidth=1.5,
                    alpha=0.6,
                    linestyle="--",
                    label="Original Label",
                )
            if corr_indices.size > 0:
                ax.axvline(
                    x=corr_indices[0] / sr,
                    color="green",
                    linewidth=1.5,
                    alpha=0.6,
                    linestyle="-",
                    label="Corrected Label",
                )
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Audio: {parquet_filename.stem}")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(np.linspace(0, duration, 10))

            output_file = OUTPUT_DIR / parquet_filename.with_suffix(".png").name
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_file), bbox_inches="tight", dpi=100)
            plt.close(fig)

            elapsed_time = time() - start_time
            logger.info(
                f"Generated plot for {parquet_filename.name} in {elapsed_time:.2f} seconds and saved to {output_file.name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to generate plot for {parquet_filename.name}: {str(e)}"
            )


if __name__ == "__main__":
    main()

"""
uv run python scripts/data_preprocessing_steps/7_labeled_audio_plots.py --run all
"""
