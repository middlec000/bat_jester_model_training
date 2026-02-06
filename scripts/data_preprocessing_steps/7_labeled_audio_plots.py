from pathlib import Path
from time import time
import sys
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Add parent directory to path so we can import bat_logging
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import utils

LABELS_DIR = Path("~/data/bat_jester_model_training/E_juggle_labels").expanduser()
AUDIO_DIR = Path(
    "~/data/bat_jester_model_training/F_audio_extracted_from_videos"
).expanduser()
OUTPUT_DIR = Path(
    "~/data/bat_jester_model_training/H_audio_plots_with_labels"
).expanduser()
LOGGING_DIR = Path("~/data/bat_jester_model_training/7_logs").expanduser()


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

    input_files = list(AUDIO_DIR.glob("*.wav"))

    if run_mode == "all":
        unprocessed_files = input_files
        logger.info(
            f"Processing all files (--run flag 'all'): {len(unprocessed_files)} files"
        )
    elif run_mode == "new":
        unprocessed_files = utils.get_unprocessed_files(
            input_dir=AUDIO_DIR,
            input_filetype="wav",
            output_dir=OUTPUT_DIR,
            output_filetype="png",
        )
        logger.info(f"Processing unprocessed files: {len(unprocessed_files)} files")
    else:
        candidate_files = input_files
        unprocessed_files = [
            f for f in candidate_files if any(s in f.name for s in substrings)
        ]
        logger.info(
            f"Filtering with substrings=%s: {len(candidate_files)} -> {len(unprocessed_files)} files",
            substrings,
        )

    for audio_filename in unprocessed_files:
        label_file = LABELS_DIR / (audio_filename.stem + ".txt")
        output_file = OUTPUT_DIR / audio_filename.with_suffix(".png").name
        start_time = time()
        try:
            # Load audio data
            audio_data, sr = librosa.load(str(audio_filename), sr=None)
            duration = librosa.get_duration(y=audio_data, sr=sr)

            # Load timestamp labels
            if label_file.exists():
                with open(label_file, "r") as f:
                    timestamps = [
                        float(line.strip()) for line in f.readlines() if line.strip()
                    ]
            else:
                timestamps = []

            # Create figure and plot waveform
            fig, ax = plt.subplots(figsize=(12, 4), dpi=100)

            # Plot audio waveform
            time_axis = np.linspace(0, duration, len(audio_data))
            ax.plot(
                time_axis,
                audio_data,
                linewidth=0.5,
                color="blue",
                alpha=0.7,
                label="Audio",
            )

            # Overlay timestamp labels as vertical lines
            for timestamp in timestamps:
                if 0 <= timestamp <= duration:
                    ax.axvline(
                        x=timestamp,
                        color="red",
                        linewidth=1.5,
                        alpha=0.6,
                        linestyle="--",
                    )

            # Add legend only if there are timestamps
            if timestamps:
                ax.axvline(
                    x=timestamps[0],
                    color="red",
                    linewidth=1.5,
                    alpha=0.6,
                    linestyle="--",
                    label="Juggle Catches",
                )

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"Audio: {audio_filename.stem}")
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(np.linspace(0, duration, 10))

            # Save plot
            output_file.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(output_file), bbox_inches="tight", dpi=100)
            plt.close(fig)

            elapsed_time = time() - start_time
            logger.info(
                f"Generated plot for {audio_filename.name} in {elapsed_time:.2f} seconds and saved to {output_file.name}"
            )

        except Exception as e:
            logger.error(f"Failed to generate plot for {audio_filename.name}: {str(e)}")


if __name__ == "__main__":
    main()

"""
uv run python scripts/data_preprocessing_steps/7_labeled_audio_plots.py --run all
"""
