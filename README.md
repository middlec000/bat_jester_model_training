# Bat Jester
A program that counts the number of soccer ball juggles recorded in an audio file. The model developed in this repo is intended to be used in a phone application (code not stored in this repo).


## Steps

### [1) Data Reformatting](/01_data_reformatting.ipynb)
The data that is used for model training must be converted to appropriate data types and data labels need correcting as well.
- Exogeneous data is converted from `.wav` files to `.csv`s of floats. Each row is a window of audio data. Data is right padded with 0s to fill the last window.
- Endogeneous data is converted from a collection of `.txt` files - each containing a list of timestamps - to boolean arrays of the same length as the exogeneous data where 1 indicates a juggle at that time index and 0 indicates no juggle.
- Endogeneous data is corrected. The labels were created by viewing a video of the juggling session and labeling where a juggle took place. These labels did not correspond exactly to the audio signal, so they were corrected by moving the initial label to the nearby maximum.
- Endogeneous data is saved as `.csv`s. Each row is a 0 (corresponding exogeneous window does not contain a juggle) or a 1 (does contain a juggle).

### [2) Train-Test Split](/02_train_test_split.ipynb)
- Create the train-test split based on file. This splitting method ensures the models do not train on any data from the same files they are tested on which is a realistic evaluation.
- Window overlap for the test data is set to 50% to match how the audio stream will be processed when deployment while the training data had an overlap of 75% to produce more training data windows (more training data).

### [3) Model Training](/03_model_training.ipynb)
- The saved data is loaded and features are extracted (Spectrogram).
- Various models are trained on the data.
- The models are evaluated and compared.
- Models are saved in either ONNX or TF Lite format.

## Future Development
- Extract additional features from the input data.
    - Refine Wavelet approach.
- Add speech recognition to "Start" and "Stop" the juggle counter.

## Contribute
Would love to collaborate on this project!
- Data Access
    - I will need to send you the data. Please see [this discussion](https://github.com/middlec000/bat_jester_model_training/discussions/3).
- Setting Up Your Environment
    - [Clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the GitHub repo.
    - Use `uv` to manage Python and Python packages.
        - Install `uv` on your system - see [here](https://docs.astral.sh/uv/getting-started/installation/).
        - Ensure `uv` is installed:
        ```bash
        uv --version
        ```
        - Use `uv` to install python and the packages for this project:
        ```bash
        uv sync
        ```
        - To add a new package:
        ```bash
        uv add <package_name>
        ```
    - Disregard the `flake.nix` and `flake.lock` files unless you are using `nix` for package management on your machine.
