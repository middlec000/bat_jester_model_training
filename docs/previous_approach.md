Moving away from this approach for data preprocessing (reformatting) in favor of a more automated data labeling and preprocessing pipeline - see [development_plan](development_plan.md).

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
