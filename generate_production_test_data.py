import os
import pickle
import json
import base64
import onnxruntime as ort
from sklearn import metrics

PREPROCESSED_DATA_TO_USE = "preprocessed_data_2025-03-05 15:15:03.477584.pkl"

# Load the preprocessed data
with open(
    os.path.join("data", "preprocessed_data", PREPROCESSED_DATA_TO_USE), "rb"
) as file:
    tracker = pickle.load(file)

print(tracker.keys())
recording = "PXL_20250222_012947893.TS"

# Load the ONNX model
model_path = "trained_models/model_svm_2025-03-12 16:57:09.189054.onnx"
session = ort.InferenceSession(model_path)


tracker[recording]["x"][0].shape
# Initialize an empty list to store predictions
# Make predictions in batch
input_name = session.get_inputs()[0].name
# Stack all examples into a single batch
batch_input = tracker[recording]["x"]
# Run inference on the batch
results = session.run(None, {input_name: batch_input})
# The first output contains all predictions
predicted = results[0].tolist()

print("Accuracy Score")
print(metrics.accuracy_score(tracker[recording]["y"], predicted))

# Get the audio data and expected class arrays
audio_data = tracker[recording]["x"]  # 2D array
expected_class = tracker[recording]["y"]  # 1D array

# Extract confidence scores from model output
# For SVM models, the second output often contains decision scores if available
if len(results) > 1:
    # Use the second output as confidence scores
    confidence_scores = results[1]
    # Take the maximum absolute value as confidence (for SVM decision scores)
    expected_confidence = (
        abs(confidence_scores).max(axis=1).tolist()
        if confidence_scores.ndim > 1
        else abs(confidence_scores).tolist()
    )
else:
    # If no explicit confidence scores, use a simple normalization of the output
    # This is a placeholder approach - actual confidence calculation depends on the specific model
    print("No explicit confidence scores found, using normalized output")
    expected_confidence = [1.0] * len(
        predicted
    )  # Default to 1.0 confidence when unavailable

# Save as JSON metadata structure with proper array handling
example_data = {
    # Convert NumPy array to base64 string
    "audio_data_base64": base64.b64encode(audio_data.tobytes()).decode("utf-8"),
    "audio_data_shape": list(audio_data.shape),  # Convert shape tuple to list for JSON
    "audio_data_dtype": str(audio_data.dtype),
    # Convert expected_class to list if it's a NumPy array
    "expected_class": expected_class.tolist()
    if hasattr(expected_class, "tolist")
    else list(expected_class),
    "expected_class_shape": list(expected_class.shape)
    if hasattr(expected_class, "shape")
    else [len(expected_class)],
    "expected_class_dtype": str(expected_class.dtype)
    if hasattr(expected_class, "dtype")
    else "int",
    "expected_confidence": expected_confidence,
}

# Ensure the directory exists
os.makedirs("data/production_test_data", exist_ok=True)

# Save to a JSON file
with open(f"data/production_test_data/{recording}.json", "w") as f:
    json.dump(example_data, f)

print(f"Saved example to data/production_test_data/{recording}.json")
