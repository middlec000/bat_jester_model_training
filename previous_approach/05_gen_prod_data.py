import os
import pandas as pd
import numpy as np
import onnxruntime as rt

# Create production data directory if it doesn't exist
os.makedirs("data/production_test_data", exist_ok=True)

# Load the trained SVM model
model_path = "data/trained_models/model_svm_2025-04-02 15:32:28.259206.onnx"
sess = rt.InferenceSession(model_path)

# Get the input name for the model
input_name = sess.get_inputs()[0].name

# Load the training data
train_data = pd.read_csv("data/train_test_split/train_x.csv")
print(f"Loaded training data with shape: {train_data.shape}")

# Make predictions
input_data = train_data.values.astype(np.float32)
predictions = sess.run(None, {input_name: input_data})[0]

# Convert predictions to DataFrame
pred_df = pd.DataFrame(predictions, columns=["predicted_value"])

# Save the predictions and original data
pred_df.to_csv("data/production_test_data/predictions.csv", index=False)
train_data.to_csv("data/production_test_data/original_data.csv", index=False)

print(f"Saved predictions and original data to data/production_test_data/")
print(f"Number of predictions: {len(predictions)}")
