import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def create_yamnet_binary_classifier():
    """
    Creates a model that uses YAMNet as a feature extractor and adds a binary classification layer.
    """

    # Load YAMNet from TensorFlow Hub
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")

    class YAMNetBinary(tf.keras.Model):
        def __init__(self, yamnet_model):
            super(YAMNetBinary, self).__init__()
            self.yamnet = yamnet_model
            self.dense = tf.keras.layers.Dense(128, activation="relu")
            self.binary_output = tf.keras.layers.Dense(1, activation="sigmoid")

        def call(self, inputs):
            # Reshape input to YAMNet's expected shape (pad or truncate)
            reshaped_inputs = tf.reshape(
                inputs, [-1, 15600]
            )  # YAMNet expects audio of length 15600.
            # Pad or truncate the input to the required length
            padded_inputs = tf.pad(
                reshaped_inputs,
                [[0, 0], [0, max(0, 15600 - tf.shape(reshaped_inputs)[1])]],
            )
            truncated_inputs = padded_inputs[:, :15600]

            # Run YAMNet
            scores, embeddings, spectrogram = self.yamnet(truncated_inputs)

            # Binary classification layers
            x = self.dense(embeddings)
            binary_predictions = self.binary_output(x)

            return binary_predictions

    model = YAMNetBinary(yamnet_model)
    return model


# Example Usage:
model = create_yamnet_binary_classifier()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Create a dummy input array of size 6616
dummy_input = np.random.rand(1, 6616).astype(np.float32)

# Make a prediction
prediction = model(dummy_input)

print(f"Prediction: {prediction}")

# Example of training.
# Generate dummy training data (replace with your actual data)
num_samples = 100
input_data = np.random.rand(num_samples, 6616).astype(np.float32)
labels = np.random.randint(0, 2, size=(num_samples, 1)).astype(
    np.float32
)  # Binary labels

model.fit(input_data, labels, epochs=5)  # Train the model.
