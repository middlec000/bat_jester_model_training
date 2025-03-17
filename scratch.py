import numpy as np
import pywt
import sklearn as sk


class WaveletClassifier:
    def __init__(
        self, wavelet="db4", level=4, test_size=0.2, random_state=42, n_estimators=100
    ):
        """
        Initialize the wavelet classifier.

        Parameters:
        -----------
        wavelet : str, default="db4"
            The wavelet to use for feature extraction
        level : int, default=4
            The level of wavelet decomposition
        test_size : float, default=0.2
            The proportion of the dataset to include in the test split
        random_state : int, default=42
            Controls the randomness of the classifier
        n_estimators : int, default=100
            The number of trees in the forest
        """
        self.wavelet = wavelet
        self.level = level
        self.test_size = test_size
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.clf = None
        self.scaler = None

    def _wavelet_feature_extraction(self, audio_data):
        """Extracts wavelet features from audio data."""
        coeffs = pywt.wavedec(audio_data, self.wavelet, level=self.level)
        features = np.concatenate(
            [np.atleast_1d(np.mean(np.abs(c), axis=0)) for c in coeffs]
        )
        return features

    def fit(self, X, y):
        """
        Train the classifier on the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The training audio data
        y : array-like of shape (n_samples,)
            The target labels

        Returns:
        --------
        self : object
            Returns self
        """
        features = np.array([self._wavelet_feature_extraction(audio) for audio in X])

        # Standardize features
        self.scaler = sk.preprocessing.StandardScaler()
        features = self.scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
            features, y, test_size=self.test_size, random_state=self.random_state
        )

        # Train a classifier (Random Forest in this example)
        self.clf = sk.ensemble.RandomForestClassifier(
            n_estimators=self.n_estimators, random_state=self.random_state
        )
        self.clf.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.clf.predict(X_test)
        accuracy = sk.metrics.accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {accuracy}")

        return self

    def predict(self, X):
        """
        Predict labels for the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input audio data

        Returns:
        --------
        y : array-like of shape (n_samples,)
            The predicted labels
        """
        if self.clf is None:
            raise ValueError("Classifier not fitted. Call fit first.")

        if isinstance(X, list) or (
            isinstance(X, np.ndarray) and len(X.shape) > 1 and X.ndim > 1
        ):
            # Multiple samples
            predictions = []
            for audio in X:
                feature = self._wavelet_feature_extraction(audio).reshape(1, -1)
                feature = self.scaler.transform(feature)
                predictions.append(self.clf.predict(feature)[0])
            return np.array(predictions)
        else:
            # Single sample
            feature = self._wavelet_feature_extraction(X).reshape(1, -1)
            feature = self.scaler.transform(feature)
            return self.clf.predict(feature)[0]


# Example Usage (replace with your actual data)
# Generate dummy audio data and labels
num_samples = 100
audio_length = 6616
audio_arrays = [
    np.random.randn(audio_length) for _ in range(num_samples)
]  # Replace with actual loading
labels = np.random.randint(0, 2, num_samples)  # Binary labels

# Train the classifier
model = WaveletClassifier(wavelet="db4", level=4)
model.fit(audio_arrays, labels)

# Make a prediction on a new audio array
new_audio = np.random.randn(audio_length)  # replace with real data.
prediction = model.predict(new_audio)
print(f"Prediction: {prediction}")
