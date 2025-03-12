* Having trouble with NumPy dependencies in NixOS -> try to move forward without using NumPy.
    * Plotly Graph Objects do not depend on Numpy (but most other graphing packages do - including Plotly Express)


## Model Deployment in Flutter App
## Scikit-Learn Models
Deploying a scikit-learn SVM model within a Flutter app involves a few key steps, primarily because scikit-learn is a Python library, and Flutter uses Dart. Here's a breakdown of the process:

**1. Model Serialization (Python)**

* **Train Your Model:** First, train your SVM model using scikit-learn in Python.
* **Serialize the Model:** Use the `pickle` or `joblib` libraries to serialize your trained model. `joblib` is often preferred for scikit-learn models due to its efficiency with large NumPy arrays.

```python
import joblib
from sklearn import svm
from sklearn import datasets

# Train your SVM model
iris = datasets.load_iris()
X, y = iris.data, iris.target
model = svm.SVC()
model.fit(X, y)

# Serialize the model
joblib.dump(model, 'svm_model.joblib')
```

**2. Model Conversion (Python)**

* The most common way to use a python model in Flutter is to use a server, or to convert the model to a format that can be used within a flutter application.
* **ONNX Conversion (Recommended):**
    * ONNX (Open Neural Network Exchange) is a standard format for machine learning models that allows interoperability between different frameworks.
    * You can convert your scikit-learn model to ONNX using the `skl2onnx` library.
    * This is the most effective approach for deploying a scikit-learn model in Flutter, because there are ONNX runtime libraries available for flutter.
    * Install the necessary libraries:

    ```bash
    pip install skl2onnx onnxruntime
    ```

    * Convert your model:

    ```python
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    import joblib

    # Load the serialized model
    model = joblib.load('svm_model.joblib')

    # Convert the model to ONNX
    initial_type = [('float_input', FloatTensorType([None, X.shape[1]])) ] #Replace X.shape[1] with the correct shape.
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    # Save the ONNX model
    with open('svm_model.onnx', 'wb') as f:
        f.write(onnx_model.SerializeToString())
    ```

**3. Flutter Integration**

* **Add ONNX Runtime Flutter Package:**
    * Use the `onnxruntime_flutter` package. Add it to your `pubspec.yaml`:

    ```yaml
    dependencies:
      flutter:
        sdk: flutter
      onnxruntime_flutter: ^latest_version
    ```

* **Add ONNX Model to Assets:**
    * Place your `svm_model.onnx` file in the `assets` folder of your Flutter project.
    * Update your `pubspec.yaml` file:

    ```yaml
    flutter:
      assets:
        - assets/svm_model.onnx
    ```

* **Load and Run the Model:**
    * In your Dart code:

    ```dart
    import 'package:flutter/services.dart';
    import 'package:onnxruntime_flutter/onnxruntime_flutter.dart';
    import 'dart:typed_data';

    Future<List<double>> runSvmInference(List<double> input) async {
      final interpreter = await OrtInterpreter.fromAsset('assets/svm_model.onnx');
      final inputTensor = Tensor.create(input, [1, input.length]); //Reshape to the correct input shape.
      final outputs = await interpreter.run([inputTensor]);
      final outputTensor = outputs.first;
      return outputTensor.data.cast<double>().toList();
    }

    //Example of usage.
    Future<void> exampleFunction(List<double> inputData) async {
        List<double> results = await runSvmInference(inputData);
        print(results);
    }
    ```

**4. Alternative: Server-Side Deployment**

* Another option is to deploy your scikit-learn model as a web service using a framework like Flask or FastAPI.
* Then, your Flutter app can make HTTP requests to the server for predictions.
* This approach offers flexibility but requires a server infrastructure.

**Key Considerations:**

* **Input Data:** Ensure that the input data in your Flutter app matches the format expected by the ONNX model.
* **Error Handling:** Implement robust error handling to handle potential issues during model loading and inference.
* **Performance:** Performance is crucial, especially for real-time applications. Test thoroughly on target devices.
* **Model Size:** ONNX models can be large, so consider model optimization techniques if size becomes a concern.
* **Data preprocessing:** Any preprocessing done in the python training scripts, must also be performed within the flutter application before the data is passed to the ONNX model.


### Tensorflow Models
Deploying a trained TensorFlow model from Python into a Flutter app involves converting it to TensorFlow Lite (TFLite) and then integrating it into your Flutter project. Here's a step-by-step guide:

**1. Convert the TensorFlow Model to TensorFlow Lite (TFLite)**

* **Install TensorFlow:** Ensure you have TensorFlow installed in your Python environment.
* **Load Your Model:** Load your trained TensorFlow model (e.g., a Keras model).
* **Convert to TFLite:** Use the TensorFlow Lite converter to convert your model.

```python
import tensorflow as tf

# Load your TensorFlow model
model = tf.keras.models.load_model('your_model.h5') # or saved_model

# Convert the model
converter = tf.lite.TFLiteConverter.from_keras_model(model) # or from_saved_model
tflite_model = converter.convert()

# Save the TFLite model
with open('your_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

* **Optimization (Optional):**
    * For smaller model sizes and faster inference, you can apply optimizations like quantization.
    * Post-training quantization can significantly reduce the size of your model with minimal impact on accuracy.

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

**2. Integrate the TFLite Model into Your Flutter Project**

* **Add the TFLite Model to Assets:**
    * Create an `assets` folder in your Flutter project's root directory (if you don't have one).
    * Copy your `your_model.tflite` file into the `assets` folder.
    * Update your `pubspec.yaml` file to include the asset:

```yaml
flutter:
  assets:
    - assets/your_model.tflite
```

* **Add the `tflite_flutter` Package:**
    * Add the `tflite_flutter` package to your `pubspec.yaml` file:

```yaml
dependencies:
  flutter:
    sdk: flutter
  tflite_flutter: ^0.10.3 #use the latest version.
```

* **Import the Package:**
    * In your Dart file, import the necessary packages:

```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:typed_data';
```

**3. Load and Use the TFLite Model in Flutter**

* **Load the Model:**
    * Load the TFLite model from the assets folder:

```dart
late Interpreter interpreter;

Future<void> loadModel() async {
  try {
    interpreter = await Interpreter.fromAsset('assets/your_model.tflite');
  } catch (e) {
    print('Error loading model: $e');
  }
}
```

* **Prepare Input Data:**
    * Prepare your input data in the format expected by the model. This will likely involve converting your audio features (e.g., MFCCs) into a `Float32List` or similar.
* **Run Inference:**
    * Run inference using the `interpreter.run()` method:

```dart
Future<List<double>> runInference(Float32List input) async {
  var output = List.filled(outputShape, 0.0).reshape(outputShape); //replace outputshape with the correct shape.
  interpreter.run(input.reshape(inputShape), output); //replace inputshape with the correct shape.
  return output.reshape(outputShape).cast<double>().toList();
}
```

* **Process Output:**
    * Process the output from the model to obtain your desired results.
* **Example of complete usage:**

```dart
Future<List<double>> processAudioAndRunInference(Float32List audioData) async {
  // Preprocess audioData (e.g., calculate MFCCs)
  // ...
  Float32List input = preprocessAudio(audioData); //This would be your audio processing function.
  return await runInference(input);
}
```

**Important Considerations:**

* **Input and Output Shapes:** Ensure that the input and output shapes of your data match the shapes expected by the TFLite model.
* **Data Types:** Pay attention to data types (e.g., `Float32List`).
* **Error Handling:** Implement proper error handling to catch potential issues during model loading and inference.
* **Performance:** Optimize your code for performance, especially when dealing with real-time audio processing.
* **Debugging:** Use logging and debugging tools to identify and resolve issues.
* **Model Metadata:** Ensure you know the input and output shapes of your model. If you are unsure, you can use tools like netron to view your model's architecture.
