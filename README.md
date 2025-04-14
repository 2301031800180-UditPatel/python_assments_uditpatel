# python_assments_uditpatel
Udit Patel, 2301031800180, Cse D-4

# Image Classifier using TensorFlow and Keras

This mini project is a simple handwritten digit classifier using the MNIST dataset. It demonstrates the use of advanced Python libraries including TensorFlow, Keras, NumPy, and Matplotlib.

## Libraries Used
- TensorFlow
- Keras
- NumPy
- Matplotlib

## How to Run

1. Install required libraries:
```
pip install tensorflow numpy matplotlib
```

2. Run the script:
```
python image_classifier.py
```

## Output
The model trains on the MNIST dataset and visualizes a few predictions.

## Author
Udit



3. python code :

   # Image Classifier using Keras and TensorFlow (MNIST Dataset)

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predict on test data
predictions = model.predict(x_test)

# Plot some predictions
plt.figure(figsize=(10, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Pred: {np.argmax(predictions[i])}")
    plt.axis('off')
plt.tight_layout()
plt.show()
