# python_assignments_uditpatel
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
