import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model/cifar10_model.h5')

# Load an image for prediction (you can replace this with your own image)
# Make sure the image size matches the input shape of the model
image = ...  # Load your image here

# Preprocess the image (resize, normalize, etc.) to match model input
# Replace this with your own preprocessing logic

# Make a prediction
prediction = model.predict(np.expand_dims(image, axis=0))

# Get the predicted class label
predicted_class = np.argmax(prediction)

# Print the result
print(f"Predicted class: {predicted_class}")
