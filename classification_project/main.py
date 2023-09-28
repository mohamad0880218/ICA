import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('model/cifar10_model.h5')

# Load an image for prediction (replace 'image_path' with the path to your image file)
image_path = 'image.jpg'
img = image.load_img(image_path, target_size=(32, 32))  # Adjust target size as needed
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values to be between 0 and 1

# Make a prediction
prediction = model.predict(img_array)

# Get the predicted class label
predicted_class = np.argmax(prediction)

# Write the result to a text file
with open('prediction_result.txt', 'w') as f:
    f.write(f"Predicted class: {predicted_class}\n")
