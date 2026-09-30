# Exercise 4.10
# Image Classification menggunakan VGG19

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

# Load VGG19 model
model = VGG19(weights='imagenet')

# Load image
img_path = 'Elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))

# Convert image to array
x = image.img_to_array(img)

# Add batch dimension
x = np.expand_dims(x, axis=0)

# Preprocess image
x = preprocess_input(x)

# Prediction
predictions = model.predict(x)

# Decode prediction results
results = decode_predictions(predictions)

print(results)