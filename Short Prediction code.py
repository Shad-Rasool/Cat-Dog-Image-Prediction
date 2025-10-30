#IT is A short Code for Prediction
#This will work afteryou save the model after training it in your PC

import numpy as np
from tensorflow.keras.preprocessing import image
import os

# Load your saved model
from tensorflow import keras
model = keras.models.load_model("cat_dog_model.h5")

# Path to your image
test_img = r"C:\Shad\python\Image classification (Cat or Dog)\Put Test Image here\Snapchat-233816535.jpg"

# Load and preprocess
img = image.load_img(test_img, target_size=(150,150))
x = image.img_to_array(img)/255.0
x = np.expand_dims(x, axis=0)

# Predict
pred = model.predict(x)

# Print result
print("\nPrediction:", "🐶 Dog" if pred[0][0] > 0.5 else "🐱 Cat")
