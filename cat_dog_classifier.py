print("Script started ✅")
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

print("📥 Downloading dataset...")

import tensorflow as tf, os, zipfile

url = "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip"
dataset_path = tf.keras.utils.get_file("cats_and_dogs_filtered.zip", origin=url)

# Extract manually (to ensure path correctness)
extract_path = os.path.join(os.path.dirname(dataset_path), "cats_and_dogs_filtered_extracted")
if not os.path.exists(extract_path):
    with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

base_dir = os.path.join(extract_path, "cats_and_dogs_filtered")
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

print("✅ Dataset extracted to:", base_dir)
print("📂 Train folder exists:", os.path.exists(train_dir))
print("📂 Validation folder exists:", os.path.exists(val_dir))
# 🧹 Prepare Data
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=(150,150), batch_size=20, class_mode='binary')
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(150,150), batch_size=20, class_mode='binary')

# 🧠 Build CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


print("Model built successfully ✅")

#🚀 Train Model
print("\n🧠 Training model...")
model.fit(train_gen, epochs=5, validation_data=val_gen)

print("Training completed ✅")

#These 2 Codes below are used to save model so It does not train every time but can't be uploaded on github due to file size so when using on PC train it onece with save model uncommented and then can comment traing data and uncomment on load data to save time!!

'''

# Save model to prevent training it every time

model.save("cat_dog_model.h5")
#print("Model saved ✅")

# Load your saved model
from tensorflow import keras
model = keras.models.load_model("cat_dog_model.h5")
'''

# 📊 Evaluate
val_loss, val_acc = model.evaluate(val_gen)
print(f"\n✅ Validation Accuracy: {val_acc*100:.2f}%")

# 🐱 Test Prediction
import numpy as np
from tensorflow.keras.preprocessing import image

test_img = os.path.join(val_dir, 'cats', 'cat.2000.jpg')
img = image.load_img(test_img, target_size=(150,150))
x = image.img_to_array(img)/255.0
x = np.expand_dims(x, axis=0)

pred = model.predict(x)
print("\nPrediction:", "🐶 Dog" if pred[0] > 0.5 else "🐱 Cat")