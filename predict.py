import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model
model = tf.keras.models.load_model(
    "D:\AI Hardware\models\waste_classifier.keras"
)

# Load classes
with open("D:\AI Hardware\models\classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Image path
IMAGE_PATH = "D:\AI Hardware\dataset-resized\paper\paper4.jpg"

# Load image
img = image.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)

img_array = image.img_to_array(img)

img_array = np.expand_dims(
    img_array,
    axis=0
)

# Prediction
prediction = model.predict(img_array)

predicted_index = np.argmax(prediction[0])

predicted_class = class_names[predicted_index]

confidence = prediction[0][predicted_index] * 100

print("--------------------------------")
print("AI WASTE CLASSIFICATION")
print("--------------------------------")

print("Prediction :", predicted_class)
print("Confidence :", f"{confidence:.2f}%")