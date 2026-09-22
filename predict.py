import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


# =========================
# CONFIGURATION
# =========================

MODEL_PATH = r"D:\AI Hardware\models\waste_classifier_finetuned.keras"
CLASSES_PATH = r"D:\AI Hardware\models\classes.txt"
IMAGE_PATH = r"D:\AI Hardware\dataset-resized\paper\paper4.jpg"

IMG_SIZE = (224, 224)

# Change this after testing
CONFIDENCE_THRESHOLD = 0.70


# =========================
# CHECK FILES
# =========================

for file_path in [MODEL_PATH, CLASSES_PATH, IMAGE_PATH]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(MODEL_PATH)


# =========================
# LOAD CLASS NAMES
# =========================

with open(CLASSES_PATH, "r") as f:
    class_names = [line.strip() for line in f if line.strip()]

print("Classes:", class_names)


# =========================
# LOAD IMAGE
# =========================

img = image.load_img(
    IMAGE_PATH,
    target_size=IMG_SIZE,
    color_mode="rgb"
)

img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)


# =========================
# PREDICTION
# =========================

prediction = model.predict(img_array, verbose=0)[0]

predicted_index = int(np.argmax(prediction))
predicted_class = class_names[predicted_index]
confidence = float(prediction[predicted_index])


# =========================
# CONFIDENCE THRESHOLD
# =========================

if confidence < CONFIDENCE_THRESHOLD:
    final_result = "unknown"
else:
    final_result = predicted_class


# =========================
# DISPLAY RESULT
# =========================

print("--------------------------------")
print("AI WASTE CLASSIFICATION")
print("--------------------------------")
print("Prediction :", final_result)
print("Confidence :", f"{confidence * 100:.2f}%")
print("Raw class  :", predicted_class)
print("--------------------------------")


# Display all class probabilities
print("\nClass probabilities:")

for class_name, score in zip(class_names, prediction):
    print(f"{class_name:10s}: {score * 100:.2f}%")