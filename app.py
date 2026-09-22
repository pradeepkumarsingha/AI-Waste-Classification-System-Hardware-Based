import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = r"D:\AI Hardware\models\waste_classifier_finetuned.keras"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.70


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Waste Classifier",
    page_icon="♻️",
    layout="centered"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# CHECK MODEL FILE
# ==========================================

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model()


# ==========================================
# CHECK MODEL OUTPUTS
# ==========================================

if model.output_shape[-1] != len(CLASS_NAMES):
    st.error(
        f"Model has {model.output_shape[-1]} outputs, "
        f"but CLASS_NAMES contains {len(CLASS_NAMES)} classes."
    )
    st.stop()


# ==========================================
# HEADER
# ==========================================

st.title("♻️ AI Waste Classification System")

st.write(
    "Upload an image. The AI model will classify it into one of six "
    "waste categories. Low-confidence predictions are marked as unknown."
)

st.info(
    f"Unknown threshold: {CONFIDENCE_THRESHOLD * 100:.0f}%"
)

st.divider()


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png", "webp"]
)


# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    input_image = Image.open(uploaded_file).convert("RGB")

    st.image(
        input_image,
        caption="Uploaded Image"
    )

    # Resize image
    resized_image = input_image.resize(IMG_SIZE)

    # Convert image to NumPy array
    image_array = np.array(
        resized_image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    # Find highest probability
    predicted_index = int(np.argmax(predictions))
    raw_predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    # Apply threshold
    if confidence < CONFIDENCE_THRESHOLD:
        final_class = "unknown"
        is_unknown = True
    else:
        final_class = raw_predicted_class
        is_unknown = False

    # ======================================
    # RESULT
    # ======================================

    st.divider()
    st.subheader("🔍 Classification Result")

    if is_unknown:
        st.warning(
            f"Detected: UNKNOWN\n\n"
            f"The highest model confidence was "
            f"{confidence * 100:.2f}%, which is below the "
            f"{CONFIDENCE_THRESHOLD * 100:.0f}% threshold."
        )
    else:
        st.success(
            f"Detected: {final_class.upper()}"
        )

    st.metric(
        "Final Confidence",
        f"{confidence * 100:.2f}%"
    )

    st.caption(
        f"Raw model prediction: {raw_predicted_class.upper()}"
    )

    # ======================================
    # CLASS PROBABILITIES
    # ======================================

    st.subheader("Class Probabilities")

    for class_name, probability in zip(CLASS_NAMES, predictions):
        probability = float(probability)

        st.write(
            f"{class_name.capitalize()}: "
            f"{probability * 100:.2f}%"
        )

        st.progress(probability)