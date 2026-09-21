import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = r"D:\AI Hardware\models\waste_classifier.keras"

CLASS_NAMES = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

IMG_SIZE = (224, 224)


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

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    return model


model = load_model()


# ==========================================
# CHECK MODEL
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

st.title(
    "♻️ AI Waste Classification System"
)

st.write(
    "Upload an image of waste and the AI model "
    "will classify it as Metal, Paper, or Plastic."
)

st.divider()


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ==========================================
# PREDICTION
# ==========================================

if uploaded_file is not None:

    # --------------------------------------
    # Load image
    # --------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # --------------------------------------
    # Display image
    # --------------------------------------

    st.image(
        image,
        caption="Uploaded Image"
    )


    # --------------------------------------
    # Resize
    # --------------------------------------

    resized_image = image.resize(
        IMG_SIZE
    )


    # --------------------------------------
    # Convert to NumPy
    # --------------------------------------

    image_array = np.array(
        resized_image,
        dtype=np.float32
    )


    # --------------------------------------
    # Add batch dimension
    # --------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # --------------------------------------
    # Prediction
    # --------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    # --------------------------------------
    # Prediction details
    # --------------------------------------

    predicted_index = np.argmax(
        predictions
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        predictions[predicted_index] * 100
    )


    # ======================================
    # RESULT
    # ======================================

    st.divider()

    st.subheader(
        "🔍 Classification Result"
    )


    st.success(
        f"Detected: {predicted_class.upper()}"
    )


    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # ======================================
    # ALL PROBABILITIES
    # ======================================

    st.subheader(
        "Class Probabilities"
    )


    for i, class_name in enumerate(
        CLASS_NAMES
    ):

        probability = (
            predictions[i] * 100
        )

        st.write(
            f"**{class_name.capitalize()}** "
            f"{probability:.2f}%"
        )

        st.progress(
            float(predictions[i])
        )