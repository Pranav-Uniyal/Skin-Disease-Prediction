import streamlit as st
import numpy as np
from PIL import Image
import requests
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --------------------------------------------------
# Download model from GitHub Releases (once)
# --------------------------------------------------
MODEL_URL = (
    "https://github.com/Pranav-Uniyal/"
    "Skin-Disease-Prediction/releases/download/"
    "keras-model/skin_disease_fixed.keras"
)
MODEL_PATH = "skin_disease_fixed.keras"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading model..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

    # ✅ Correct loader for .keras models
    return tf.keras.models.load_model(MODEL_PATH)

# Load model
model = load_trained_model()

# --------------------------------------------------
# Class labels (MUST match training order)
# --------------------------------------------------
LABELS = [
    'BA- cellulitis',
    'BA-impetigo',
    'FU-athlete-foot',
    'FU-nail-fungus',
    'FU-ringworm',
    'PA-cutaneous-larva-migrans',
    'VI-chickenpox',
    'VI-shingles'
]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🩺 Skin Disease Classification")
st.write(
    "Upload a skin image and the model will predict the most likely disease. "
    "This is for **educational purposes only**."
)

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # --------------------------------------------------
    # Preprocessing (must match training)
    # --------------------------------------------------
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = (img_array / 127.5) - 1.0  # ResNet-style scaling

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    preds = model.predict(img_array)[0]
    predicted_idx = int(np.argmax(preds))

    st.subheader("🔍 Prediction")
    st.success(f"**{LABELS[predicted_idx]}**")

    st.subheader("📊 Confidence Scores")
    st.bar_chart(
        {LABELS[i]: float(preds[i]) for i in range(len(LABELS))}
    )

st.caption(
    "⚠️ This tool is for educational purposes only and not a medical diagnosis."
)
