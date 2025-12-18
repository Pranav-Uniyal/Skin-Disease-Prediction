import streamlit as st
import numpy as np
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(
    page_title="Skin Disease Classification",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# Model Download from GitHub Releases
# -------------------------------
MODEL_URL = "https://github.com/Pranav-Uniyal/Skin-Disease-Prediction/releases/download/Skin-disease-model/Skin_disease_model.h5"
MODEL_PATH = "Skin_disease_model.h5"

@st.cache_resource
def load_skin_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first time only)..."):
            r = requests.get(MODEL_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
    return load_model(MODEL_PATH)

model = load_skin_model()

# -------------------------------
# Class Labels
# -------------------------------
class_indices = {
    'BA- cellulitis': 0,
    'BA-impetigo': 1,
    'FU-athlete-foot': 2,
    'FU-nail-fungus': 3,
    'FU-ringworm': 4,
    'PA-cutaneous-larva-migrans': 5,
    'VI-chickenpox': 6,
    'VI-shingles': 7
}

index_to_label = {v: k for k, v in class_indices.items()}

# -------------------------------
# UI
# -------------------------------
st.title("🩺 Skin Disease Classification")
st.write("Upload an image of a skin disease to get a predicted diagnosis.")

uploaded_file = st.file_uploader(
    "📁 Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load & show image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # -------------------------------
    # Preprocess Image (ResNet152V2)
    # -------------------------------
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # -------------------------------
    # Prediction
    # -------------------------------
    with st.spinner("🔍 Analyzing image..."):
        prediction = model.predict(img_array)
        predicted_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

    predicted_label = index_to_label[predicted_index]

    # -------------------------------
    # Results
    # -------------------------------
    st.subheader("🔍 Prediction Result")
    st.success(f"**Predicted Disease:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.subheader("📊 Class Probabilities")
    st.bar_chart(prediction[0])

