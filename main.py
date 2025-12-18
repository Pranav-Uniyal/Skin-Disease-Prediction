import streamlit as st
import numpy as np
from PIL import Image
import urllib.request
import os
import tflite_runtime.interpreter as tflite

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(
    page_title="Skin Disease Classification",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# Model Download (NO EXTRA LIBS)
# -------------------------------
MODEL_URL = (
    "https://huggingface.co/Pranav-Uniyal/"
    "Skin_Disease_Classifer/resolve/main/skin_disease_model.tflite"
)
MODEL_PATH = "skin_disease_model.tflite"

@st.cache_resource
def load_tflite_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("⬇️ Downloading model (one-time)..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# Class Labels
# -------------------------------
labels = [
    'BA- cellulitis',
    'BA-impetigo',
    'FU-athlete-foot',
    'FU-nail-fungus',
    'FU-ringworm',
    'PA-cutaneous-larva-migrans',
    'VI-chickenpox',
    'VI-shingles'
]

# -------------------------------
# UI
# -------------------------------
st.title("🩺 Skin Disease Classification")
st.write("Upload an image of a skin disease to get a predicted diagnosis.")

uploaded_file = st.file_uploader(
    "📁 Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess (ResNet-style)
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = (arr - 127.5) / 127.5

    # Inference
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))

    st.subheader("🔍 Prediction Result")
    st.success(f"**Predicted Disease:** {labels[idx]}")
    st.write(f"**Confidence:** {confidence:.2f}")

    st.subheader("📊 Class Probabilities")
    st.bar_chart(preds)
