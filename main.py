import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from huggingface_hub import hf_hub_download

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(
    page_title="Skin Disease Classification",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------
# Load Model from Hugging Face Hub
# -------------------------------
@st.cache_resource
def load_skin_model():
    try:
        model_path = hf_hub_download(
            repo_id="Pranav-Uniyal/Skin_Disease_Classifer",
            filename="Skin_disease_model.h5",
            repo_type="model"
        )
        return load_model(model_path, compile=False)
    except Exception as e:
        st.error("❌ Failed to load model from Hugging Face.")
        st.exception(e)
        st.stop()

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
