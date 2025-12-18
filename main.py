import streamlit as st
import numpy as np
from PIL import Image
import requests
import io

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/Pranav-Uniyal/Skin_Disease_Classifer"
HF_HEADERS = {
    "Authorization": "Bearer YOUR_HF_API_TOKEN_HERE"
}

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

# -------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------------------------
# UI HEADER
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center;">🩺 AI Skin Disease Detection</h1>
    <p style="text-align:center; color:gray;">
    Upload a skin image and get an AI-powered diagnosis
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# -------------------------------------------------
# IMAGE UPLOAD
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a skin image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# PREDICTION FUNCTION
# -------------------------------------------------
def query_model(image_bytes):
    response = requests.post(
        HF_API_URL,
        headers=HF_HEADERS,
        data=image_bytes
    )
    response.raise_for_status()
    return response.json()

# -------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Analyzing image using AI..."):
        img_bytes = uploaded_file.getvalue()
        result = query_model(img_bytes)

    # -------------------------------------------------
    # DISPLAY RESULTS
    # -------------------------------------------------
    if isinstance(result, list):
        st.subheader("🧠 Prediction Result")

        probs = np.zeros(len(LABELS))
        for item in result:
            label = item["label"]
            score = item["score"]
            if label in LABELS:
                probs[LABELS.index(label)] = score

        top_idx = int(np.argmax(probs))

        st.success(f"**Predicted Disease:** {LABELS[top_idx]}")
        st.write(f"**Confidence:** {probs[top_idx]:.2f}")

        st.subheader("📊 Class Probabilities")
        st.bar_chart(probs)

    else:
        st.error("❌ Model inference failed. Try again later.")

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.divider()
st.markdown(
    "<p style='text-align:center; font-size:12px; color:gray;'>"
    "⚠️ This tool is for educational purposes only. Not a medical diagnosis."
    "</p>",
    unsafe_allow_html=True
)
