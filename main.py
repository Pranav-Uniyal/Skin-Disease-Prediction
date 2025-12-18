import streamlit as st
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="Skin Disease Classification", page_icon="🩺")

@st.cache_resource
def load_tflite_model():
    model_path = hf_hub_download(
        repo_id="Pranav-Uniyal/Skin_Disease_Classifer",
        filename="skin_disease_model.tflite"
    )
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

st.title("🩺 Skin Disease Classification")

file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB").resize((224,224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = (arr - 127.5) / 127.5  # ResNetV2 style

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = np.argmax(preds)
    st.success(f"Prediction: {labels[idx]}")
    st.bar_chart(preds)
