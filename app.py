import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# --------------------------------------------------
# Load model (download once, cache automatically)
# --------------------------------------------------
def load_model():
    model_path = hf_hub_download(
        repo_id="Pranav-Uniyal/skin-disease-model",   # 👈 CHANGE IF NEEDED
        filename="Skin_disease_model.h5"
    )
    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# --------------------------------------------------
# Class labels (must match training order)
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
# Prediction function
# --------------------------------------------------
def predict(image: Image.Image):
    image = image.resize((224, 224))

    img_array = np.array(image, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # ResNet-style preprocessing
    img_array = (img_array / 127.5) - 1.0

    preds = model.predict(img_array)[0]

    return {
        LABELS[i]: float(preds[i])
        for i in range(len(LABELS))
    }

# --------------------------------------------------
# Gradio Interface
# --------------------------------------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Skin Image"),
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="🩺 Skin Disease Classification (CNN)",
    description=(
        "Upload a skin image and the AI model will predict the most "
        "likely skin disease.\n\n"
        "⚠️ For educational purposes only. Not a medical diagnosis."
    ),
    allow_flagging="never"
)

demo.launch()
