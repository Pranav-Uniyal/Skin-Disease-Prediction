import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

# Load .h5 model
model = tf.keras.models.load_model(
    "Skin_disease_model.h5",
    compile=False
)

# Class labels (MUST match training order)
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

def predict(image: Image.Image):
    # Resize to model input size
    image = image.resize((224, 224))

    # Convert to numpy
    img_array = np.array(image, dtype=np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # ResNetV2 preprocessing
    img_array = (img_array / 127.5) - 1.0

    # Predict
    preds = model.predict(img_array)[0]

    # Return top probabilities
    return {
        LABELS[i]: float(preds[i])
        for i in range(len(LABELS))
    }

# Gradio Interface
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
