import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("Skin_disease_model.h5")  

# Class index to label mapping
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

# App title
st.title("🩺 Skin Disease Classification")
st.write("Upload an image of a skin disease to get a predicted diagnosis.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_label = index_to_label[predicted_index]

    # Display prediction
    st.subheader("🔍 Prediction")
    st.write(f"**Predicted Disease:** {predicted_label}")
    st.bar_chart(prediction[0])
