import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Path to your trained model
MODEL_PATH = r'D:\BoneFractureDetection\model\fracture_classifier.h5'

# Load model with caching
@st.cache_resource
def load_model_cached():
    return load_model(MODEL_PATH)

model = load_model_cached()

st.title("Bone Fracture Detection")

# Upload image
uploaded_file = st.file_uploader("Upload an X-ray image (jpg, png, webp)", type=['jpg', 'jpeg', 'png', 'webp'])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    fractured_confidence = prediction
    not_fractured_confidence = 1 - prediction
    label = "Fractured" if fractured_confidence >= 0.5 else "Not Fractured"

    # Display results
    st.write(f"Prediction: **{label}**")
    st.write(f"Confidence - Fractured: {fractured_confidence:.4f}")
    st.write(f"Confidence - Not Fractured: {not_fractured_confidence:.4f}")


# Button to trigger fine-tuning






