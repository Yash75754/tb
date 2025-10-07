# src/app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import io

MODEL_DIR = "../models/best_model"  # adjust path if running from project root

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_DIR)

model = load_model()
IMG_SIZE = (224,224)

st.title("Tuberculosis Detection from Chest X-ray")
st.write("Upload a chest X-ray image. Model returns probability of tuberculosis. Not a medical diagnosis.")

uploaded = st.file_uploader("Choose an X-ray image", type=['png','jpg','jpeg'])
if uploaded:
    image = Image.open(uploaded).convert('RGB')
    st.image(image, caption="Uploaded image", use_column_width=True)
    # preprocess
    img = image.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    proba = model.predict(arr)[0][0]
    st.metric("TB probability", f"{proba*100:.2f}%")
    label = "TB" if proba >= 0.5 else "NORMAL"
    st.success(f"Predicted: {label}")
    st.warning("This tool is for educational/demo purposes only — not a clinical diagnosis.")
