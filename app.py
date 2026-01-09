import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model once when the app starts
@st.cache_resource
def load_pneumonia_model():
    return load_model("final_vgg_model.h5")

model = load_pneumonia_model()

# Title
st.title("Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image to predict whether it shows signs of pneumonia.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((320,320)) # resize to match model input
    x = np.array(img) / 255.0     # normalize
    x = np.expand_dims(x, axis=0) # batch size

    # Prediction
    pred = model.predict(x)[0][0]
    label = "Pneumonia" if pred > 0.5 else "Normal"
    confidence = pred if pred > 0.5 else 1 - pred

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"### Confidence: **{confidence:.2f}**")
