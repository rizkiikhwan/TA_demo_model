import streamlit as st
import tensorflow as tf
import numpy as np
import io
import requests
import os
from contextlib import redirect_stdout

# --- Mapping model names to Hugging Face URLs ---
MODEL_URLS = {
    'Model EfficientNetV2B3': 'https://huggingface.co/Cryonox756/TA_models/resolve/main/model_EfficientNetV2B3.keras',
    'Model EfficientNetV2S': 'https://huggingface.co/Cryonox756/TA_models/resolve/main/model_EfficientNetV2S.keras',
    'Model ConvNeXtTiny': 'https://huggingface.co/Cryonox756/TA_models/resolve/main/model_ConvNeXtTiny.keras'
}

# --- Download and cache the model ---
@st.cache_resource
def download_and_load_model(url, local_filename):
    if not os.path.exists(local_filename):
        response = requests.get(url)
        with open(local_filename, "wb") as f:
            f.write(response.content)
    return tf.keras.models.load_model(local_filename)

@st.cache_resource
def load_labels():
    with open("eyeDiseasesLabel.txt", "r") as f:
        return f.read().splitlines()

# --- Load all models ---
@st.cache_resource
def load_model(model_name):
    url = MODEL_URLS.get(model_name)
    filename = os.path.basename(url)
    return download_and_load_model(url, filename)

# --- Preprocess image: Resize only (no rescale) ---
def preprocess_image(image, target_size):
    # Resize image using tf.image.resize
    image = tf.image.resize(image, target_size)
    # Convert to dtype float32
    image = tf.cast(image, tf.float32)
    # Add batch dimension: (H, W, C) → (1, H, W, C)
    image = tf.expand_dims(image, axis=0)
    return image

# --- Streamlit App ---
st.title("Evaluation of CNN Models for Multi-Class Classification of Fundus Images")

st.write('Rizki Ikhwan Nur Rahim - 20210801121')
st.write('Universitas Esa Unggul - Ilmu Komputer - Teknik Informatika')
st.write('e-mail: rizkiikhwan96@gmail.com')

st.subheader("Model Architecture")

# Model selection
model_choice = st.selectbox("Choose a model to use", list(MODEL_URLS.keys()))

st.info("📌 Note: First-time setup might take a bit longer due to model download.")
with st.spinner(f"Downloading and loading model: {model_choice}..."):
    model = load_model(model_choice)

labels = load_labels()

# --- Show model architecture ---
summary_str = io.StringIO()
with redirect_stdout(summary_str):
    model.summary()

st.code(summary_str.getvalue())

# Automatically get required input size from a model
input_shape = model.input_shape[1:3]  # Only height and width, ignore batch and channels

st.subheader("Prediction")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = tf.image.decode_image(image_bytes, channels=3)
    image.set_shape([None, None, 3])  # Ensure shape is compatible
    st.image(np.array(image), caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image, input_shape)
        prediction = model.predict(processed_image)

        predicted_class = int(np.argmax(prediction))
        predicted_label = labels[predicted_class]
        confidence = float(np.max(prediction))

        st.success(f"Prediction: {predicted_label} ({confidence:.2%})")

        # Show warning based on confidence level
        if confidence < 0.5:
            st.error("High level Warning: Prediction confidence is very low.")
        elif confidence < 0.7:
            st.warning("Medium level Warning: Prediction confidence is below 70%.")
        elif confidence < 0.8:
            st.info("Low level Warning: Prediction confidence is below 80%.")

        st.subheader("Class Probabilities:")
        for i, prob in enumerate(prediction[0]):
            st.write(f"{labels[i]}: {prob:.2%}")
