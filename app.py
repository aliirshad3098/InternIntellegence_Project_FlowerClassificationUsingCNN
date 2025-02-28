from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

# Streamlit header for the app
st.header("Flower Classification CNN Model")

# Class labels
flowers_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the pre-trained model
def load_model_file(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model_path = "Flower_Recognition_Model.keras"
model = load_model_file(model_path)

if model is None:
    st.stop()

# Function to classify an uploaded image
def classify_img(image):
    try:
        input_image = tf.keras.utils.load_img(image, target_size=(180, 180))
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = np.expand_dims(input_image_array, axis=0)
        predictions = model.predict(input_image_exp_dim)
        max_prob = np.argmax(predictions)
        confidence_score = np.max(predictions)
        threshold = 0.9
        if confidence_score < threshold:
            outcome = "The image is not a flower."
        else:
            outcome = f"The Image belongs to {flowers_names[max_prob]} with a score of {confidence_score * 100:.2f}%"
        return outcome
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None

# File uploader for the user to upload an image
def upload_image():
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    return uploaded_file

uploaded_file = upload_image()

# Check if a file is uploaded
if uploaded_file is not None:
    st.image(uploaded_file, width=400)
    with st.spinner("Classifying image..."):
        result = classify_img(uploaded_file)
    if result is not None:
        st.write(result)
else:
    st.write("Please upload an image")
