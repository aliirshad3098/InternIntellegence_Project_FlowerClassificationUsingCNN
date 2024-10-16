import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

# Streamlit header for the app
st.header("Flower Classification CNN Model")

# Class labels
flowers_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# Load the pre-trained model
model = load_model("Flower_Recognition_Model.keras")

# Function to classify an uploaded image
def classify_img(image):
    input_image = tf.keras.utils.load_img(image, target_size=(180, 180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = np.expand_dims(input_image_array, axis=0)  # Add batch dimension
    predictions = model.predict(input_image_exp_dim)
    max_prob = np.argmax(predictions)
    outcome = f"The image belongs to {flowers_names[max_prob]} with a score of {np.max(predictions) * 100:.2f}%"
    return outcome

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# Check if a file is uploaded
if uploaded_file is not None:
    st.image(uploaded_file, width=200)  # Display the uploaded image
    # Run classification and display result
    result = classify_img(uploaded_file)
    st.write(result)  # Output the result
else:
    st.write("Please upload an image")
