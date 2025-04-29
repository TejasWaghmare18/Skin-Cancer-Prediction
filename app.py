import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("cnn1_model.h5")

# Define function to preprocess the uploaded image
def preprocess_image(image, target_size=(64, 64)):  # Adjusted target size
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0
    return image

# Define prediction function with error handling
def predict(image):
    processed_image = preprocess_image(image)
    try:
        prediction = model.predict(processed_image)
        return prediction[0][0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        print("Error during prediction:", e)
        return None

# Add background image (optional)
def add_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background: url("https://images.unsplash.com/photo-1576765607924-4d73a5ff1aaf");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #00416A;'>Skin Cancer Prediction Using CNN</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #4F4F4F;'>Upload an image of a skin lesion to predict if it's cancerous or benign.</p>", unsafe_allow_html=True)

# Upload image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        prediction = predict(image)
        if prediction is not None:
            result = "Cancerous" if prediction > 0.5 else "Benign"
            confidence = prediction * 100 if prediction > 0.5 else (1 - prediction) * 100
            color = "#E74C3C" if prediction > 0.5 else "#27AE60"
            icon = "🚨" if prediction > 0.5 else "✅"
            
            # Display prediction result
            st.markdown(
                f"<h2 style='text-align: center; color: {color};'>{icon} {result}</h2>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<p style='text-align: center; font-size: 18px; color: #4F4F4F;'>Confidence: <strong>{confidence:.2f}%</strong></p>",
                unsafe_allow_html=True
            )
        else:
            st.error("Could not complete the prediction. Please try again with a different image.")
