import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

from tensorflow import keras

# Load model
model = keras.models.load_model("cat_dog_classifier.h5", compile=False)

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Load trained model
# Function to predict image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return "Dog" if prediction[0][0] > 0.5 else "Cat"

# Streamlit UI
st.title("🐶🐱 Cat vs. Dog Image Classifier")
st.write("Upload an image to predict whether it's a cat or a dog!")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Save uploaded image
    img_path = "uploaded_image.jpg"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    # Display image
    st.image(img_path, caption="Uploaded Image", use_container_width=True)

    # Predict
    label = predict_image(img_path)
    st.write(f"****Prediction: {label}****")
