# Import necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your specific .h5 model file
model_path = "cot.h5"
loaded_model = tf.keras.models.load_model(model_path)

# Class names
class_names = [
    "aphids",
    "army worm",
    "bacterial blight",
    "bowderly miledev",
    "curle virus",
    "diseased cotton leaf",
    "diseased cotton plant",
    "fassuriam wilt",
    "fresh cotton leaf",
    "fresh cotton plant",
    "target spot"
]
# Solutions for each class
solutions = {
    "aphids": "dainosure 2ml and starthin 25 gram...",
    "army worm": "Solution for army worm...",
    "bacterial blight": "mizine 5 gram and copper oxicloride 50 gram...",
    "bowderly miledev": "index 30 gram or nativo 15 gram...",
    "curle virus": "yellow thrips and interpitten 30 ml and starthin 25 gram...",
    "diseased cotton leaf": "Solution for diseased cotton leaf...",
    "diseased cotton plant": "Solution for diseased cotton plant...",
    "fassuriam wilt": "copper oxidoride 50 gram and camicacid 30 ml for plant 250 ml ...",
    "fresh cotton leaf": "Solution for fresh cotton leaf...",
    "fresh cotton plant": "Solution for fresh cotton plant...",
    "target spot": "amitza tap 20 ml and mizine 5 gram or caprio top 60 gram..."
}

# Function to preprocess the image for prediction
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)
    return img_array

# Streamlit app
st.title("Cotton Leaf Disease Classification")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img_array = preprocess_image(image)

    # Make prediction using the loaded model
    predictions = loaded_model.predict(img_array)

    # Find the index of the class with the highest probability
    max_prob_index = np.argmax(predictions)

    # Display the class with the highest probability
    st.subheader("Prediction:")
    predicted_class = class_names[max_prob_index]
    st.write(f"The predicted class is: {predicted_class}")

    # Display the solution for the predicted class
    st.subheader("Solution:")
solution_lines = solutions[predicted_class].split(" and")
formatted_solution = "\n".join(solution_lines)
st.write(formatted_solution)

    
