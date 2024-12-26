import os
import numpy as np
import tensorflow as tf
import cv2

# Load pre-trained model
model_path = r"training\models1\imageclassifier1.h5"
model = tf.keras.models.load_model(model_path)



def preprocess_image(image_path):
    """
    Preprocess the input image: load, convert to RGB, resize, and normalize.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or failed to load.")
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize the image
        resized_img = tf.image.resize(img_rgb, (256, 256))
        
        # Normalize the image
        normalized_img = resized_img / 255.0
        
        return normalized_img
    except Exception as e:
        raise RuntimeError(f"Error preprocessing the image: {e}")

def classify_image(image_path):
    """
    Classify the image using the pre-trained model.
    Returns: Prediction result as a string.
    """
    try:
        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)
        
        # Expand dimensions for model prediction
        image_input = np.expand_dims(preprocessed_image, axis=0)  # Define image_input here
        
        # Predict with the model
        yhat = model.predict(image_input)[0][0]
        
        confidence = yhat * 100 if yhat > 0.5 else (1 - yhat) * 100
        label = "Sad" if yhat > 0.5 else "Happy"
        return f"{label} ({confidence:.2f}% confident)"

    except Exception as e:
        return f"Error during classification: {e}"
