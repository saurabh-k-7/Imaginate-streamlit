import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import gc #garabage collection for memory management for generation section
from io import BytesIO  # Required for converting images for download

# Importing functions from models
from models.image_generation import generate_image  # Image generation
from models.image_classification import classify_image  # Image classification 
from models.image_segmentation import segmentify_image  # Image segmentation 


# Sidebar for user interaction
st.sidebar.title("🎨 App Navigation")
option = st.sidebar.selectbox(
    "Choose an Action:",
    ("✨ Generate Image", "🙂 Emotion Detection (Happy/Sad)", "🖼️ Image Segmentation"),
)

# Styling for headers
st.markdown(
    """
    <style>
    .title-style {
        font-size: 50px;
        font-weight: bold;
        color: #FFFFFF;
        text-shadow: 2px 2px 5px #000000;
    }
    .subtitle-style {
        font-size: 20px;
        color: #D3D3D3;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Ensure the `temp` directory exists
temp_dir = "temp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Image Generation Section
# Image Generation Section
if option == "✨ Generate Image":
    st.markdown('<h1 class="title-style">Generate Stunning Images</h1>', unsafe_allow_html=True)

    # 1. Input for prompt
    prompt = st.text_input(
        "💡 Enter your creative prompt:",
        "A scenic view of the mountains at sunrise",
    )

    # 2. The Button and Spinner logic
if st.button("🎨 Generate Image"):
    # We updated the message here to warn the user about the CPU speed
    with st.spinner("✨ Generating... This takes 2-4 minutes on EC2 CPU. Please wait!"):
        try:
            generated_image = generate_image(prompt)  # This calls your CPU-optimized function
                
            if generated_image:
                st.success("🎉 Image generated successfully!")
                st.image(generated_image, caption="Generated Image", use_container_width=True)

                # 3. Preparation for Download
                buffered = BytesIO()
                generated_image.save(buffered, format="PNG")
                buffered.seek(0)

                st.download_button(
                    label="💾 Download Image",
                    data=buffered,
                    file_name="generated_image.png",
                    mime="image/png",
                )
        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")
    gc.collect()

# Image Classification Section
elif option == "🙂 Emotion Detection (Happy/Sad)":
    st.markdown('<h1 class="title-style">Classify Your Images</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle-style">Discover the emotions in your photos!</h3>', unsafe_allow_html=True)

    # Upload image file
    uploaded_file = st.file_uploader(
        "📤 Upload an image to classify:",
        type=["jpg", "jpeg", "png", "bmp"],
    )

    if uploaded_file:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Save the uploaded file temporarily
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Classify the image
        if st.button("🔍 Classify Image"):
            with st.spinner("🤔 Analyzing the image... please wait!"):
                result = classify_image(temp_path)  # Call the classification function
                st.success(f"Prediction: {result}")

        # Clean up the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Image Segmentation Section
elif option == "🖼️ Image Segmentation":
    st.markdown('<h1 class="title-style">Segmentify Your Images</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle-style">Extract meaningful segments from your photos!</h3>', unsafe_allow_html=True)

    # Upload image file
    uploaded_file = st.file_uploader(
        "📤 Upload an image for segmentation:",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file:
        try:
            # Open the uploaded file as a PIL image
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            # Button to perform segmentation
            if st.button("📊 Perform Segmentation"):
                with st.spinner("✨ Segmenting the image... please wait!"):
                    # Get segmentation results
                    result = segmentify_image(image)

                    # Display segmentation results
                    st.success("Segmentation complete!")
                    st.write("Detected objects and regions:")
                    for obj in result:
                        st.write(f"- **{obj['label']}** with score {obj['score']:.2f}")

        except Exception as e:
            st.error(f"⚠️ Error occurred during segmentation: {str(e)}")



