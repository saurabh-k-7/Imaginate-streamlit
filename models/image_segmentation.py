from transformers import pipeline
import numpy as np

# Load a segmentation model
image_segmenter = pipeline("image-segmentation", model="facebook/detr-resnet-50-panoptic")

def segmentify_image(image):
    """
    Segmentifies an image using a pre-trained segmentation model.
    
    Args:
        image (PIL.Image.Image): The image for segmentation.
    
    Returns:
        List[Dict]: A list of segmentation results (labels, scores, etc.).
    """
    if image is None:
        raise ValueError("Image cannot be empty.")

    # Ensure the image is a PIL image
    from PIL import Image
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    # Use the segmentation pipeline
    result = image_segmenter(image)
    if result:
        return result
    else:
        raise RuntimeError("Segmentation failed. No results returned.")
