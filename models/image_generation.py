# image_generation.py
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 45
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (1920, 1080)

# Load Stable Diffusion model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token='YOUR_HUGGINGFACE_TOKEN', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt: str):
    """
    Function to generate image from the given prompt using Stable Diffusion.
    Args:
        prompt (str): Text prompt for image generation.
    
    Returns:
        PIL.Image: Generated image
    """
    output = image_gen_model(prompt, num_inference_steps=CFG.image_gen_steps, generator=CFG.generator)
    generated_images = output.images
    if generated_images:
        image = generated_images[0]
        image = image.resize(CFG.image_gen_size)
        return image
    else:
        return None
