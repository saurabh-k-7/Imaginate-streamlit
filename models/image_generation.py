# # image_generation.py
# from diffusers import StableDiffusionPipeline
# import torch
# from PIL import Image

# class CFG:
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     seed = 42
#     generator = torch.Generator(device).manual_seed(seed)
#     image_gen_steps = 45
#     image_gen_model_id = "stabilityai/stable-diffusion-2"
#     image_gen_size = (1920, 1080)

# # Load Stable Diffusion model
# image_gen_model = StableDiffusionPipeline.from_pretrained(
#     CFG.image_gen_model_id, torch_dtype=torch.float16, revision="fp16", use_auth_token='YOUR_HUGGINGFACE_TOKEN', guidance_scale=9
# )
# image_gen_model = image_gen_model.to(CFG.device)

# def generate_image(prompt: str):
#     """
#     Function to generate image from the given prompt using Stable Diffusion.
#     Args:
#         prompt (str): Text prompt for image generation.
    
#     Returns:
#         PIL.Image: Generated image
#     """
#     output = image_gen_model(prompt, num_inference_steps=CFG.image_gen_steps, generator=CFG.generator)
#     generated_images = output.images
#     if generated_images:
#         image = generated_images[0]
#         image = image.resize(CFG.image_gen_size)
#         return image
#     else:
#         return None


from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os

def generate_image(prompt):
    repo_id = "stabilityai/stable-diffusion-2"
    
    # Load the pipe with CPU-specific settings
    pipe = DiffusionPipeline.from_pretrained(
        repo_id, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    # Set the scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Force CPU usage
    pipe = pipe.to("cpu")

    # Optimization: Chunk the math so it fits in 4GB RAM
    pipe.enable_attention_slicing()

    # Generate the image
    # We reduce num_inference_steps slightly (to 20) to speed up the CPU wait time
    result = pipe(
        prompt, 
        guidance_scale=9, 
        num_inference_steps=20
    ).images[0]
    
    return result