from diffusers.utils import load_image
import datetime
import numpy as np
from scipy.ndimage import zoom
import torch
from PIL import Image
from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionDiffEditPipeline

pipeline = StableDiffusionDiffEditPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    use_safetensors=False,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config)
pipeline.enable_vae_slicing()

inputImg = "Input.jpg"
raw_image = load_image(inputImg).convert("RGB").resize((350, 350))

source_prompt = "Taylor Swift"
target_prompt = "Scarlett Johansson"
mask_image = pipeline.generate_mask(
    image=raw_image,
    source_prompt=source_prompt,
    target_prompt=target_prompt,
)

def SaveMask(mask):
    mask = torch.from_numpy(mask).permute(1, 2, 0).numpy()
    mask = (mask * 255).round().astype("uint8")
    mask = np.squeeze(mask, axis=-1)
    Image.fromarray(mask).save("Mask.jpg", format="JPEG")

SaveMask(mask_image)

inv_latents = pipeline.invert(prompt=source_prompt, image=raw_image).latents

image = pipeline(
    prompt="Scarlett Johansson",
    mask_image=mask_image,
    image_latents=inv_latents,
    negative_prompt="Taylor Swift",
).images[0]
image.save("edited_image.jpg")