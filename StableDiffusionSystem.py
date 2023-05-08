from dataclasses import dataclass

from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from tqdm.auto import tqdm

@dataclass
class StableDiffusionConfig:
    device = "cpu"
    model_name = "runwayml/stable-diffusion-v1-5"  #name of model used for inference
    cache_dir  = "./.cache"   #path for caching
    num_inference_steps = 25  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    seed = 0

class StableDiffusionSystem:
    def __init__(self, config : StableDiffusionConfig):
        self.config = config
        self.device = torch.device(self.config.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.model_name, subfolder="tokenizer", cache_dir = self.config.cache_dir)
        self.text_encoder = CLIPTextModel.from_pretrained(self.config.model_name, subfolder="text_encoder", cache_dir = self.config.cache_dir).to(self.device)
        self.vae = AutoencoderKL.from_pretrained(self.config.model_name, subfolder="vae", cache_dir = self.config.cache_dir).to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(self.config.model_name, subfolder="unet", cache_dir = self.config.cache_dir).to(self.device)
        self.scheduler = PNDMScheduler.from_pretrained(self.config.model_name, subfolder="scheduler")
        self.generator = torch.manual_seed(self.config.seed)

    def textToImage(self, prompt):
        batch_size = len(prompt)
        #Prepare text embeddings
        text_input = self.tokenizer(prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        #Prepare unconditional text embeddings
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        #concatenate the conditional and unconditional embeddings into a batch to avoid doing two forward passes
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        #Generate random noise
        latents = torch.randn((batch_size, self.unet.config.in_channels, self.unet.sample_size , self.unet.sample_size), generator=self.generator).to(self.device)

        #denoising loop
        self.scheduler.set_timesteps(self.config.num_inference_steps)
        for t in tqdm(self.scheduler.timesteps):
            # expand the latents to match concatenated text embeddings
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep = t)
            # predict the noise residual
            with torch.no_grad():
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_text - noise_pred_uncond)
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # scale and decode the image latents with vae
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample 
        return image
    
def ToPILImage(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images
