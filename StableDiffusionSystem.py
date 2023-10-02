from dataclasses import dataclass
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, PNDMScheduler

@dataclass
class StableDiffusionConfig:
    device = "cpu"
    model_name = "runwayml/stable-diffusion-v1-5"  #name of model used for inference
    cache_dir  = "./.cache"   #path for caching
    num_inference_steps = 25  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    seed = 0
    save_memory = True

class StableDiffusionSystem:
    def __init__(self, config : StableDiffusionConfig):
        self.config = config
        self.generator = torch.manual_seed(self.config.seed)
        self.sdpipeline = StableDiffusionPipeline(
            vae = AutoencoderKL.from_pretrained(self.config.model_name, subfolder="vae", cache_dir = self.config.cache_dir).to(self.config.device),
            text_encoder = CLIPTextModel.from_pretrained(self.config.model_name, subfolder="text_encoder", cache_dir = self.config.cache_dir).to(self.config.device),
            tokenizer = CLIPTokenizer.from_pretrained(self.config.model_name, subfolder="tokenizer", cache_dir = self.config.cache_dir),
            unet = UNet2DConditionModel.from_pretrained(self.config.model_name, subfolder="unet", cache_dir = self.config.cache_dir).to(self.config.device),
            scheduler = PNDMScheduler.from_pretrained(self.config.model_name, subfolder="scheduler"),
            safety_checker = None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        if self.config.save_memory:
            self.sdpipeline.enable_attention_slicing()
            self.sdpipeline.enable_vae_slicing()
            self.sdpipeline.enable_vae_tiling()
            if self.config.device == "cuda":
                self.sdpipeline.enable_xformers_memory_efficient_attention()
        
    def set_seed(self, seed):
        self.generator.manual_seed(seed)

    def textToImage(self, prompt, negative_prompt, steps):
        return self.sdpipeline(
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps = steps,
            guidance_scale = self.config.guidance_scale,
            generator = self.generator,
        ).images
    
    
