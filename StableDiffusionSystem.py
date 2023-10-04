from dataclasses import dataclass
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, AutoencoderKL, UNet2DConditionModel, PNDMScheduler
from PIL import Image

@dataclass
class StableDiffusionConfig:
    device = "cpu"
    cache_dir  = "./.cache"   #path for caching
    guidance_scale = 7.5  # Scale for classifier-free guidance

class StableDiffusionSystem:
    def __init__(self, config : StableDiffusionConfig):
        self.config = config
        self.generator = torch.manual_seed(0)
        self.model_name = ""
        self.sdpipeline = None
        self.inpainting_pipeline = None

    def set_seed(self, seed):
        self.generator.manual_seed(seed)

    def load_components(self):
        self.vae = AutoencoderKL.from_pretrained(self.model_name, subfolder="vae", cache_dir = self.config.cache_dir).to(self.config.device)
        self.vae.enable_tiling()
        self.vae.enable_slicing()
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_name, subfolder="text_encoder", cache_dir = self.config.cache_dir).to(self.config.device)
        self.unet = UNet2DConditionModel.from_pretrained(self.model_name, subfolder="unet", cache_dir = self.config.cache_dir).to(self.config.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name, subfolder="tokenizer", cache_dir = self.config.cache_dir)
        self.scheduler = PNDMScheduler.from_pretrained(self.model_name, subfolder="scheduler", cache_dir = self.config.cache_dir)

    def textToImage(self, model_name, prompt, negative_prompt, steps):
        if model_name != self.model_name:
            self.model_name = model_name
            self.load_components()
            self.sdpipeline = StableDiffusionPipeline(
            vae = self.vae,
            text_encoder = self.text_encoder,
            tokenizer = self.tokenizer,
            unet = self.unet,
            scheduler = self.scheduler,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False,
        )
            self.sdpipeline.enable_attention_slicing()
        return self.sdpipeline(
            prompt = prompt,
            negative_prompt = negative_prompt,
            num_inference_steps = steps,
            guidance_scale = self.config.guidance_scale,
            generator = self.generator,
        ).images

    def inpainting(self, model_name, prompt, negative_prompt, input_img, mask_img, steps):
        if model_name != self.model_name:
            self.model_name = model_name
            self.load_components()
            self.inpainting_pipeline = StableDiffusionInpaintPipeline(
            vae = self.vae,
            text_encoder = self.text_encoder,
            tokenizer = self.tokenizer,
            unet = self.unet,
            scheduler = self.scheduler,
            safety_checker = None,
            feature_extractor = None,
            requires_safety_checker = False,
            )
        self.inpainting_pipeline.enable_attention_slicing()

        input_img = Image.fromarray(input_img)
        input_img.resize((256, 256))
        mask_img = Image.fromarray(mask_img)
        mask_img.resize((128, 128))

        return self.inpainting_pipeline(
            prompt = prompt,
            negative_prompt = negative_prompt,
            image = input_img,
            mask_image = mask_img,
            num_inference_steps = steps,
            guidance_scale = self.config.guidance_scale,
            generator = self.generator,
        ).images
    
    
