import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image

from ip_adapter import IPAdapter

base_model_path = "runwayml/stable-diffusion-v1-5"
image_encoder_path = "models/image_encoder"
ip_ckpt = "models/ip-adapter_sd15.bin"
device = "cuda"

# load SDXL pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.1"] for style blocks only (experimental, not obvious as SDXL)
# target_blocks = ["down_blocks.2", "mid_block", "up_blocks.1"] # for style+layout blocks (experimental, not obvious as SDXL)
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["block"])

image = "./assets/3.jpg"
image = Image.open(image)
image.resize((512, 512))

# set negative content
neg_content = "a girl"
neg_content_scale = 0.8
if neg_content is not None:
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    text_encoder = CLIPTextModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K").to(pipe.device, 
                                                                                                           dtype=pipe.dtype)
    tokenizer = CLIPTokenizer.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")

    tokens = tokenizer([neg_content], return_tensors='pt').to(pipe.device)
    neg_content_emb = text_encoder(**tokens).text_embeds
    neg_content_emb *= neg_content_scale
else:
    neg_content_emb = None

# generate image with content subtraction
images = ip_model.generate(pil_image=image,
                           prompt="a cat, masterpiece, best quality, high quality",
                           negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                           scale=1.0,
                           guidance_scale=5,
                           num_samples=1,
                           num_inference_steps=30, 
                           seed=42,
                           neg_content_emb=neg_content_emb,
                          )

images[0].save("result.png")