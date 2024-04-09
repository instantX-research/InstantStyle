import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image

from ip_adapter import IPAdapterXL

base_model_path = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

# load SDXL pipeline
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

image = "./assets/5.jpg"
image = Image.open(image)
image.resize((512, 512))

init_image = Image.open("./assets/overture-creations-5sI6fQgYIuo.png").convert("RGB")
mask_image = Image.open("./assets/overture-creations-5sI6fQgYIuo_mask_inverse.png").convert("RGB")

# generate image
images = ip_model.generate(pil_image=image,
                            prompt="a dog sitting on, masterpiece, best quality, high quality",
                            negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                            scale=2.0,
                            guidance_scale=8,
                            num_samples=1,
                            num_inference_steps=30, 
                            image=init_image,
                            mask_image=mask_image,
                            strength=0.99
                            )

images[0].save("result.png")