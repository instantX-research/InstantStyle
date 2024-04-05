from diffusers import StableDiffusionXLPipeline
from PIL import Image
import torch

def prepare_ip_adapter_image_embeds(pipeline, ip_adapter_image, neg_content_prompt, neg_content_scale,
                                    device, num_images_per_prompt):

    if not isinstance(ip_adapter_image, list):
        ip_adapter_image = [ip_adapter_image]

    if len(ip_adapter_image) != len(pipeline.unet.encoder_hid_proj.image_projection_layers):
        raise ValueError(
            f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {len(pipeline.unet.encoder_hid_proj.image_projection_layers)} IP Adapters."
        )
    
    with torch.inference_mode():
        (
            prompt_embeds_,
            negative_prompt_embeds_,
            pooled_prompt_embeds_,
            negative_pooled_prompt_embeds_,
        ) = pipeline.encode_prompt(
            neg_content_prompt,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=True,
            negative_prompt=" ",
        )
    
    image_embeds = []
    for single_ip_adapter_image, image_proj_layer in zip(
        ip_adapter_image, pipeline.unet.encoder_hid_proj.image_projection_layers
    ):
        output_hidden_state = None
        single_image_embeds, single_negative_image_embeds = pipeline.encode_image(
            single_ip_adapter_image, device, 1, output_hidden_state
        )
        single_image_embeds = single_image_embeds - pooled_prompt_embeds_ * neg_content_scale
        single_image_embeds = torch.stack([single_image_embeds] * num_images_per_prompt, dim=0)
        single_negative_image_embeds = torch.stack(
            [single_negative_image_embeds] * num_images_per_prompt, dim=0
        )

        single_image_embeds = torch.cat([single_negative_image_embeds, single_image_embeds])
        single_image_embeds = single_image_embeds.to(device)

        image_embeds.append(single_image_embeds)

    return image_embeds

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
).to(device)

# set target blocks
# target_blocks=["block"] for original IP-Adapter,
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
pipe.load_ip_adapter(pretrained_model_name_or_path_or_dict="./", 
                     subfolder="sdxl_models", 
                     weight_name="ip-adapter_sdxl.bin",
                     image_encoder_folder=image_encoder_path,
                     target_blocks=["block"]
                    )
pipe.set_ip_adapter_scale(1.0)

image = "./assets/0.jpg"
image = Image.open(image)
image.resize((512, 512))

ip_adapter_image_embeds = prepare_ip_adapter_image_embeds(pipeline=pipe, 
                                                          ip_adapter_image=image, 
                                                          neg_content_prompt="a rabbit",
                                                          neg_content_scale=0.5,
                                                          device=pipe.device, 
                                                          num_images_per_prompt=1)

# generate image
images = pipe(
    prompt="a cat, masterpiece, best quality, high quality",
    ip_adapter_image_embeds=ip_adapter_image_embeds,
    negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
    num_inference_steps=30,
    guidance_scale=5,
).images                            

images[0].save("result.png")