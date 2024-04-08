import os 
os.system("pip install -U peft")

import spaces
import gradio as gr
import torch
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
import cv2
from PIL import Image
import numpy as np
from ip_adapter import IPAdapterXL

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"
controlnet_path = "diffusers/controlnet-canny-sdxl-1.0"
controlnet = ControlNetModel.from_pretrained(controlnet_path, use_safetensors=False, torch_dtype=torch.float16).to(device)

# load SDXL pipeline
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    add_watermarker=False,
)


# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

@spaces.GPU(enable_queue=True)
def create_image(image_pil,input_image,prompt,n_prompt,scale, control_scale, guidance_scale,num_samples,num_inference_steps,seed):

    image_pil=image_pil.resize((512, 512))

    cv_input_image = pil_to_cv2(input_image)
    detected_map = cv2.Canny(cv_input_image, 50, 200)
    canny_map = Image.fromarray(cv2.cvtColor(detected_map, cv2.COLOR_BGR2RGB))

    images = ip_model.generate(pil_image=image_pil,
                            prompt=prompt,
                            negative_prompt=n_prompt,
                            scale=scale,
                            guidance_scale=guidance_scale,
                            num_samples=num_samples,
                            num_inference_steps=num_inference_steps, 
                            seed=seed,
                            image=canny_map,
                            controlnet_conditioning_scale=control_scale,
                            )
    return images

def pil_to_cv2(image_pil):
    image_np = np.array(image_pil)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    return image_cv2

DESCRIPTION = """
# InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation
**Image Stylization Demo by [vinesmsuic] - [Twitter](https://twitter.com/vinesmsuic) - [GitHub](https://github.com/vinesmsuic)) - [Hugging Face](https://huggingface.co/vinesmsuic)**
This is a Image Stylization demo of  https://github.com/InstantStyle/InstantStyle.
"""

block = gr.Blocks(css="footer {visibility: hidden}").queue(max_size=10)
with block:
    with gr.Row():
       
        with gr.Column():
            # gr.Markdown("## <h1 align='center'>InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation  </h1>")
            gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    with gr.Column():
                        src_image_pil = gr.Image(label="Source Image", type='pil')
                    with gr.Column():
                        image_pil = gr.Image(label="Style Image", type='pil')
                prompt = gr.Textbox(label="Prompt",value="masterpiece, best quality, high quality")
                n_prompt = gr.Textbox(label="Neg Prompt",value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry")
                scale = gr.Slider(minimum=0,maximum=2.0, step=0.01,value=1.0, label="scale")
                control_scale = gr.Slider(minimum=0,maximum=1.0, step=0.01,value=0.6, label="controlnet conditioning scale")
                guidance_scale = gr.Slider(minimum=1,maximum=15.0, step=0.01,value=5.0, label="guidance scale")
                num_samples= gr.Slider(minimum=1,maximum=3.0, step=1.0,value=1.0, label="num samples")
                num_inference_steps = gr.Slider(minimum=5,maximum=50.0, step=1.0,value=30, label="num inference steps")
                seed = gr.Slider(minimum=-1000000,maximum=1000000,value=1, step=1, label="Seed Value")
                generate_button = gr.Button("Generate Image")
            with gr.Column():
                generated_image = gr.Gallery(label="Generated Image")

        generate_button.click(fn=create_image, 
                            inputs=[image_pil,src_image_pil,prompt,n_prompt,scale, control_scale, guidance_scale,num_samples,num_inference_steps,seed], 
                            outputs=[generated_image])

block.launch()
