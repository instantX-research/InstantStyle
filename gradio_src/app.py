import os 
donwload_repo_loc= "./models/image_encoder/"
os.system("pip install -U peft")
# os.system(f"wget -O {donwload_repo_loc}config.json https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/config.json?download=true")
# os.system(f"wget -O {donwload_repo_loc}model.safetensors https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/model.safetensors?download=true")
# os.system(f"wget -O {donwload_repo_loc}pytorch_model.bin https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/pytorch_model.bin?download=true")

import spaces
import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from ip_adapter import IPAdapterXL
base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
device = "cuda"

image_encoder_path = donwload_repo_loc #"sdxl_models/image_encoder"
ip_ckpt = "./models/ip-adapter_sdxl.bin"
# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)


# generate image variations with only image prompt
@spaces.GPU(enable_queue=True)
def create_image(image_pil,target,prompt,n_prompt,scale, guidance_scale,num_samples,num_inference_steps,seed):
    # load ip-adapter
    if target =="Load original IP-Adapter":
        # target_blocks=["blocks"] for original IP-Adapter
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["blocks"])
    elif target=="Load only style blocks":
        # target_blocks=["up_blocks.0.attentions.1"] for style blocks only
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])
    elif target == "Load style+layout block":
        # target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"])
    

    image_pil=image_pil.resize((512, 512))
    images = ip_model.generate(pil_image=image_pil,
                            prompt=prompt,
                            negative_prompt=n_prompt,
                            scale=scale,
                            guidance_scale=guidance_scale,
                            num_samples=num_samples,
                            num_inference_steps=num_inference_steps, 
                            seed=seed,
                            #neg_content_prompt="a rabbit",
                            #neg_content_scale=0.5,
                            )

    # images[0].save("result.png")    
    del ip_model
    
    return images


DESCRIPTION = """
# InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation
**Demo by [ameer azam] - [Twitter](https://twitter.com/Ameerazam18) - [GitHub](https://github.com/AMEERAZAM08)) - [Hugging Face](https://huggingface.co/ameerazam08)**
This is a demo of  https://github.com/InstantStyle/InstantStyle.
"""

block = gr.Blocks(css="footer {visibility: hidden}").queue()
with block:
    with gr.Row():
       
        with gr.Column():
            # gr.Markdown("## <h1 align='center'>InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation  </h1>")
            gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.Row():
            with gr.Column():
                image_pil = gr.Image(label="Style Image", type='pil')
                target = gr.Dropdown(["Load original IP-Adapter","Load only style blocks","Load style+layout block"], label="LORA Model", info="Which finetuned model to use?")
                prompt = gr.Textbox(label="Prompt",value="a cat, masterpiece, best quality, high quality")
                n_prompt = gr.Textbox(label="Neg Prompt",value="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry")
                scale = gr.Slider(minimum=0,maximum=2.0, step=0.01,value=1.0, label="scale")
                guidance_scale = gr.Slider(minimum=1,maximum=15.0, step=0.01,value=5.0, label="guidance_scale")
                num_samples= gr.Slider(minimum=1,maximum=3.0, step=1.0,value=1.0, label="num_samples")
                num_inference_steps = gr.Slider(minimum=5,maximum=50.0, step=1.0,value=30, label="num_inference_steps")
                seed = gr.Slider(minimum=-1000000,maximum=1000000,value=1, step=1, label="Seed Value")
                generate_button = gr.Button("Generate Image")
            with gr.Column():
                generated_image = gr.Gallery(label="Generated Image")

        generate_button.click(fn=create_image, inputs=[image_pil,target,prompt,n_prompt,scale, guidance_scale,num_samples,num_inference_steps,seed], 
                                  outputs=[generated_image])

block.launch(max_threads=10)
