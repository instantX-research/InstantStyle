<div align="center">
<h1>InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation</h1>

[**Haofan Wang**](https://haofanwang.github.io/)<sup>*</sup> · [**Matteo Spinelli**](https://github.com/cubiq) · [**Qixun Wang**](https://github.com/wangqixun) · [**Xu Bai**](https://huggingface.co/baymin0220) · [**Zekui Qin**](https://github.com/ZekuiQin) · [**Anthony Chen**](https://antonioo-c.github.io/)

InstantX Team 

<sup>*</sup>corresponding authors

<a href='https://instantstyle.github.io/'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2404.02733'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-red)](https://huggingface.co/spaces/InstantX/InstantStyle)
[![ModelScope](https://img.shields.io/badge/ModelScope-Studios-blue)](https://modelscope.cn/studios/instantx/InstantID/summary)
[![GitHub](https://img.shields.io/github/stars/InstantStyle/InstantStyle?style=social)](https://github.com/InstantStyle/InstantStyle)

</div>

InstantStyle is a general framework that employs two straightforward yet potent techniques for achieving an effective disentanglement of style and content from reference images.

<img src='assets/pipe.png'>

## Principle

Separating Content from Image. Benefit from the good characterization of CLIP global features, after subtracting the content text fea- tures from the image features, the style and content can be explicitly decoupled. Although simple, this strategy is quite effective in mitigating content leakage.
<p align="center">
  <img src="assets/subtraction.png">
</p>

Injecting into Style Blocks Only. Empirically, each layer of a deep network captures different semantic information the key observation in our work is that there exists two specific attention layers handling style. Specifically, we find up blocks.0.attentions.1 and down blocks.2.attentions.1 capture style (color, material, atmosphere) and spatial layout (structure, composition) respectively.
<p align="center">
  <img src="assets/tree.png">
</p>

## Release
- [2024/04/11] 🔥 We add the experimental distributed inference feature. Check it [here](https://github.com/InstantStyle/InstantStyle?tab=readme-ov-file#distributed-inference).
- [2024/04/10] 🔥 We support an [online demo](https://modelscope.cn/studios/instantx/InstantStyle/summary) on ModelScope.
- [2024/04/09] 🔥 We support an [online demo](https://huggingface.co/spaces/InstantX/InstantStyle) on Huggingface.
- [2024/04/09] 🔥 We support SDXL-inpainting, more information can be found [here](https://github.com/InstantStyle/InstantStyle/blob/main/infer_style_inpainting.py).
- [2024/04/08] 🔥 InstantStyle is supported in [AnyV2V](https://tiger-ai-lab.github.io/AnyV2V/) for stylized video-to-video editing, demo can be found [here](https://twitter.com/vinesmsuic/status/1777170927500787782).
- [2024/04/07] 🔥 We support image-based stylization, more information can be found [here](https://github.com/InstantStyle/InstantStyle/blob/main/infer_style_controlnet.py).
- [2024/04/07] 🔥 We support an experimental version for SD1.5, more information can be found [here](https://github.com/InstantStyle/InstantStyle/blob/main/infer_style_sd15.py).
- [2024/04/03] 🔥 InstantStyle is supported in [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) developed by our co-author.
- [2024/04/03] 🔥 We release the [technical report](https://arxiv.org/abs/2404.02733).

## Demos

### Stylized Synthesis

<p align="center">
  <img src="assets/example1.png">
  <img src="assets/example2.png">
</p>

### Image-based Stylized Synthesis

<p align="center">
  <img src="assets/example3.png">
</p>

### Comparison with Previous Works

<p align="center">
  <img src="assets/comparison.png">
</p>

## Download
Follow [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter?tab=readme-ov-file#download-models) to download pre-trained checkpoints from [here](https://huggingface.co/h94/IP-Adapter).

```
git clone https://github.com/InstantStyle/InstantStyle.git
cd InstantStyle

# download the models
git lfs install
git clone https://huggingface.co/h94/IP-Adapter
mv IP-Adapter/models models
mv IP-Adapter/sdxl_models sdxl_models
```

## Usage

Our method is fully compatible with [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter). For feature subtraction, it only works for global feature instead of patch features. For SD1.5, you can find a demo at [infer_style_sd15.py](https://github.com/InstantStyle/InstantStyle/blob/main/infer_style_sd15.py), but we find that SD1.5 has weaker perception and understanding of style information, thus this demo is experimental only. All block names can be found in [attn_blocks.py](https://github.com/InstantStyle/InstantStyle/blob/main/attn_blocks.py) and [attn_blocks_sd15.py](https://github.com/InstantStyle/InstantStyle/blob/main/attn_blocks_sd15.py) for SDXL and SD1.5 respectively.

```python
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

from ip_adapter import IPAdapterXL

base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
image_encoder_path = "sdxl_models/image_encoder"
ip_ckpt = "sdxl_models/ip-adapter_sdxl.bin"
device = "cuda"

# load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)

# reduce memory consumption
pipe.enable_vae_tiling()

# load ip-adapter
# target_blocks=["block"] for original IP-Adapter
# target_blocks=["up_blocks.0.attentions.1"] for style blocks only
# target_blocks = ["up_blocks.0.attentions.1", "down_blocks.2.attentions.1"] # for style+layout blocks
ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device, target_blocks=["up_blocks.0.attentions.1"])

image = "./assets/0.jpg"
image = Image.open(image)
image.resize((512, 512))

# generate image variations with only image prompt
images = ip_model.generate(pil_image=image,
                            prompt="a cat, masterpiece, best quality, high quality",
                            negative_prompt= "text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
                            scale=1.0,
                            guidance_scale=5,
                            num_samples=1,
                            num_inference_steps=30, 
                            seed=42,
                            #neg_content_prompt="a rabbit",
                            #neg_content_scale=0.5,
                          )

images[0].save("result.png")
```

## Distributed Inference
On distributed setups, you can run inference across multiple GPUs with 🤗 Accelerate or PyTorch Distributed, which is useful for generating with multiple prompts in parallel, in case you have limited VRAM on each GPU. More information can be found [here](https://huggingface.co/docs/diffusers/main/en/training/distributed_inference#device-placement). Make sure you have installed diffusers from the source and the lastest accelerate.

```
max_memory = {0:"10GB", 1:"10GB"}
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
    device_map="balanced",
    max_memory=max_memory
)
```

## Start a local gradio demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>
Run the following command:
```
git clone https://github.com/InstantStyle/InstantStyle.git
cd ./InstantStyle/gradio_demo/
pip install -r requirements.txt
python app.py
```

## Resources
- [InstantStyle for ComfyUI](https://github.com/cubiq/ComfyUI_IPAdapter_plus)
- [InstantID](https://github.com/InstantID/InstantID)

## TODO
- Support in diffusers API, check our [PR](https://github.com/huggingface/diffusers/pull/7586).
- Support InstantID for face stylization once stars reach 1K.

## Disclaimer
Our released codes and checkpoints are for non-commercial research purposes only. Users are granted the freedom to create images using this tool, but they are obligated to comply with local laws and utilize it responsibly. The developers will not assume any responsibility for potential misuse by users.

## Acknowledgements
InstantStyle is developed by the InstantX team and is highly built on [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), which has been unfairly compared by many other works. We at InstantStyle make IP-Adapter great again. Additionally, we acknowledge [Hu Ye](https://github.com/xiaohu2015) for his valuable discussion.

## Star History
[![Star History Chart](https://api.star-history.com/svg?repos=InstantStyle/InstantStyle&type=Date)](https://star-history.com/#InstantStyle/InstantStyle&Date)

## Cite
If you find InstantStyle useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{wang2024instantstyle,
  title={InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation},
  author={Wang, Haofan and Wang, Qixun and Bai, Xu and Qin, Zekui and Chen, Anthony},
  journal={arXiv preprint arXiv:2404.02733},
  year={2024}
}
```

For any question, feel free to contact us via haofanwang.ai@gmail.com.
