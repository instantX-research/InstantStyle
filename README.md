<div align="center">
<h1>InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation</h1>

[**Haofan Wang**](https://haofanwang.github.io/)<sup>*</sup> · [**Qixun Wang**](https://github.com/wangqixun) · [**Xu Bai**](https://huggingface.co/baymin0220) · [**Zekui Qin**](https://github.com/ZekuiQin) · [**Anthony Chen**](https://antonioo-c.github.io/)

InstantX Team 

<sup>*</sup>corresponding authors

<a href='[https://instantid.github.io/](https://instantstyle.github.io/)'><img src='https://img.shields.io/badge/Project-Page-green'></a>
<a href='https://arxiv.org/abs/2404.02733'><img src='https://img.shields.io/badge/Technique-Report-red'></a>
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
- [2024/04/03] 🔥 We release the [technical report](https://arxiv.org/abs/2404.02733).

## Download
Follow [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter?tab=readme-ov-file#download-models) to download pre-trained checkpoints.

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

## Usage

Our method is fully compatible with [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter). But for feature subtraction, it only works with IP-Adapter using global embeddings.

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

# load ip-adapter
# target_blocks=["blocks"] for original IP-Adapter
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

We will support diffusers API soon.

## TODO
- Support in diffusers API.
- Support InstantID.

## Sponsor Us
If you find this project useful, you can buy us a coffee via Github Sponsor! We support [Paypal](https://ko-fi.com/instantx) and [WeChat Pay](https://tinyurl.com/instantx-pay).

## Cite
If you find InstantStyle useful for your research and applications, please cite us using this BibTeX:

```bibtex
@misc{wang2024instantstyle,
      title={InstantStyle: Free Lunch towards Style-Preserving in Text-to-Image Generation}, 
      author={Haofan Wang and Qixun Wang and Xu Bai and Zekui Qin and Anthony Chen},
      year={2024},
      eprint={2404.02733},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

For any question, please feel free to contact us via haofanwang.ai@gmail.com.
