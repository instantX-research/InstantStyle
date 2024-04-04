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

## Release

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

## Usage

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

For any question, please feel free to contact us via <haofanwang.ai@gmail.com>.
