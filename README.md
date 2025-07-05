# PathOrchestra (V1.0.0)

<p align="center">
    <img src="./figures/PathOrchestra.png" width="400"/>
<p>
  
<p align="center">
  <a href="https://arxiv.org/abs/2503.24345">📑Article Link</a> |
  <a href="#model-weights">🤗 Download Models</a> |
  <a href="#pre-extracted-embeddings">🤗 Download Pre-extracted Embeddings</a> |
  <a href="#reference">📑 Cite</a>
</p>

## Introduction
The official Repo for *Arixv* 2025 Paper [**PathOrchestra: A Comprehensive Foundation Model for Computational Pathology with Over 100 Diverse Clinical-Grade Tasks**](https://arxiv.org/abs/2503.24345)

### Updates
- 3-31-2025: Article Online
- 3-02-2025: Model Weights (V1.0.0) Online

## Model weights
| Model Name    | Release Date | Model Architecture | Download Link            |
|---------------------|--------------|---------------------|-------------------------------------------------------------|
| PathOrchestra_V1.0.0          |   03-2025        | ViT-l/16                 | [🤗 Hugging Face](https://huggingface.co/AI4Pathology/PathOrchestra/)  |

## Pre-extracted Embeddings
To support downstream applications, we provide pre-extracted embeddings from PathOrchestra_V1.0.0, which are available for download on [🤗 Hugging Face](https://huggingface.co/datasets/AI4Pathology/pathorchestra-image-features/).

## Installation
First, clone the repository and navigate into the project directory:
```shell
git clone https://github.com/yanfang-research/PathOrchestra.git
cd PathOrchestra
```
Next, create a Conda environment and install the required dependencies:
```shell
conda create -n PathOrchestra python=3.10 -y
conda activate PathOrchestra
pip install -e .
```

### 1. Getting access
To access the model weights, please request permission via the Hugging Face model page using the links provided in the [Model Weights](#model-weights). Note that you must be logged into your Hugging Face account to download the weights.

### 2. Downloading weights + Creating model
Following authentication (using ```huggingface_hub```), the pretrained checkpoints and image transforms for PathOrchestra can be directly loaded using the [timm](https://huggingface.co//github/hub/en/timm) library. This method automatically downloads the model weights to the [huggingface_hub cache](https://huggingface.co//github/huggingface_hub/en/guides/manage-cache) in your home directory, which ```timm``` will automatically find when using the commands below:

```python
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login

login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

# pretrained=True needed to load PathOrchestra_v1.0 weights 
model = timm.create_model("hf-hub:AI4Pathology/PathOrchestra_V1.0.0.0", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
model.eval()
```
You can use the pretrained encoder to extract features from pathology patches, as follows:
```python
from PIL import Image
from torchvision import transforms

transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

image = Image.open("example.png")
image = transform(image).unsqueeze(dim=0) 

with torch.inference_mode():
    feature_emb = model(image) 
```
These pre-extracted features can be used for ROI classification (e.g., via linear probing), slide-level classification (e.g., using multiple instance learning), and various other machine learning applications.

## Acknowledgements
The project was built on top of amazing repositories such as [DINOv2](https://github.com/facebookresearch/dinov2), [UNI](https://github.com/mahmoodlab/UNI),  and [Timm](https://github.com/huggingface/pytorch-image-models/) (ViT model implementation). We thank the authors and developers for their contribution. 

## Reference
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://arxiv.org/abs/2503.24345):

Yan, F., Wu, J., Li, J., Wang, W., Lu, J., Chen, W., ... & Wang, Z. (2025). Pathorchestra: A comprehensive foundation model for computational pathology with over 100 diverse clinical-grade tasks. arXiv preprint arXiv:2503.24345

```
@article{yan2025pathorchestra,
  title={Pathorchestra: A comprehensive foundation model for computational pathology with over 100 diverse clinical-grade tasks},
  author={Yan, Fang and Wu, Jianfeng and Li, Jiawen and Wang, Wei and Lu, Jiaxuan and Chen, Wen and Gao, Zizhao and Li, Jianan and Yan, Hong and Ma, Jiabo and others},
  journal={arXiv preprint arXiv:2503.24345},
  year={2025}
}
``` 
        
        .
