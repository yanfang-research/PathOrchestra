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
- 7-07-2025: Features Released (ing)  
- 3-31-2025: Article Online
- 3-02-2025: Model Weights (V1.0.0) Released
- 7-14-2024: Initial Release

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

## Public Datasets Used in Downstream Tasks

| Dataset | Reference | Link |
|--------|-----------|------|
| FocusPath-UofT | Hosseini et al., 2019 | https://sites.google.com/view/focuspathuoft/database |
| CAMELYON16 | CAMELYON16 | https://camelyon16.grand-challenge.org |
| CAMELYON17 | CAMELYON17 | https://camelyon17.grand-challenge.org |
| TCGA-TILs | TCGA-TILs | https://zenodo.org/records/6604094 |
| PCam | Veeling et al. | https://github.com/basveeling/pcam |
| GlaS | GlaS Challenge | https://www.kaggle.com/datasets/sani84/glasmiccai2015-gland-segmentation |
| PanNuke | Gamper et al. | https://link.springer.com/chapter/10.1007/978-3-030-23937-4_2 
| CoNSeP | Graham et al. | https://paperswithcode.com/dataset/consep |
| COSAS | COSAS Challenge | https://cosas.grand-challenge.org/teams/ |
| TissueNet | DrivenData | https://www.drivendata.org/competitions/67/competition-cervical-biopsy/page/255/ |
| LC25K | tampapath | https://github.com/tampapath/lung_colon_image_set |
| BreakHis | Spanhol et al. | https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/ |
| TCGA-NSCLC | TCGA | https://portal.gdc.cancer.gov/ |
| TCGA-RCC | TCGA | https://portal.gdc.cancer.gov/ |
| BACH | ICiar2018 | https://iciar2018-challenge.grand-challenge.org/Dataset/ |
| TCGA-ESCA | TCGA | https://zenodo.org/record/7548828 |
| HunCRC | Feczko et al., 2022 | https://www.nature.com/articles/s41597-022-01450-y |
| PANDA | Bulten et al. | https://panda.grand-challenge.org/data/ |
| PatchGastricADC22 | Lee et al., 2022 | https://zenodo.org/records/6550925 |
| AGGC | AGGC Challenge | https://aggc22.grand-challenge.org |
| TCGA-IDH1 | TCGA | https://www.nature.com/articles/s41597-022-01450-y |
| CRC-100K | CRC-100K Dataset | https://zenodo.org/records/1214456 |
| Chaoyang | HSA-NRL Project | https://bupt-ai-cz.github.io/HSA-NRL/ |
| WSSS4LUAD | WSSS4LUAD | https://wsss4luad.grand-challenge.org/ |
| Kather | Kather et al. | https://zenodo.org/records/53169 |
| Ebrains | EBRAINS | https://search.kg.ebrains.eu/instances/Dataset/8fc108ab-e2b4-406f-8999-60269dc1f994 |
| HEST | Jaume et al., 2024 | https://github.com/mahmoodlab/HEST |
| DeepCell | DeepCell Team | https://datasets.deepcell.org/data |
| DigestPath | Da et al., 2022 | https://paperswithcode.com/dataset/digestpath |
| SegPath | Komura et al. | https://dakomura.github.io/SegPath/ |
| CoNSeG | Wu et al., 2023 | https://github.com/zzw-szu/CoNuSeg |

## Acknowledgements
The project was built on top of amazing repositories such as [DINOv2](https://github.com/facebookresearch/dinov2), [UNI](https://github.com/mahmoodlab/UNI),  and [Timm](https://github.com/huggingface/pytorch-image-models/) (ViT model implementation). We thank the authors and developers for their contribution. 

## External Evaluation
We are pleased to assist researchers in evaluating their models using our private datasets. For more details, please feel free to contact us at the provided email (yanfang@pjlab.org.cn).

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
