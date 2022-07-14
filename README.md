# HRInversion

A minimal PyTorch implementation of VGG conv-based perceptual loss (not the VGG relu-based perceptual loss). 

# Install

```bash
# pip
pip install git+https://github.com/PeterouZh/HRInversion.git

# local
git clone https://github.com/PeterouZh/HRInversion.git
pip install -e HRInversion

```

# Usage

```python
import torch
import torch.nn.functional as F
from hrinversion import VGG16ConvLoss

input = (torch.rand(1, 3, 1024, 1024).cuda() - 0.5) * 2  # [-1, 1]
input = torch.nn.Parameter(input)
target = (torch.rand(1, 3, 1024, 1024).cuda() - 0.5) * 2  # [-1, 1]

percep_loss = VGG16ConvLoss().cuda()

fea_input = percep_loss(input)
fea_target = percep_loss(target)
loss = F.mse_loss(fea_input, fea_target, reduction='sum')
loss.backward()

```


# Demo videos

<details open>
<summary>
  Inversion (please zoom in to see artifacts. )
</summary>
  
https://user-images.githubusercontent.com/26176709/178883567-032c562c-b529-4eb4-af3d-5f0013d87db6.mp4
</details>

<details open>
<summary>
  LSO for ukiyo_e
</summary>
  
https://user-images.githubusercontent.com/26176709/178882562-a59990db-f01c-43e5-8600-bbbd7e9eac4b.mp4
</details>

<details open>
<summary>
  LSO for disney
</summary>
  
https://user-images.githubusercontent.com/26176709/178883223-a0498706-ed5c-4413-9412-8a63f99ae305.mp4
</details>

# A web demo using hrinversion

<!-- https://user-images.githubusercontent.com/26176709/177040601-17c9581c-eac7-498c-b486-a7cbcdc417c2.mp4 -->

- Prepare models
```bash
# Download StyleGAN2 models
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl -P datasets/pretrained/
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt -P datasets/pretrained/
  
```
```text
tree datasets/

datasets/
├── aligned_faces
│   └── test.png
└── pretrained
    ├── ffhq-res1024-mirror-stylegan2-noaug.pkl
    └── vgg16.pt
```

- Start a web demo:
```bash
streamlit run --server.port 8501 \
  hrinversion/scripts/projector_web.py -- \
  --cfg_file hrinversion/configs/projector_web.yaml \
  --command projector_web \
  --outdir results/projector_web

```

- (optional) Debug the script with this command:
```bash
python hrinversion/scripts/projector_web.py \
  --cfg_file hrinversion/configs/projector_web.yaml \
  --command projector_web \
  --outdir results/projector_web \
  --debug True

```

- Results

<img src=".github/screen.png" width="600">


## Stylization

TBD

## Acknowledgments

- stylegan2-ada from [https://github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)





