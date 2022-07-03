# HRInversion



# Install

```bash
# pip
pip install git+https://github.com/PeterouZh/HRInversion.git

# local
git clone https://github.com/PeterouZh/HRInversion.git
pip install -e HRInversion

```

# Usage

```bash


```

# A GAN inversion demo using hrinversion

```bash
pip install torch

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

Start a web demo:
```bash
streamlit run --server.port 8501 \
  hrinversion/scripts/projector_web.py -- \
  --cfg_file hrinversion/configs/projector_web.yaml \
  --command projector_web \
  --outdir results/projector_web

```

Debug the script with this command:
```bash
python hrinversion/scripts/projector_web.py \
  --cfg_file hrinversion/configs/projector_web.yaml \
  --command projector_web \
  --outdir results/projector_web \
  --debug True

```


## Acknowledgments

- stylegan2-ada from [https://github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)





