# HRInversion

## Envs
```bash
# Create virtual environment
conda create -y --name hrinversion python=3.6.7
conda activate hrinversion

pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install --no-cache-dir -r requirements.txt

```

## Perceptual loss: downsampling (artifacts) vs. no downsampling

Download the official pre-trained StyleGAN2 model [ffhq-res1024-mirror-stylegan2-noaug.pkl](https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl).
Put the model file in `datasets2/pretrained`.

Run the command below:
```bash

export CUDA_HOME=/usr/local/cuda-10.2/
streamlit run --server.port 8661 \
  exp2/hrinversion/scripts/projector_web.py -- \
  --cfg_file exp2/hrinversion/configs/projector_web.yaml \
  --command projector_web \
  --outdir results/projector_web

```
and then open the browser: `http://your_ip_address:8661`

Note that the MSE loss may dominate the total loss, impeding us from analyzing the nature of perceptual loss. 
In order to observe the artifacts, we use only the perceptual loss to jointly optimize w and n. 
If you select the downsampling option on the web page, you will observe artifacts caused by the perceptual loss. 


You can debug the script with this command:
```bash

export CUDA_HOME=/usr/local/cuda-10.2/
python exp2/hrinversion/scripts/projector_web.py \
  --cfg_file exp2/hrinversion/configs/projector_web.yaml \
  --command projector_web \
  --outdir results/projector_web \
  --debug True

```

The LSO experiments requires fine-tuning models on target domains first. 
Please refer to `exp2/hrinversion/scripts` for more scripts. 


## Acknowledgments

- stylegan2-ada from [https://github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)





