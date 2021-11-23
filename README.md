# HRInversion

## Envs
```bash
# Create virtual environment
conda create -y --name hrinversion python=3.6.7
conda activate hrinversion

pip install torch==1.8.2+cu102 torchvision==0.9.2+cu102 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
pip install --no-cache-dir -r requirements.txt

```

## Downsampling vs. No downsampling

Please run the command below to do inversion for a real image:
```bash
streamlit run --server.port 8661 \
  exp2/hrinversion/scripts/projector_web.py -- \
  --cfg_file exp2/hrinversion/configs/projector_web.yaml \
  --command projector_web \
  --outdir results/projector_web

```

```bash
python exp2/hrinversion/scripts/projector_web.py \
  --cfg_file exp2/hrinversion/configs/projector_web.yaml \
  --command projector_web \
  --outdir results/projector_web \
  --debug True

```








