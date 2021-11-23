import pickle
import copy
import numpy as np
from PIL import Image
from matplotlib import cm

import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as trans_f

from tl2.proj.pil import pil_utils
from tl2.proj.numpy import np_utils

from torch_utils import misc
import dnnlib
import legacy


def get_optim_noise_maps(G, ignore_name=None):
  noises_dict = {}
  # {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
  for name, buf in G.synthesis.named_buffers():
    if 'noise_const' in name and ignore_name != name:
      noises_dict[name] = buf
  return noises_dict

def get_noises_pil(noises_dict):
  pyramid0 = []
  pyramid1 = []
  for name, noise in noises_dict.items():
    noise_np = noise.detach().cpu().numpy()
    if 'conv0' in name:
      pyramid0.append(noise_np)
    if 'conv1' in name:
      pyramid1.append(noise_np)
    pass
  pyramid0.reverse()
  pyramid1.reverse()

  image_list = []
  image_list.extend(pyramid0[:6] + pyramid1[:6])
  for i in range(6, 9):
    image_list.append(pyramid0[i])
    image_list.append(pyramid1[i])

  image_pil_list = []
  size = 340
  for img_np in image_list:
    gray_np_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    img_cmap = cm.Greys(1 - gray_np_norm)
    img_pil = Image.fromarray(np.uint8(img_cmap * 255))
    cur_size = img_pil.size[0]
    img_pil = img_pil.resize((size, size), Image.NEAREST)
    pil_utils.add_text(img_pil, text=f"{cur_size}", size=size // 8)
    image_pil_list.append(img_pil)

  noise_pil = pil_utils.merge_image_pil(image_pil_list, nrow=6, pad=1)
  return noise_pil


@torch.no_grad()
def generate_pil(G, z, label, shape, cur_trunc=1., noise_mode='const'):
  img = G(z, label, shape=shape, truncation_psi=cur_trunc, noise_mode=noise_mode)
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  img_pil = Image.fromarray(img[0].cpu().numpy(), 'RGB')
  return img_pil


@torch.no_grad()
def generate_grid_pil(G, zs, labels, shape, nrow, cur_trunc=1., noise_mode='const'):
  img_pils = []
  for idx in range(len(zs)):
    img_pil = generate_pil(G, zs[idx, None], labels[[idx]], shape, cur_trunc, noise_mode)
    img_pils.append(img_pil)
  merged_pil = pil_utils.merge_image_pil(img_pils, nrow=nrow, pad=1, dst_size=2048)
  return merged_pil


@torch.no_grad()
def to_pil(Gz):
  Gz = Gz.squeeze()
  synth_image = ((Gz + 1) / 2).clamp(0, 1)
  img_pil = trans_f.to_pil_image(synth_image)
  # synth_image = synth_image.permute(2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
  return img_pil


def load_pil_crop_resize(image_path, out_size=None):
  target_pil = pil_utils.pil_open_rgb(image_path)
  # target_pil = Image.open(image_path).convert('RGB')
  w, h = target_pil.size
  s = min(w, h)
  target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
  if out_size:
    target_pil = target_pil.resize((out_size, out_size), Image.LANCZOS)
  return target_pil


def load_pkl(url):
  with open(url, "rb") as fp:
    model = legacy.load_network_pkl(fp)
  return model


def load_G_ema(network_pkl):
  device = torch.device('cuda')
  # Load networks.
  print('Loading networks from "%s"...' % network_pkl)
  model = load_pkl(network_pkl)
  G = model['G_ema'].requires_grad_(False).to(device)
  return G


def save_model_dict(model_dict, snapshot_pkl, rank=0, num_gpus=1):
  snapshot_data = model_dict
  # for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
  for name in ['G', 'D', 'G_ema', 'augment_pipe']:
    module = model_dict[name]
    if module is not None:
      if num_gpus > 1:
        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
      module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
    snapshot_data[name] = module
    del module  # conserve memory
  if rank == 0:
    with open(snapshot_pkl, 'wb') as f:
      pickle.dump(snapshot_data, f)
  print(f"Save model_dict to {snapshot_pkl}")


def load_w(w_file):
  w = np.load(w_file)['w']
  return w


def load_w_and_n(w_file):
  w = np.load(w_file)['w']
  ns = np_utils.np_load_dict(w_file, key="noise_bufs")
  return w, ns


def load_w_and_n_tensor(w_file, device):
  w, ns = load_w_and_n(w_file)

  w_tensor = torch.from_numpy(w).to(device).squeeze().unsqueeze(dim=0)
  ns_tensor = {}
  for name, n in ns.items():
    ns_tensor[name] = torch.from_numpy(n).to(device)
  return w_tensor, ns_tensor


@torch.no_grad()
def G_synthesis(G, w, noise_mode='const'):
  """
  noise_mode: ['random', 'const', 'none']
  """
  device = torch.device('cuda')
  if not isinstance(w, torch.Tensor):
    w = torch.from_numpy(w).to(device)
  w = w.squeeze().unsqueeze(dim=0)
  synthesis_kwargs = {'noise_mode': noise_mode,
                      'return_ori_out': False}
  img = G.synthesis(w, **synthesis_kwargs)
  return img


@torch.no_grad()
def G_w(G, w, W=None, H=None, noise_mode='const'):
  """
  noise_mode: ['random', 'const', 'none']
  """
  shape = G.synthesis.shape
  if W and H:
    G.synthesis.shape = (H, W)

  img = G_synthesis(G, w, noise_mode=noise_mode)

  G.synthesis.shape = shape
  return img

@torch.no_grad()
def replace_ns(G, ns):
  ns_original = {}
  for name, buf in G.synthesis.named_buffers():
    if name in ns:
      ns_original[name] = buf.clone()
      buf.data.copy_(ns[name].data)
  return ns_original


@torch.no_grad()
def G_w_ns(G, w, ns, W=None, H=None):
  ns_original = replace_ns(G, ns)
  shape = G.synthesis.shape
  if W and H:
    G.synthesis.shape = (H, W)

  img = G_synthesis(G, w)

  replace_ns(G, ns_original)
  G.synthesis.shape = shape
  return img


@torch.no_grad()
def get_mean_w(G, class_id=-1, w_avg_samples=10000, seed=0, device = torch.device('cuda')):
  z_samples = np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)

  label = torch.zeros([1, G.c_dim], device=device)
  if class_id >= 0:
    label[:, class_id] = 1

  w_samples = G.mapping(torch.from_numpy(z_samples).to(device), label.to(device))  # [N, L, C]
  w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
  w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
  # w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5
  w_avg = w_avg.repeat(G.num_ws, axis=1)
  w_avg = torch.from_numpy(w_avg).to(device=device)
  return w_avg


def get_style_mixing_idx_dict(num_w, mid_w_idx, source_w=1):
  w_source_dict = {}
  for i in range(num_w):
    if i < mid_w_idx:
      w_source_dict[i] = source_w
    else:
      w_source_dict[i] = 1 - source_w
  return w_source_dict


def get_mixed_w(source_w, style_w, source_w_idx):
  if isinstance(source_w_idx, dict):
    source_w_idx = list(source_w_idx.values())
  mixed_w = style_w.clone()
  source_w_idx_bool = [bool(v) for v in source_w_idx]
  mixed_w[0, source_w_idx_bool] = source_w[0, source_w_idx_bool]
  return mixed_w


def get_mixed_w_mixing_and_interp(source_w, style_w, source_w_idx, gama_interp):
  """
  first mixing then linear interpolation
  """
  mixed_w = get_mixed_w(source_w=source_w, style_w=style_w, source_w_idx=source_w_idx)
  mixed_w = gama_interp * style_w.clone() + (1 - gama_interp) * mixed_w.clone()

  # mixed_w = gama_interp * style_w.clone() + (1 - gama_interp) * source_w.clone()
  # source_w_idx_bool = [bool(v) for v in source_w_idx]
  # mixed_w[0, source_w_idx_bool] = source_w[0, source_w_idx_bool]
  return mixed_w


def get_mixed_w_dict_mixing_and_interp(source_w, style_w, source_w_idx, num_interp):
  gamas = np.linspace(0, 1, num_interp)
  mixed_w_dict = {}
  for gama in gamas:
    mixed_w = get_mixed_w_mixing_and_interp(source_w=source_w, style_w=style_w, source_w_idx=source_w_idx, gama_interp=gama)
    mixed_w_dict[str(gama)] = mixed_w
  return mixed_w_dict


def save_w_ns(saved_file, w, ns):
  ns_np = {name: buf.detach().cpu().numpy() for (name, buf) in ns.items()}
  np.savez(saved_file, w=w.detach().cpu().numpy(), noise_bufs=ns_np)
  pass


def switch_inr_sample_mode(G, sample_mode='bicubic'):
  """
  sample_mode: ['bilinear', 'bicubic']
  """
  G.synthesis.inr_net.sample_mode = sample_mode
  pass
