import json
import collections
import pathlib
from PIL import Image
import tqdm
import itertools
import PIL.Image
import pickle
import copy
from time import perf_counter
import os
import streamlit as st
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as trans_f

from tl2.proj.streamlit import st_utils
from tl2.proj.fvcore import global_cfg, build_model, MODEL_REGISTRY
from tl2.proj.pil import pil_utils
from tl2.proj.cv2 import cv2_utils
from tl2.proj.skimage import skimage_utils
from tl2.proj.pytorch import torch_utils
from tl2.proj.logger import logger_utils
from tl2.proj.pytorch.ddp import ddp_utils
from tl2.proj.fvcore.checkpoint import Checkpointer

import dnnlib
from ..utils import stylegan_utils, stylegan_utils_v1
from hrinversion.models.vgg_conv_perceptual_loss import VGG16ConvLoss


@MODEL_REGISTRY.register(name_prefix=__name__)
class StyleGAN2Projector(object):
  def __init__(self,
               G_pkl,
               loss_name,
               device='cuda',
               **kwargs
               ):
    self.loss_name = loss_name

    self.device = torch.device(device)

    # Load generator.
    print('Loading networks from "%s"...' % G_pkl)
    model = stylegan_utils.load_pkl(G_pkl)
    self.G = model['G_ema'].requires_grad_(False).to(self.device)

    self.G_weight = copy.deepcopy(self.G.state_dict())

    self.synthesis_kwargs = {'noise_mode': 'const'}

    if loss_name == 'vgg16_jit':
      # Load VGG16 feature detector.
      # vgg16_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
      vgg16_url = 'datasets/pretrained/vgg16.pt'
      print('Loading vgg16_relu...')
      with dnnlib.util.open_url(vgg16_url) as f:
        self.percep_loss = torch.jit.load(f).eval().to(self.device)
    
    elif loss_name == 'vgg16_conv':
      self.percep_loss = VGG16ConvLoss().to(self.device)
    else:
      raise NotImplementedError
    pass

  def reset(self):
    self.G.load_state_dict(self.G_weight, strict=True)
    self.G = self.G.requires_grad_(False).to(self.device)
    pass

  def compute_w_stat(self, G, label, w_avg_samples, device, seed):
    # Compute w stats.
    if isinstance(label, int):
      label_c_tensor = torch.zeros([1, G.c_dim], device=device)
      label_c_tensor[:, label] = 1
      label = label_c_tensor

    print(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(seed).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), label.to(device))  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    w_avg = torch.tensor(w_avg, dtype=torch.float32, device=device)
    return w_avg, w_std

  def get_vgg16_fea(self, image_tensor, **kwargs):
    """
    
    :param image_tensor: [-1, 1]
    :param kwargs:
    :return:
    """
    if self.loss_name == 'vgg16_jit':
      # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
      image_tensor = (image_tensor + 1) * (255 / 2)
      if image_tensor.shape[2] > 256:
        image_tensor = F.interpolate(image_tensor, size=(256, 256), mode='area')
      features = self.percep_loss(image_tensor, resize_images=False, return_lpips=True)
      
    else:
      features = self.percep_loss(image_tensor, **kwargs)
      
    return features

  def project(
        self,
        outdir,
        image_path,
        class_id,
        optim_noise_bufs=True,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        seed=123,
        st_web=False,
        fps=10,
  ):
    device = self.device
    # vgg16 = self.vgg16
    G = copy.deepcopy(self.G).eval().requires_grad_(False).to(self.device)  # type: ignore

    # Load target image.
    target_pil = stylegan_utils.load_pil_crop_resize(image_path, out_size=self.G.img_resolution)
    # pil_utils.imshow_pil(target_pil)
    # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    target_uint8 = np.array(target_pil, dtype=np.uint8).transpose([2, 0, 1])
    target_images = torch.tensor(target_uint8, device=device, dtype=torch.float32).unsqueeze(0)
    target_images = (target_images / 255. - 0.5) * 2
    label = torch.zeros([1, self.G.c_dim], device=self.device)
    label[:, class_id] = 1

    assert target_images.shape == (1, G.img_channels, G.img_resolution, G.img_resolution)
    with torch.no_grad():
      target_features = self.get_vgg16_fea(image_tensor=target_images, c=label)

    w_avg, w_std = self.compute_w_stat(G=G, label=label, w_avg_samples=w_avg_samples, device=device, seed=seed)
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    if optim_noise_bufs:
      optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
      # Init noise.
      for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True
    else:
      optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    if st_web or global_cfg.tl_debug:
      st_chart_lr = st_utils.LineChart(x_label='step', y_label='lr')
      st_chart_dist = st_utils.LineChart(x_label='step', y_label='dist')
      st_chart_loss = st_utils.LineChart(x_label='step', y_label='loss')
      st_image = st.empty()
      video_f = cv2_utils.ImageioVideoWriter(outfile=f"{outdir}/video.mp4", fps=fps)

    for step in range(num_steps):
      # Learning rate schedule.
      t = step / num_steps
      w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
      lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
      lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
      lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
      lr = initial_learning_rate * lr_ramp
      for param_group in optimizer.param_groups:
        param_group['lr'] = lr
      if st_web:
        st_chart_lr.write(step, lr)

      # Synth images from opt_w.
      w_noise = torch.randn_like(w_opt) * w_noise_scale
      ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
      synth_images = G.synthesis(ws, **self.synthesis_kwargs)

      if st_web or global_cfg.tl_debug:
        img_noise_pil = stylegan_utils.to_pil(synth_images.detach())
        with torch.no_grad():
          img_pil = G.synthesis(w_opt.detach().repeat([1, G.mapping.num_ws, 1]), **self.synthesis_kwargs)
          img_pil = stylegan_utils.to_pil(img_pil.detach())
        merged_pil = pil_utils.merge_image_pil([target_pil, img_noise_pil, img_pil], nrow=3, dst_size=2048, pad=1)
        pil_utils.add_text(merged_pil, text=f"step: {step}", size=merged_pil.size[0] // 18)
        st_image.image(merged_pil, caption=f"target {target_pil.size}, img_noise_pil {img_noise_pil.size},"
                                           f"img_pil {img_pil.size}")
        video_f.write(merged_pil)

      # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
      # synth_images = (synth_images + 1) * (255 / 2)

      # Features for synth images.
      synth_features = self.get_vgg16_fea(image_tensor=synth_images, c=label)
      # if synth_images.shape[2] > 256:
      #   synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')
      # synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)

      dist = (target_features - synth_features).square().sum()

      # Noise regularization.
      reg_loss = 0.0
      for v in noise_bufs.values():
        noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
        while True:
          reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
          reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
          if noise.shape[2] <= 8:
            break
          noise = F.avg_pool2d(noise, kernel_size=2)

      loss = dist + reg_loss * regularize_noise_weight

      # Step
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      # logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
      if step > 50 and st_web:
        st_chart_dist.write(x=step, y=dist.item())
        st_chart_loss.write(x=step, y=loss.item())

      # Normalize noise.
      with torch.no_grad():
        for buf in noise_bufs.values():
          buf -= buf.mean()
          buf *= buf.square().mean().rsqrt()
      if global_cfg.tl_debug:
        break

    if st_web or global_cfg.tl_debug:
      video_f.release(st_video=True)

    # Save final projected frame and W vector.
    target_file = f'{outdir}/{image_path.stem}.jpg'
    target_pil.save(target_file)
    projected_w = w_opt.repeat([1, G.mapping.num_ws, 1])
    proj_images = G.synthesis(projected_w, **self.synthesis_kwargs)
    proj_img_pil = stylegan_utils.to_pil(proj_images)
    proj_img_pil.save(f"{outdir}/{image_path.stem}_proj.jpg")
    merged_pil = pil_utils.merge_image_pil([target_pil, proj_img_pil], nrow=2, pad=1, dst_size=2048)
    merged_pil.save(f"{outdir}/target_proj.jpg")

    proj_w_file = f'{outdir}/{image_path.stem}.npz'
    np.savez(proj_w_file, w=projected_w.detach().cpu().numpy())
    if st_web:
      st.image(merged_pil, caption=f'target_pil, proj_image {proj_img_pil.size}')
      st.subheader(f"proj w:")
      st.write(f"{target_file}")
      st.write(f"{proj_w_file}")

    return

  def _get_target_image(self, image_path):
    # Load target image.
    target_pil = stylegan_utils.load_pil_crop_resize(image_path, out_size=self.G.img_resolution)
    target_np = np.array(target_pil).transpose([2, 0, 1]) / 255.

    # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    target_uint8 = np.array(target_pil, dtype=np.uint8).transpose([2, 0, 1])
    target_images = torch.tensor(target_uint8, device=self.device, dtype=torch.float32).unsqueeze(0)
    target_images = (target_images / 255. - 0.5) * 2

    assert target_images.shape == (1, self.G.img_channels, self.G.img_resolution, self.G.img_resolution)

    label = torch.zeros([1, self.G.c_dim], device=self.device)
    return target_pil, target_np, target_images, label

  def _get_noise_bufs(self, G):
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}
    return noise_bufs

  def _get_cur_lr(self,
                  step,
                  num_steps,
                  initial_learning_rate,
                  lr_rampdown_length=0.25,
                  lr_rampup_length=0.05,
                  ):
    t = step / num_steps
    lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
    lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
    lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
    lr = initial_learning_rate * lr_ramp
    return lr

  def _get_w_noise_scale(self,
                         w_std,
                         step,
                         num_steps,
                         initial_noise_factor = 0.05,
                         noise_ramp_length = 0.75,
                         ):
    w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - (step / num_steps) / noise_ramp_length) ** 2
    return w_noise_scale

  def _get_proj_w_name(self,
                       image_path,
                       optim_noise_bufs):
    image_path = pathlib.Path(image_path)

    if optim_noise_bufs:
      proj_w_name = f"{image_path.stem}_wn"
    else:
      proj_w_name = f"{image_path.stem}_w"
    return proj_w_name

  def project_wplus(
        self,
        outdir,
        image_path,
        w_avg_samples=10000,
        optim_noise_bufs=True,
        initial_learning_rate=0.1,
        num_steps=1000,
        normalize_noise=True,
        regularize_noise_weight=1e5,
        mse_weight=0.,
        seed=123,
        lpips_metric=None,
        fps=10,
        hd_video=True,
        save_noise_bufs=False,
        st_log_every=100,
        st_web=False,
        **kwargs
  ):
    if optim_noise_bufs:
      save_noise_bufs = True

    device = self.device
    G = copy.deepcopy(self.G).eval().requires_grad_(False).to(self.device)  # type: ignore

    image_path = pathlib.Path(image_path)
    image_name = image_path.stem
    proj_w_name = self._get_proj_w_name(image_path=image_path, optim_noise_bufs=optim_noise_bufs)

    target_pil, target_np, target_images, label = self._get_target_image(image_path=image_path)

    with torch.no_grad():
      target_features = self.get_vgg16_fea(image_tensor=target_images, c=label)

    w_avg, w_std = self.compute_w_stat(G=G, label=label, w_avg_samples=w_avg_samples, device=device, seed=seed)
    # init w_opt with w_avg
    w_opt = torch.zeros(1, G.num_ws, G.w_dim, dtype=torch.float32, device=device, requires_grad=True)
    w_opt.data.copy_(w_avg)

    # Setup noise inputs.
    noise_bufs = self._get_noise_bufs(G=G)

    if optim_noise_bufs:
      optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()),
                                   betas=(0.9, 0.999), lr=initial_learning_rate)
      # Init noise.
      for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True
    else:
      optimizer = torch.optim.Adam([w_opt], betas=(0.9, 0.999), lr=initial_learning_rate)

    if st_web:
      st_chart_lr = st_utils.LineChart(x_label='step', y_label='lr')
      st_chart_noise_scale = st_utils.LineChart(x_label='step', y_label='st_chart_noise_scale')
      st_chart_percep_loss = st_utils.LineChart(x_label='step', y_label='percep_loss')
      st_chart_mse_loss = st_utils.LineChart(x_label='step', y_label='mse_loss')
      st_chart_reg_loss = st_utils.LineChart(x_label='step', y_label='reg_loss')
      st_chart_loss = st_utils.LineChart(x_label='step', y_label='loss')
      st_chart_psnr = st_utils.LineChart(x_label='step', y_label='psnr')
      st_chart_ssim = st_utils.LineChart(x_label='step', y_label='ssim')
      st_image = st.empty()
      video_f_target_noise_proj = cv2_utils.ImageioVideoWriter(
        outfile=f"{outdir}/{proj_w_name}_target_noise_proj.mp4", fps=fps, hd_video=hd_video)
      video_f_target_proj = cv2_utils.ImageioVideoWriter(
        outfile=f"{outdir}/{proj_w_name}_target_proj.mp4", fps=fps, hd_video=hd_video)
      video_f_inversion = cv2_utils.ImageioVideoWriter(
        outfile=f"{outdir}/{proj_w_name}_inversion.mp4", fps=fps, hd_video=hd_video)

    pbar = range(num_steps)
    if ddp_utils.d2_get_rank() == 0:
      pbar = tqdm.tqdm(pbar, desc=f"{image_path.stem}")

    dummy_zero = torch.tensor(0., device=device)

    for step in pbar:
      if step == num_steps - 1:
        st_log_every = 1
      # Learning rate schedule.
      lr = self._get_cur_lr(step=step, num_steps=num_steps, initial_learning_rate=initial_learning_rate)
      w_noise_scale = self._get_w_noise_scale(w_std=w_std, step=step, num_steps=num_steps)

      torch_utils.set_optimizer_lr(optimizer=optimizer, lr=lr)

      if st_web and step % st_log_every == 0:
        st_chart_lr.write(step, lr)
        st_chart_noise_scale.write(step, w_noise_scale)

      # Synth images from opt_w.
      w_noise = torch.randn_like(w_opt) * w_noise_scale
      ws = (w_opt + w_noise)
      synth_images = G.synthesis(ws, **self.synthesis_kwargs)

      if (st_web and step % st_log_every == 0):
        with torch.no_grad():
          img_pil = G.synthesis(w_opt.detach(), **self.synthesis_kwargs)
          img_pil = stylegan_utils.to_pil(img_pil.detach())
          # img_pil_np = np.array(img_pil).transpose([2, 0, 1]) / 255.
          # psnr = peak_signal_noise_ratio(image_true=target_np, image_test=img_pil_np)
          psnr = skimage_utils.sk_psnr(image_true_pil=target_pil, image_test_pil=img_pil)
          ssim = skimage_utils.sk_ssim(image_true_pil=target_pil, image_test_pil=img_pil)

          # psnr line
          st_chart_psnr.write(step, psnr)
          st_chart_ssim.write(step, ssim)

        img_noise_pil = stylegan_utils.to_pil(synth_images.detach())
        merged_pil = pil_utils.merge_image_pil([target_pil, img_noise_pil, img_pil], nrow=3, dst_size=2048, pad=1)
        pil_utils.add_text(
          merged_pil, text=f"step: {step}, psnr: {psnr:.2f}dB, ssim: {ssim:.2f}", size=merged_pil.size[0] // 10)

        st_image.image(merged_pil, caption=f"target {target_pil.size}, img_noise_pil {img_noise_pil.size},"
                                           f"img_pil {img_pil.size}")
        video_f_target_noise_proj.write(merged_pil)

        merged_pil = pil_utils.merge_image_pil([target_pil, img_pil], nrow=2, pad=0)
        video_f_target_proj.write(merged_pil)

        video_f_inversion.write(img_pil)

      synth_features = self.get_vgg16_fea(image_tensor=synth_images, c=label)
      # percep_loss = (target_features - synth_features).square().sum()
      percep_loss = F.mse_loss(synth_features, target_features, reduction='sum')

      if mse_weight > 0:
        mse_loss = F.mse_loss(synth_images, target=target_images.detach())
        mse_loss = mse_loss * mse_weight
      else:
        mse_loss = dummy_zero

      # Noise regularization.
      if regularize_noise_weight > 0:
        reg_loss = 0
        for v in noise_bufs.values():
          noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
          while True:
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
            reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
            if noise.shape[2] <= 8:
              break
            noise = F.avg_pool2d(noise, kernel_size=2)
        reg_loss = reg_loss * regularize_noise_weight
      else:
        reg_loss = dummy_zero

      loss = percep_loss + mse_loss + reg_loss

      # Step
      optimizer.zero_grad(set_to_none=True)
      loss.backward()
      optimizer.step()
      # logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
      if st_web and step >= 50 and step % st_log_every == 0:
        st_chart_percep_loss.write(x=step, y=percep_loss.item())
        st_chart_mse_loss.write(x=step, y=mse_loss.item())
        st_chart_reg_loss.write(x=step, y=reg_loss.item())
        st_chart_loss.write(x=step, y=loss.item())

      # Normalize noise.
      if normalize_noise:
        with torch.no_grad():
          for buf in noise_bufs.values():
            buf -= buf.mean()
            buf *= buf.square().mean().rsqrt()

      if global_cfg.tl_debug:
        break

    if st_web:
      video_f_target_noise_proj.release(st_video=True)
      video_f_target_proj.release(st_video=True)
      video_f_inversion.release(st_video=True)

    # save ori image
    pil_utils.pil_save(target_pil, image_path=f'{outdir}/{image_name}.png', save_png=False)
    # save proj image
    projected_w = w_opt
    proj_images = G.synthesis(projected_w, **self.synthesis_kwargs)
    proj_img_pil = stylegan_utils.to_pil(proj_images)
    pil_utils.pil_save(proj_img_pil, f"{outdir}/{proj_w_name}_proj.png", save_png=False)

    # save w and noise
    final_noise_bufs = self._get_noise_bufs(G=G)
    if save_noise_bufs:
      noise_bufs_np = {name: buf.detach().cpu().numpy() for (name, buf) in final_noise_bufs.items()}
    else:
      noise_bufs_np = {}
    np.savez(f'{outdir}/{proj_w_name}.npz',
             w=projected_w.detach().cpu().numpy(),
             noise_bufs=noise_bufs_np)
    # np_utils.np_load_dict(proj_w_file, key='noise_bufs')

    # proj_psnr image
    file_logger = logger_utils.get_file_logger(filename=f"{outdir}/{proj_w_name}.txt")
    psnr = skimage_utils.sk_psnr(image_true_pil=target_pil, image_test_pil=proj_img_pil)
    ssim = skimage_utils.sk_ssim(image_true_pil=target_pil, image_test_pil=proj_img_pil)
    if lpips_metric is None:
      lpips_metric = skimage_utils.LPIPS(device=device)
    lpips = lpips_metric.calc_lpips(target_pil, proj_img_pil)
    file_logger.info_msg(f"psnr: {psnr}\n"
                         f"ssim: {ssim}\n"
                         f"lpips: {lpips}")

    ret_dict = {'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips}
    return ret_dict

  def _get_w_n_decoded_image(self,
                             G,
                             synthesis_kwargs,
                             w_file_path,
                             ):

    w_tensor, ns_tensor = stylegan_utils.load_w_and_n_tensor(w_file=w_file_path, device=self.device)
    if len(ns_tensor):
      img_decoded = stylegan_utils_v1.G_w_ns(G=G, w=w_tensor, ns=ns_tensor, synthesis_kwargs=synthesis_kwargs)
    else:
      img_decoded = stylegan_utils_v1.G_w(G=G, w=w_tensor, synthesis_kwargs=synthesis_kwargs)
    img_decoded_pil = stylegan_utils.to_pil(img_decoded)

    return img_decoded_pil, w_tensor, ns_tensor

  def _lerp_ns(self, ns0, ns1, gamma):
    ret_ns = {}
    for name, value0 in ns0.items():
      value1 = ns1[name]
      ret_ns[name] = value0 + gamma * (value1 - value0)

    return ret_ns

  def lerp_image_list(
        self,
        outdir,
        w_file_list,
        num_interp,
        num_pause,
        resolution,
        author_name_list=None,
        fps=10,
        hd_video=False,
        st_web=False,
        **kwargs
  ):
    self.reset()
    device = self.device
    # G_c = copy.deepcopy(self.G_c).eval().requires_grad_(False).to(self.device)
    # G_s = copy.deepcopy(self.G_s).eval().requires_grad_(False).to(self.device)
    G = self.G

    decoded_pil_list = []
    w_list = []
    ns_list = []

    # load proj w and ns from npz file
    for w_file_path in w_file_list:
      img_decoded_pil, w_tensor, ns_tensor = self._get_w_n_decoded_image(
        G=G, synthesis_kwargs=self.synthesis_kwargs, w_file_path=w_file_path)
      decoded_pil_list.append(img_decoded_pil)
      w_list.append(w_tensor)
      ns_list.append(ns_tensor)

    if st_web:
      merged_pil = pil_utils.merge_image_pil(decoded_pil_list, nrow=4, dst_size=2048)
      st_utils.st_image(merged_pil, caption=img_decoded_pil.size, debug=global_cfg.tl_debug, )
      st_image_interp_model = st.empty()

    video_f_interp_model = cv2_utils.ImageioVideoWriter(
      outfile=f"{outdir}/author_list.mp4", fps=fps, hd_video=hd_video, save_gif=True)

    num_authors = len(w_list)
    if author_name_list is not None:
      assert len(author_name_list) == num_authors

    pbar = tqdm.tqdm(range(num_authors))
    for idx in pbar:
      pbar.update()
      pbar.set_description_str(w_file_list[idx])

      cur_w = w_list[idx]
      next_w = w_list[(idx + 1) % num_authors]
      cur_ns = ns_list[idx]
      next_ns = ns_list[(idx + 1) % num_authors]

      pbar_gama = tqdm.tqdm(np.linspace(0, 1, num_interp))
      for gama in pbar_gama:
        pbar_gama.update()

        # interp w
        w_ = cur_w + gama * (next_w - cur_w)
        ns_ = self._lerp_ns(cur_ns, next_ns, gamma=gama)

        with torch.no_grad():
          img_decoded = stylegan_utils_v1.G_w_ns(G=G, w=w_, ns=ns_,
                                                 synthesis_kwargs=self.synthesis_kwargs)
        img_decoded_pil = stylegan_utils.to_pil(img_decoded)

        if author_name_list is not None:
          if gama == 0:
            img_decoded_pil = img_decoded_pil.resize((resolution, resolution), Image.LANCZOS)
            author_name = f"{author_name_list[idx]}".replace("_", " ")
            pil_utils.add_text(img_decoded_pil, text=author_name,
                               size=img_decoded_pil.size[0] // 15, color=(0, 255, 0), xy=(4, 0))
            for _ in range(num_pause):
              if st_web:
                st_utils.st_image(img_decoded_pil, caption=f"{img_decoded_pil.size}", debug=global_cfg.tl_debug,
                                  st_empty=st_image_interp_model)
              video_f_interp_model.write(img_decoded_pil)
              if global_cfg.tl_debug: break

        if st_web:
          st_utils.st_image(img_decoded_pil, caption=f"{img_decoded_pil.size}", debug=global_cfg.tl_debug,
                            st_empty=st_image_interp_model)
        video_f_interp_model.write(img_decoded_pil, dst_size=resolution)
        if global_cfg.tl_debug: break
    video_f_interp_model.release(st_video=st_web)

    return



