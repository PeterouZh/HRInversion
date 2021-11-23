import json
import collections
import pathlib
import tqdm
import itertools
import PIL.Image
import pickle
import copy
from time import perf_counter
import numpy as np
import collections
from pathlib import Path
import logging
import os
import sys
from PIL import Image
import streamlit as st

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, f"{os.getcwd()}/ada_lib")
sys.path.insert(0, f"{os.getcwd()}/timm_lib")

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.proj.streamlit import SessionState
from tl2.proj.streamlit import st_utils
from tl2.proj.logger.logger_utils import get_file_logger
from tl2 import tl2_utils
from tl2.proj.pil import pil_utils
from tl2.proj.streamlit import st_utils
from tl2.proj.fvcore import build_model, MODEL_REGISTRY
from tl2.proj.pytorch import torch_utils

# sys.path.insert(0, f"{os.getcwd()}/DGP_lib")
# from DGP_lib import utils
# sys.path.pop(0)


@MODEL_REGISTRY.register(name_prefix=__name__)
class HRInversion_Web(object):
  def __init__(self):



    pass

  def project_image_web(self,
                        cfg,
                        outdir,
                        saved_suffix_state=None,
                        **kwargs):

    st_utils.st_set_sep(msg="image_list")
    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      data_k = k
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=data_k, )
      image_list_kwargs[data_k]['image_path'] = image_path
    data_k = st_utils.radio('source', options=image_list_kwargs.keys(), index=0, sidebar=True)
    image_path = image_list_kwargs[data_k]['image_path']

    image_input = st_utils.text_input('input image path:', '')
    if image_input:
      image_path = pathlib.Path(image_input)

    img_pil = Image.open(image_path)
    st_utils.st_image(img_pil, caption=f"{img_pil.size}, {data_k}", debug=False, )
    st.write(image_path)

    # ****************************************************************************

    st_utils.st_set_sep(msg='miscellaneous')
    net_name = st_utils.selectbox(label='network_pkl', options=cfg.network_pkl.keys(),
                                  default_value=cfg.default_network_pkl, sidebar=True)
    network_pkl = cfg.network_pkl[net_name]
    seed = st_utils.number_input('seed', cfg.seed, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', False, sidebar=True)

    st_utils.st_set_sep(msg='perceptual loss')
    loss_name = st_utils.selectbox('loss_cfg', cfg.loss_cfg.keys(), default_value=cfg.default_loss_cfg, sidebar=True)
    loss_cfg = cfg.loss_cfg[loss_name]
    st_utils.st_set_sep(msg=loss_name, num_symbol=2)
    if loss_name in ['vgg16_jit', ]:
      loss_cfg.resize_input = st_utils.checkbox('resize_input', loss_cfg.resize_input, sidebar=True)
    elif loss_name in ['vgg16_conv_r1024_loss_cfg']:
      loss_cfg.downsample_size = st_utils.number_input('downsample_size', loss_cfg.downsample_size, sidebar=True)
      loss_cfg.use_stat_loss = st_utils.checkbox('use_stat_loss', loss_cfg.use_stat_loss, sidebar=True)
      loss_cfg.layers = st_utils.parse_list_from_st_text_input('layers', loss_cfg.layers)
      loss_w_dict = st_utils.parse_dict_from_st_text_input('loss_w_dict', loss_cfg.loss_w_dict)
      loss_cfg.update({'loss_w_dict': loss_w_dict})
    else:
      assert 0, loss_name

    st_utils.st_set_sep(msg='project')
    num_steps = st_utils.number_input('num_steps', cfg.num_steps, sidebar=True)
    optim_noise_bufs = st_utils.checkbox('optim_noise_bufs', cfg.optim_noise_bufs, sidebar=True)
    regularize_noise_weight = st_utils.number_input(
      'regularize_noise_weight', float(cfg.regularize_noise_weight), sidebar=True)
    initial_learning_rate = st_utils.number_input(
      'initial_learning_rate', cfg.initial_learning_rate, sidebar=True, format="%.6f")
    proj_w_plus = st_utils.checkbox('proj_w_plus', cfg.proj_w_plus, sidebar=True)
    mse_weight = st_utils.number_input('mse_weight', loss_cfg.get('mse_weight', cfg.mse_weight), sidebar=True)
    normalize_noise = st_utils.checkbox('normalize_noise', cfg.normalize_noise, sidebar=True)
    st_log_every = st_utils.number_input('st_log_every', 100, sidebar=True)

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    # ****************************************************************************

    # np.random.seed(seed)
    # torch.manual_seed(seed)
    torch_utils.init_seeds(seed=seed, rank=0)

    class_id = None

    from .projector import StyleGAN2Projector

    projector = StyleGAN2Projector(
      network_pkl=network_pkl,
      loss_cfg=loss_cfg)

    start_time = perf_counter()
    if not proj_w_plus:
      projector.project(
        outdir=outdir,
        image_path=image_path,
        class_id=class_id,
        optim_noise_bufs=optim_noise_bufs,
        num_steps=num_steps,
        regularize_noise_weight=regularize_noise_weight,
        initial_learning_rate=initial_learning_rate,
        seed=seed,
        st_web=True,
        fps=fps,
      )
    else:
      projector.project_wplus(
        outdir=outdir,
        image_path=image_path,
        class_id=class_id,
        optim_noise_bufs=optim_noise_bufs,
        num_steps=num_steps,
        regularize_noise_weight=regularize_noise_weight,
        initial_learning_rate=initial_learning_rate,
        seed=seed,
        st_web=True,
        fps=fps,
        mse_weight=mse_weight,
        normalize_noise=normalize_noise,
        st_log_every=st_log_every,
        hd_video=hd_video,
      )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    pass

  def mix_image_web(self,
                    cfg,
                    outdir,
                    saved_suffix_state=None,
                    **kwargs):

    st_utils.st_set_sep(msg="image_list")
    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      data_k = k
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=data_k, )
      image_list_kwargs[data_k]['image_path'] = image_path
      image_list_kwargs[data_k]['proj_dir'] = v.proj_dir

    default_content = st_utils.radio(label='default_content', options=image_list_kwargs.keys(),
                                     default_value=cfg.default_content, sidebar=True)
    default_style = st_utils.radio(label='default_style', options=image_list_kwargs.keys(),
                                   default_value=cfg.default_style, sidebar=True)

    image_c_path = image_list_kwargs[default_content]['image_path']
    proj_c_dir = image_list_kwargs[default_content]['proj_dir']
    image_s_path = image_list_kwargs[default_style]['image_path']
    proj_s_dir = image_list_kwargs[default_style]['proj_dir']

    image_input = st_utils.text_input('input image path:', '')
    if image_input:
      image_c_path = pathlib.Path(image_input)

    img_content_pil = Image.open(image_c_path)
    img_style_pil = Image.open(image_s_path)
    merged_img = pil_utils.merge_image_pil([img_content_pil, img_style_pil], nrow=2, pad=1)
    st_utils.st_image(merged_img, caption=f"{img_content_pil.size}, {img_style_pil.size}", debug=False, )
    st.write(f"{image_c_path}, {image_s_path}")

    # ****************************************************************************

    st_utils.st_set_sep(msg='miscellaneous')
    network_pkl_c = st_utils.radio(label='default_network_pkl_c', options=cfg.network_pkl.keys(),
                                   default_value=cfg.default_network_pkl_c, sidebar=True,
                                   value_dict=cfg.network_pkl)

    network_pkl_s = st_utils.radio(label='default_network_pkl_s', options=cfg.network_pkl.keys(),
                                   default_value=cfg.default_network_pkl_s, sidebar=True,
                                   value_dict=cfg.network_pkl)

    seed = st_utils.number_input('seed', cfg.seed, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    st_log_every = st_utils.number_input('st_log_every', 100, sidebar=True)

    # ****************************************************************************

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    # ****************************************************************************

    # np.random.seed(seed)
    # torch.manual_seed(seed)
    torch_utils.init_seeds(seed=seed, rank=0)

    from .projector import MixImages

    mixer = MixImages(
      network_pkl_c=network_pkl_c,
      network_pkl_s=network_pkl_s,
      loss_cfg=None)

    start_time = perf_counter()
    mixer.mix_images(
      outdir=outdir,
      image_c_path=image_c_path,
      image_s_path=image_s_path,
      proj_c_dir=proj_c_dir,
      proj_s_dir=proj_s_dir,
      st_web=True,
      fps=fps,
      st_log_every=st_log_every,
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    pass

  def lso_image_web(self,
                    cfg,
                    outdir,
                    saved_suffix_state=None,
                    **kwargs):

    st_utils.st_set_sep(msg="image_list")
    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      data_k = k
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=data_k, )
      image_list_kwargs[data_k]['image_path'] = image_path
      image_list_kwargs[data_k]['proj_dir'] = v.proj_dir

    default_content = st_utils.selectbox(label='default_content', options=image_list_kwargs.keys(),
                                         default_value=cfg.default_content, sidebar=True)
    default_style = st_utils.selectbox(label='default_style', options=image_list_kwargs.keys(),
                                       default_value=cfg.default_style, sidebar=True)

    image_c_path = image_list_kwargs[default_content]['image_path']
    proj_c_dir = image_list_kwargs[default_content]['proj_dir']
    image_s_path = image_list_kwargs[default_style]['image_path']
    proj_s_dir = image_list_kwargs[default_style]['proj_dir']

    image_input = st_utils.text_input('input image path:', '')
    if image_input:
      image_c_path = pathlib.Path(image_input)

    img_content_pil = Image.open(image_c_path)
    img_style_pil = Image.open(image_s_path)
    merged_img = pil_utils.merge_image_pil([img_content_pil, img_style_pil], nrow=2, pad=1)
    st_utils.st_image(merged_img, caption=f"{img_content_pil.size}, {img_style_pil.size}", debug=False, )
    st.write(f"{image_c_path}, {image_s_path}")

    # ****************************************************************************

    st_utils.st_set_sep(msg='miscellaneous')
    network_pkl_c = st_utils.selectbox(label='default_network_pkl_c', options=cfg.network_pkl.keys(),
                                       default_value=cfg.default_network_pkl_c, sidebar=True,
                                       value_dict=cfg.network_pkl)
    network_pkl_s = st_utils.selectbox(label='default_network_pkl_s', options=cfg.network_pkl.keys(),
                                       default_value=cfg.default_network_pkl_s, sidebar=True,
                                       value_dict=cfg.network_pkl)

    seed = st_utils.number_input('seed', cfg.seed, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)

    st_utils.st_set_sep(msg='perceptual loss')
    loss_name = st_utils.selectbox('loss_cfg', cfg.loss_cfg.keys(), default_value=cfg.default_loss_cfg, sidebar=True)
    loss_cfg = cfg.loss_cfg[loss_name]
    st_utils.st_set_sep(msg=loss_name, num_symbol=2)
    if loss_name in ['vgg16_jit', ]:
      loss_cfg.resize_input = st_utils.checkbox('resize_input', loss_cfg.resize_input, sidebar=True)
    elif loss_name in ['vgg16_conv_r1024_loss_cfg']:
      loss_cfg.downsample_size = st_utils.number_input('downsample_size', loss_cfg.downsample_size, sidebar=True)
      loss_cfg.use_stat_loss = st_utils.checkbox('use_stat_loss', loss_cfg.use_stat_loss, sidebar=True)
      loss_cfg.layers = st_utils.parse_list_from_st_text_input('layers', loss_cfg.layers)
      loss_w_dict = st_utils.parse_dict_from_st_text_input('loss_w_dict', loss_cfg.loss_w_dict)
      loss_cfg.update({'loss_w_dict': loss_w_dict})
    else:
      assert 0, loss_name

    st_utils.st_set_sep(msg='project')
    num_steps = st_utils.number_input('num_steps', cfg.num_steps, sidebar=True)
    optim_noise_bufs = st_utils.checkbox('optim_noise_bufs', cfg.optim_noise_bufs, sidebar=True)
    regularize_noise_weight = st_utils.number_input(
      'regularize_noise_weight', float(cfg.regularize_noise_weight), sidebar=True)
    initial_learning_rate = st_utils.number_input(
      'initial_learning_rate', cfg.initial_learning_rate, sidebar=True, format="%.6f")

    mse_weight = st_utils.number_input('mse_weight', loss_cfg.get('mse_weight', cfg.mse_weight), sidebar=True)
    normalize_noise = st_utils.checkbox('normalize_noise', cfg.normalize_noise, sidebar=True)
    st_log_every = st_utils.number_input('st_log_every', cfg.st_log_every, sidebar=True)

    st_utils.st_set_sep(msg='LSO')
    swapped_blocks = st_utils.parse_list_from_st_text_input('swapped_blocks', cfg.swapped_blocks, sidebar=True)
    gamma_style_block = st_utils.number_input('gamma_style_block', cfg.gamma_style_block, sidebar=True)
    gamma_style_w = st_utils.number_input('gamma_style_w', cfg.gamma_style_block, sidebar=True)
    lso_optim_idx = st_utils.parse_list_from_st_text_input('lso_optim_idx', cfg.lso_optim_idx, sidebar=True)
    L2_reg = st_utils.number_input('L2_reg', cfg.L2_reg, sidebar=True, format="%.5f")
    w_noise_std = st_utils.number_input('w_noise_std', cfg.w_noise_std, sidebar=True, format="%.5f")
    # ****************************************************************************

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    # ****************************************************************************

    # np.random.seed(seed)
    # torch.manual_seed(seed)
    torch_utils.init_seeds(seed=seed, rank=0)

    from .projector import MixImages

    mixer = MixImages(
      network_pkl_c=network_pkl_c,
      network_pkl_s=network_pkl_s,
      loss_cfg=loss_cfg)

    start_time = perf_counter()
    mixer.lso_optim(
      outdir=outdir,
      image_c_path=image_c_path,
      image_s_path=image_s_path,
      proj_c_dir=proj_c_dir,
      proj_s_dir=proj_s_dir,
      swapped_blocks=swapped_blocks,
      gamma_style_block=gamma_style_block,
      gamma_style_w=gamma_style_w,
      lso_optim_idx=lso_optim_idx,
      L2_reg=L2_reg,
      w_noise_std=w_noise_std,
      optim_noise_bufs=optim_noise_bufs,
      num_steps=num_steps,
      regularize_noise_weight=regularize_noise_weight,
      initial_learning_rate=initial_learning_rate,
      seed=seed,
      st_web=True,
      fps=fps,
      mse_weight=mse_weight,
      normalize_noise=normalize_noise,
      st_log_every=st_log_every,
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    pass

  def layer_swapping_image_web(self,
                               cfg,
                               outdir,
                               saved_suffix_state=None,
                               **kwargs):

    st_utils.st_set_sep(msg="image_list")
    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      data_k = k
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=data_k, )
      image_list_kwargs[data_k]['image_path'] = image_path

    default_content = st_utils.selectbox(label='default_content', options=image_list_kwargs.keys(),
                                         default_value=cfg.default_content, sidebar=True)
    image_c_path = image_list_kwargs[default_content]['image_path']

    image_input = st_utils.text_input('input image path:', '')
    if image_input:
      image_c_path = pathlib.Path(image_input)

    img_content_pil = Image.open(image_c_path)
    st_utils.st_image(img_content_pil, caption=f"{img_content_pil.size}", debug=False, )
    st.write(f"{image_c_path}")

    proj_c_dir = st_utils.text_input('proj_c_dir', cfg.proj_c_dir, sidebar=True)

    # ****************************************************************************

    st_utils.st_set_sep(msg='miscellaneous')
    default_network_pkl_c = st_utils.selectbox(label='default_network_pkl_c', options=cfg.network_pkl.keys(),
                                               default_value=cfg.default_network_pkl_c, sidebar=True)
    default_network_pkl_s = st_utils.selectbox(label='default_network_pkl_s', options=cfg.network_pkl.keys(),
                                               default_value=cfg.default_network_pkl_s, sidebar=True)
    network_pkl_c = cfg.network_pkl[default_network_pkl_c]
    network_pkl_s = cfg.network_pkl[default_network_pkl_s]

    seed = st_utils.number_input('seed', cfg.seed, sidebar=True)

    st_utils.st_set_sep(msg='LSO')
    swapped_blocks = st_utils.parse_list_from_st_text_input('swapped_blocks', cfg.swapped_blocks, sidebar=True)
    gamma_style_block = st_utils.number_input('gamma_style_block', cfg.gamma_style_block, sidebar=True)

    # ****************************************************************************

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    # ****************************************************************************

    # np.random.seed(seed)
    # torch.manual_seed(seed)
    torch_utils.init_seeds(seed=seed, rank=0)

    from .projector import MixImages

    mixer = MixImages(
      network_pkl_c=network_pkl_c,
      network_pkl_s=network_pkl_s,
      loss_cfg=None)

    start_time = perf_counter()
    mixer.layer_swapping(
      outdir=outdir,
      image_c_path=image_c_path,
      proj_c_dir=proj_c_dir,
      swapped_blocks=swapped_blocks,
      gamma_style_block=gamma_style_block,
      st_web=True,
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    pass

  def lso_image_web_v1(self,
                       cfg,
                       outdir,
                       saved_suffix_state=None,
                       **kwargs):

    st_utils.st_set_sep(msg="image_list")
    image_list_kwargs = collections.defaultdict(dict)
    for k, v in cfg.image_list_files.items():
      data_k = k
      image_path = st_utils.parse_image_list(image_list_file=v.image_list_file, header=data_k, )
      image_list_kwargs[data_k]['image_path'] = image_path
      image_list_kwargs[data_k]['proj_dir'] = v.proj_dir

    default_content = st_utils.selectbox(label='default_content', options=image_list_kwargs.keys(),
                                         default_value=cfg.default_content, sidebar=True)
    default_style = st_utils.selectbox(label='default_style', options=image_list_kwargs.keys(),
                                       default_value=cfg.default_style, sidebar=True)

    image_c_path = image_list_kwargs[default_content]['image_path']
    proj_c_dir = image_list_kwargs[default_content]['proj_dir']
    image_s_path = image_list_kwargs[default_style]['image_path']
    proj_s_dir = image_list_kwargs[default_style]['proj_dir']

    image_input = st_utils.text_input('input image path:', '')
    if image_input:
      image_c_path = pathlib.Path(image_input)

    img_content_pil = Image.open(image_c_path)
    img_style_pil = Image.open(image_s_path)
    merged_img = pil_utils.merge_image_pil([img_content_pil, img_style_pil], nrow=2, pad=1)
    st_utils.st_image(merged_img, caption=f"{img_content_pil.size}, {img_style_pil.size}", debug=False, )
    st.write(f"{image_c_path}, {image_s_path}")

    # ****************************************************************************

    st_utils.st_set_sep(msg='miscellaneous')
    network_pkl_c = st_utils.selectbox(label='default_network_pkl_c', options=cfg.network_pkl.keys(),
                                       default_value=cfg.default_network_pkl_c, sidebar=True,
                                       value_dict=cfg.network_pkl)
    network_pkl_s = st_utils.selectbox(label='default_network_pkl_s', options=cfg.network_pkl.keys(),
                                       default_value=cfg.default_network_pkl_s, sidebar=True,
                                       value_dict=cfg.network_pkl)

    seed = st_utils.number_input('seed', cfg.seed, sidebar=True)
    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    hd_video = st_utils.checkbox('hd_video', False, sidebar=True)

    st_utils.st_set_sep(msg='perceptual loss')
    loss_name = st_utils.selectbox('loss_cfg', cfg.loss_cfg.keys(), default_value=cfg.default_loss_cfg, sidebar=True)
    loss_cfg = cfg.loss_cfg[loss_name]
    st_utils.st_set_sep(msg=loss_name, num_symbol=2)
    if loss_name in ['vgg16_jit', ]:
      loss_cfg.resize_input = st_utils.checkbox('resize_input', loss_cfg.resize_input, sidebar=True)
    elif loss_name in ['vgg16_conv_r1024_loss_cfg']:
      loss_cfg.downsample_size = st_utils.number_input('downsample_size', loss_cfg.downsample_size, sidebar=True)
      loss_cfg.use_stat_loss = st_utils.checkbox('use_stat_loss', loss_cfg.use_stat_loss, sidebar=True)
      loss_cfg.layers = st_utils.parse_list_from_st_text_input('layers', loss_cfg.layers)
      loss_w_dict = st_utils.parse_dict_from_st_text_input('loss_w_dict', loss_cfg.loss_w_dict)
      loss_cfg.update({'loss_w_dict': loss_w_dict})
    else:
      assert 0, loss_name

    st_utils.st_set_sep(msg='project')
    num_steps = st_utils.number_input('num_steps', cfg.num_steps, sidebar=True)
    optim_noise_bufs = st_utils.checkbox('optim_noise_bufs', cfg.optim_noise_bufs, sidebar=True)
    regularize_noise_weight = st_utils.number_input(
      'regularize_noise_weight', float(cfg.regularize_noise_weight), sidebar=True)
    initial_learning_rate = st_utils.number_input(
      'initial_learning_rate', cfg.initial_learning_rate, sidebar=True, format="%.6f")

    mse_weight = st_utils.number_input('mse_weight', loss_cfg.get('mse_weight', cfg.mse_weight), sidebar=True)
    normalize_noise = st_utils.checkbox('normalize_noise', cfg.normalize_noise, sidebar=True)
    st_log_every = st_utils.number_input('st_log_every', cfg.st_log_every, sidebar=True)

    st_utils.st_set_sep(msg='LSO')
    swapped_blocks_high = st_utils.parse_list_from_st_text_input('swapped_blocks_high', cfg.swapped_blocks_high, sidebar=True)
    gamma_style_high = st_utils.number_input('gamma_style_high', cfg.gamma_style_high, sidebar=True)

    swapped_blocks_low = st_utils.parse_list_from_st_text_input('swapped_blocks_low', cfg.swapped_blocks_low, sidebar=True)
    gamma_style_low = st_utils.number_input('gamma_style_low', cfg.gamma_style_low, sidebar=True)

    lso_optim_idx = st_utils.parse_list_from_st_text_input('lso_optim_idx', cfg.lso_optim_idx, sidebar=True)
    L2_reg = st_utils.number_input('L2_reg', cfg.L2_reg, sidebar=True, format="%.5f")
    w_noise_std = st_utils.number_input('w_noise_std', cfg.w_noise_std, sidebar=True, format="%.5f")
    # ****************************************************************************

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    # ****************************************************************************

    # np.random.seed(seed)
    # torch.manual_seed(seed)
    torch_utils.init_seeds(seed=seed, rank=0)

    from .projector import MixImages

    mixer = MixImages(
      network_pkl_c=network_pkl_c,
      network_pkl_s=network_pkl_s,
      loss_cfg=loss_cfg)

    start_time = perf_counter()
    mixer.lso_optim_v1(
      outdir=outdir,
      image_c_path=image_c_path,
      image_s_path=image_s_path,
      proj_c_dir=proj_c_dir,
      proj_s_dir=proj_s_dir,
      swapped_blocks_high=swapped_blocks_high,
      gamma_style_high=gamma_style_high,
      swapped_blocks_low=swapped_blocks_low,
      gamma_style_low=gamma_style_low,
      lso_optim_idx=lso_optim_idx,
      L2_reg=L2_reg,
      w_noise_std=w_noise_std,
      optim_noise_bufs=optim_noise_bufs,
      num_steps=num_steps,
      regularize_noise_weight=regularize_noise_weight,
      initial_learning_rate=initial_learning_rate,
      seed=seed,
      st_web=True,
      hd_video=hd_video,
      fps=fps,
      mse_weight=mse_weight,
      normalize_noise=normalize_noise,
      st_log_every=st_log_every,
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    pass

  def lerp_image_list_web(self,
                          cfg,
                          outdir,
                          saved_suffix_state=None,
                          **kwargs):

    network_pkl = st_utils.selectbox('network_pkl', options=cfg.network_pkl.keys(),
                                     default_value=cfg.default_network_pkl, value_dict=cfg.network_pkl.items(),
                                     sidebar=True)

    fps = st_utils.number_input('fps', cfg.fps, sidebar=True)
    num_interp = st_utils.number_input('num_interp', cfg.num_interp, sidebar=True)
    num_pause = st_utils.number_input('num_pause', cfg.num_pause, sidebar=True)
    resolution = st_utils.number_input('resolution', cfg.resolution, sidebar=True)

    # ****************************************************************************

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    if saved_suffix_state is not None:
      saved_suffix_state.saved_suffix = saved_suffix_state.saved_suffix + 1

    # ****************************************************************************

    from .projector import StyleGAN2Projector

    mixer = StyleGAN2Projector(
      network_pkl=network_pkl,
      loss_cfg=None)

    start_time = perf_counter()
    mixer.lerp_image_list(
      outdir=outdir,
      w_file_list=cfg.w_file_list,
      num_interp=num_interp,
      num_pause=num_pause,
      resolution=resolution,
      author_name_list=cfg.author_name_list,
      st_web=True,
      fps=fps,
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    pass
