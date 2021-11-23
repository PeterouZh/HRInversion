import shutil
import logging
import json
import tqdm
from time import perf_counter
from pathlib import Path
import argparse
import os
import random
import numpy as np

import torch

from tl2.launch.launch_utils import global_cfg, TLCfgNode, set_global_cfg
from tl2.modelarts import modelarts_utils
from tl2.proj.pil import pil_utils
from tl2.proj.skimage import skimage_utils
from tl2.proj.pytorch.ddp import ddp_utils
from tl2.tl2_utils import AverageMeter
from tl2.proj.logger.textlogger import summary_dict2txtfig, global_textlogger
from tl2.proj.fvcore import build_model
from tl2.proj.argparser import argparser_utils
from tl2.proj.pytorch import torch_utils
from tl2 import tl2_utils


def build_parser():
  ## runtime arguments
  parser = argparse.ArgumentParser()

  argparser_utils.add_argument_list_of_str(parser, name='aligned_img_list')
  argparser_utils.add_argument_str(parser, name='cfg_file', default='')
  argparser_utils.add_argument_str(parser, name='command', default='')
  argparser_utils.add_argument_str(parser, name='outdir', default='results')
  argparser_utils.add_argument_bool(parser, name='debug', default=False)

  return parser

def main(cfg,
         aligned_img_list,
         outdir,
         debug=False,
         ):

  set_global_cfg(cfg)
  global_cfg.tl_debug = debug
  print(global_cfg.dump())

  device = torch.device('cuda')

  outdir = outdir
  os.makedirs(outdir, exist_ok=True)

  # model
  projector = build_model(cfg=global_cfg.model_cfg, kwargs_priority=True,
                          network_pkl=global_cfg.model_cfg.network_pkl[global_cfg.model_cfg.default_network_pkl])

  image_list_todo = []
  w_file_list = []
  for image_path in aligned_img_list:
    proj_w_name = projector._get_proj_w_name(
      image_path=image_path, optim_noise_bufs=global_cfg.project_wplus_kwargs.optim_noise_bufs)
    w_file = f"{outdir}/{proj_w_name}.npz"
    w_file_list.append(w_file)
    if not os.path.isfile(w_file):
      image_list_todo.append(image_path)

  print(len(image_list_todo))

  num_images = len(image_list_todo)
  pbar = tqdm.tqdm(desc=outdir, total=num_images)

  lpips_metric = skimage_utils.LPIPS(device=device)

  for idx, image_path in enumerate(image_list_todo):
    pbar.update()
    image_path = Path(image_path)

    start_time = perf_counter()
    projector.reset()
    ret_metric_dict = projector.project_wplus(
      outdir=outdir,
      image_path=image_path,
      lpips_metric=lpips_metric,
      **global_cfg.project_wplus_kwargs
    )

    elapsed_time = (perf_counter() - start_time)
    tqdm.tqdm.write(f'processing {image_path.stem} elapsed: {elapsed_time:.1f} s')

    # if global_cfg.tl_debug:
    #   break

  return w_file_list


if __name__ == '__main__':
  parser = build_parser()
  args, _ = parser.parse_known_args()
  argparser_utils.print_args(args)

  cfg = TLCfgNode.load_yaml_with_command(cfg_filename=args.cfg_file, command=args.command)

  main(cfg=cfg,
       aligned_img_list=args.aligned_img_list,
       outdir=args.outdir,
       debug=args.debug
       )
