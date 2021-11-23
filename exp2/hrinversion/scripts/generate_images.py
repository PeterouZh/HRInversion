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
import torch.distributed as dist
import torch.utils.data as data_utils
import torchvision.transforms as tv_trans

from tl2.launch.launch_utils import update_parser_defaults_from_yaml, global_cfg
from tl2.modelarts import modelarts_utils
from tl2.proj.pil import pil_utils
from tl2.proj.skimage import skimage_utils
from tl2.proj.pytorch.ddp import ddp_utils
from tl2.tl2_utils import AverageMeter
from tl2.proj.logger.textlogger import summary_dict2txtfig, global_textlogger
from tl2.proj.fvcore import build_model
from tl2.proj.pytorch.examples.multi_process_main.dataset import ImageListDataset
from tl2.proj.argparser import argparser_utils
from tl2.proj.pytorch import torch_utils
from tl2 import tl2_utils
from tl2.proj.fvcore.checkpoint import Checkpointer


from exp2.comm import stylegan_utils


def build_parser():
  ## runtime arguments
  parser = argparse.ArgumentParser(description='Training configurations.')

  argparser_utils.add_argument_int(parser, name='local_rank', default=0)
  argparser_utils.add_argument_int(parser, name='seed', default=0)
  argparser_utils.add_argument_int(parser, name='num_workers', default=0)

  argparser_utils.add_argument_str(parser, name='result_dir')


  return parser


def get_dataset(image_list,
                distributed,
                num_workers=0,
                debug=False):

  dataset = ImageListDataset(meta_file=image_list)

  if distributed:
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
  else:
    sampler = None

  data_loader = data_utils.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=False)

  if debug:
    data_iter = iter(data_loader)
    data = next(data_iter)

  num_images = len(dataset)
  return data_loader, num_images


def main():

  parser = build_parser()
  args, _ = parser.parse_known_args()

  rank, world_size = ddp_utils.ddp_init(seed=args.seed)
  torch_utils.init_seeds(seed=args.seed, rank=rank)
  device = torch.device('cuda')

  is_main_process = (rank == 0)

  update_parser_defaults_from_yaml(parser, is_main_process=is_main_process)
  logger = logging.getLogger('tl')
  if is_main_process:
    modelarts_utils.setup_tl_outdir_obs(global_cfg)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_download', {}), global_cfg=global_cfg)

  outdir = f"{args.result_dir}"
  shutil.rmtree(outdir, ignore_errors=True)
  os.makedirs(outdir, exist_ok=True)

  # Load networks.
  G_tmp = stylegan_utils.load_G_ema(global_cfg.network_pkl[global_cfg.default_network_pkl]).eval()

  G = build_model(cfg=global_cfg.G_cfg).eval().requires_grad_(False).to(device)
  ret = Checkpointer(G).load_state_dict(G_tmp.state_dict())
  G.eval()

  if rank == 0:
    pbar = tqdm.tqdm(total=global_cfg.num_images, desc=outdir)

  count = 0
  with torch.no_grad():
    while count < global_cfg.num_images:
      z = torch.randn([global_cfg.batch_size, G.z_dim], device=device)
      c = torch.zeros(global_cfg.batch_size, 0)

      batch_ws = G.mapping(z, c, truncation_psi=global_cfg.truncation_psi, truncation_cutoff=None)
      batch_imgs = G.synthesis(batch_ws, return_ori_out=False, noise_mode='const')
      for idx_b in range(global_cfg.batch_size):
        cur_img = batch_imgs[idx_b]
        cur_ws = batch_ws[[idx_b]]
        img_pil = stylegan_utils.to_pil(cur_img)

        cur_name = count + rank * global_cfg.batch_size + idx_b
        cur_path = Path(f"{outdir}/{cur_name:05d}.png")
        pil_utils.pil_save(img_pil, image_path=cur_path, save_png=False)

        cur_w_name = f"{cur_path.stem}_w.npz"
        stylegan_utils.save_w_ns(saved_file=f"{outdir}/{cur_w_name}", w=cur_ws, ns={})
        pass

      count += world_size * global_cfg.batch_size
      if rank ==0: pbar.update(world_size * global_cfg.batch_size)
      ddp_utils.d2_synchronize()


  if is_main_process:
    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
  ddp_utils.d2_synchronize()
  pass


if __name__ == '__main__':
  main()
