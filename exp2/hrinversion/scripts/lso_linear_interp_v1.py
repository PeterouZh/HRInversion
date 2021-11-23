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


def build_parser():
  ## runtime arguments
  parser = argparse.ArgumentParser(description='Training configurations.')

  argparser_utils.add_argument_int(parser, name='local_rank', default=0)
  argparser_utils.add_argument_int(parser, name='seed', default=0)
  argparser_utils.add_argument_int(parser, name='num_workers', default=0)

  argparser_utils.add_argument_str(parser, name='source_image_dir')
  argparser_utils.add_argument_str(parser, name='source_proj_dir')
  argparser_utils.add_argument_str(parser, name='target_image_dir')
  argparser_utils.add_argument_str(parser, name='target_proj_dir')
  argparser_utils.add_argument_str(parser, name='result_dir')
  argparser_utils.add_argument_int(parser, name='num_processed_images', default=-1)

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
  if rank == 0:
    if global_cfg.del_outdir:
      shutil.rmtree(outdir, ignore_errors=True)
    os.makedirs(outdir, exist_ok=True)

  ddp_utils.d2_synchronize()

  # model
  projector = build_model(cfg=global_cfg.model_cfg, kwargs_priority=True,
                          network_pkl_c=global_cfg.model_cfg.network_pkl[global_cfg.model_cfg.network_pkl_c],
                          network_pkl_s=global_cfg.model_cfg.network_pkl[global_cfg.model_cfg.network_pkl_s],
                          )

  target_image_list = tl2_utils.get_filelist_recursive(args.target_image_dir, ext="*.png", to_str=True)
  if len(target_image_list) == 0:
    target_image_list = tl2_utils.get_filelist_recursive(args.target_image_dir, ext="*.jpg", to_str=True)

  print(len(target_image_list))
  assert len(target_image_list) >= world_size
  # dataset
  distributed = ddp_utils.is_distributed()
  data_loader, num_target_images = get_dataset(image_list=target_image_list,
                                               distributed=distributed,
                                               num_workers=args.num_workers,
                                               debug=global_cfg.tl_debug)

  # image_list_todo = []
  # for image_path in image_list:
  #   image_path = Path(image_path)
  #   img_ls_path = f"{outdir}/{image_path.stem}.jpg"
  #   if not os.path.isfile(img_ls_path):
  #     image_list_todo.append(str(image_path))

  source_image_list = tl2_utils.get_filelist_recursive(args.source_image_dir, ext="*.png", to_str=True)
  if len(source_image_list) == 0:
    source_image_list = tl2_utils.get_filelist_recursive(args.source_image_dir, ext="*.jpg", to_str=True)
  if args.num_processed_images > 0:
    source_image_list = source_image_list[:args.num_processed_images]
  num_source_images = len(source_image_list)

  if rank == 0:
    pbar_source = tqdm.tqdm(desc=outdir, total=num_source_images)
    pbar_target = tqdm.tqdm(total=num_target_images)

  lpips_metric = skimage_utils.LPIPS(device=device)

  for source_image_path in source_image_list:
    source_image_path = Path(source_image_path)
    if rank == 0:
      pbar_source.update(1)
      pbar_target.reset(num_target_images)

    # check if skip cur source_image
    metric_file = f"{outdir}/{source_image_path.stem}.txt"
    if os.path.isfile(metric_file):
      continue

    ddp_utils.d2_synchronize()

    avg_recorder_dict = {}

    for idx, target_image_path in enumerate(data_loader):
      if rank == 0:
        pbar_target.update(world_size)
      target_image_path = Path(target_image_path[0])

      metric_dict = {
        'rank': rank + 1
      }

      start_time = perf_counter()
      # projector.reset()
      sub_outdir = f"{outdir}/{source_image_path.stem}"
      os.makedirs(sub_outdir, exist_ok=True)
      ret_metric_dict = projector.lso_optim_v1(
        outdir=f"{sub_outdir}",
        image_c_path=source_image_path,
        image_s_path=target_image_path,
        proj_c_dir=args.source_proj_dir,
        proj_s_dir=args.target_proj_dir,
        lpips_metric=lpips_metric,
        **global_cfg.lso_optim_kwargs
      )
      metric_dict.update(ret_metric_dict)
      elapsed_time = (perf_counter() - start_time)
      if rank == 0:
        tqdm.tqdm.write(f'processing {target_image_path.stem} elapsed: {elapsed_time:.1f} s')

      metric_dict['elapsed_time_second'] = elapsed_time
      metric_dict['elapsed_time_minute'] = elapsed_time / 60
      # with open(f"{outdir}/metrics.json", 'w') as f:
      #   json.dump(metric_dict, f, indent=2)

      # average value if distributed
      if distributed:
        metric_dict = ddp_utils.d2_reduce_dict(input_dict=metric_dict, average=True)
        ddp_utils.d2_synchronize()

      if rank == 0:
        if len(avg_recorder_dict) == 0:
          for k in metric_dict.keys():
            avg_recorder_dict[k] = AverageMeter()
        for k in metric_dict.keys():
          avg_recorder_dict[k].update(metric_dict[k])

        summary_dict = {}
        loss_str = ""
        for k in avg_recorder_dict.keys():
          v_avg = avg_recorder_dict[k].avg
          summary_dict[k] = v_avg
          loss_str += f'{k}: {v_avg:.4g}, '
        tqdm.tqdm.write(loss_str)

      ddp_utils.d2_synchronize()
      if global_cfg.tl_debug:
        break

    if rank == 0:
      summary_dict['Average_target_images'] = 0
      with open(metric_file, 'w') as f:
        json.dump(summary_dict, f, indent=2)
        # summary_dict2txtfig(summary_dict, prefix='eval', step=idx, textlogger=global_textlogger, save_fig_sec=30)

      # if idx % 20 == 0 and rank == 0:
      #   modelarts_utils.modelarts_sync_results_dir(cfg=global_cfg, join=False)


    if global_cfg.tl_debug:
      break

  if is_main_process:
    modelarts_utils.prepare_dataset(global_cfg.get('modelarts_upload', {}), global_cfg=global_cfg, download=False)
    modelarts_utils.modelarts_sync_results_dir(global_cfg, join=True)
  if distributed: ddp_utils.d2_synchronize()
  pass


if __name__ == '__main__':
  main()
