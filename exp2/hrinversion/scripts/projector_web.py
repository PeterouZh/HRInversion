import pathlib
import collections
from pathlib import Path
import logging
import os
import sys
from PIL import Image
import streamlit as st
from time import perf_counter

sys.path.insert(0, os.getcwd())

from tl2.launch.launch_utils import global_cfg, TLCfgNode, set_global_cfg
from tl2.proj.streamlit import SessionState
from tl2.proj.streamlit import st_utils
from tl2.proj.logger.logger_utils import get_file_logger
from tl2.proj.logger import logging_utils_v2
from tl2 import tl2_utils
from tl2.proj.streamlit import st_utils
from tl2.proj.fvcore import build_model, MODEL_REGISTRY
from tl2.proj.argparser import argparser_utils
from tl2.proj.pytorch import torch_utils


# @MODEL_REGISTRY.register(name_prefix=__name__)
class STModel(object):
  def __init__(self):

    pass

  def projector_web(self,
                    cfg,
                    outdir,
                    **kwargs):

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

    loss_cfg = cfg.loss_cfg['vgg16_conv_r1024_loss_cfg']
    downsampling = st_utils.checkbox('downsampling', value=True, sidebar=True)
    if downsampling:
      loss_cfg.downsample_size = 256

    if not global_cfg.tl_debug:
      if not st.sidebar.button("run_web"):
        return

    # ****************************************************************************

    # np.random.seed(seed)
    # torch.manual_seed(seed)
    torch_utils.init_seeds(seed=303, rank=0)

    class_id = None

    from exp2.hrinversion.models.projector import StyleGAN2Projector

    projector = StyleGAN2Projector(
      network_pkl=cfg.network_pkl.FFHQ_r1024,
      loss_cfg=loss_cfg)

    start_time = perf_counter()
    projector.project_wplus(
      outdir=outdir,
      image_path=image_path,
      class_id=class_id,
      st_web=True,
      **cfg.projector_web
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    pass


def main(cfg_file,
         command,
         outdir,
         debug=False):

  cfg = TLCfgNode.load_yaml_with_command(cfg_filename=cfg_file, command=command)
  cfg.tl_debug = debug
  set_global_cfg(cfg)

  # outdir
  kwargs = {}
  kwargs['outdir'] = outdir
  os.makedirs(outdir, exist_ok=True)

  get_file_logger(filename=f"{outdir}/log.txt", logger_names=['st'])
  logging_utils_v2.get_logger(filename=f"{outdir}/log.txt")
  logger = logging.getLogger('tl2')
  logger.info(f"global_cfg:\n{global_cfg.dump()}")

  st_model = STModel()

  st_model.projector_web(cfg=global_cfg.get('projector_web', {}), **kwargs)

  pass

if __name__ == '__main__':

  parser = argparser_utils.get_parser()

  argparser_utils.add_argument_str(parser, name="cfg_file", default="")
  argparser_utils.add_argument_str(parser, name="command", default="projector_web")
  argparser_utils.add_argument_str(parser, name="outdir", default="results")
  argparser_utils.add_argument_bool(parser, name='debug', default=False)

  args, _ = parser.parse_known_args()
  argparser_utils.print_args(args)

  main(**vars(args))
