import os
import tqdm

from tl2.proj.argparser import argparser_utils
from tl2.launch.launch_utils import TLCfgNode, global_cfg, set_global_cfg

from exp2.hrinversion.scripts import align_raw_faces
from exp2.hrinversion.scripts import project_image_list
from exp2.hrinversion.models.projector import StyleGAN2Projector


def main(raw_list,
         name_list,
         outdir,
         cfg_file,
         command,
         debug):

  assert len(raw_list) == len(name_list)

  # align raw images
  aligned_image_list = align_raw_faces.main(raw_list=raw_list, outdir=outdir)

  cfg = TLCfgNode.load_yaml_with_command(cfg_filename=cfg_file, command=command)

  # project images
  w_file_list = project_image_list.main(cfg=cfg.proj_image_list,
                                        aligned_img_list=aligned_image_list,
                                        outdir=outdir,
                                        debug=debug)

  cfg.tl_debug = debug
  set_global_cfg(cfg)

  mixer = StyleGAN2Projector(
    network_pkl=cfg.network_pkl[cfg.default_network_pkl],
    loss_cfg=None)

  mixer.lerp_image_list(
    outdir=outdir,
    w_file_list=w_file_list,
    author_name_list=name_list,
    st_web=False,
    **cfg.kwargs
  )
  print(f"Save to {outdir}/author_list.gif")


  pass



if __name__ == '__main__':
  parser = argparser_utils.get_parser()

  argparser_utils.add_argument_list_of_str(parser, name='raw_list', )
  argparser_utils.add_argument_list_of_str(parser, name='name_list', )
  argparser_utils.add_argument_str(parser, name='outdir', default='results')
  argparser_utils.add_argument_str(parser, name='cfg_file', default='')
  argparser_utils.add_argument_str(parser, name='command', default='')
  argparser_utils.add_argument_bool(parser, name='debug')

  args, _ = parser.parse_known_args()
  argparser_utils.print_args(args)

  main(**vars(args))

