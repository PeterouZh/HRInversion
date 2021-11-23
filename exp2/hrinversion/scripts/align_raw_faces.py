import os
import tqdm

from tl2.proj.dlib.ffhq_face_align import align_images
from tl2.proj.argparser import argparser_utils


LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'

def main(raw_list,
         outdir,
         landmark_model_bz2="datasets/pretrained/shape_predictor_68_face_landmarks.dat.bz2",
         output_size=1024):

  aligned_image_list = []

  for image_path in tqdm.tqdm(raw_list, desc=outdir):
    image_path = image_path.strip()

    align_image_path = align_images.get_saved_align_image_path(
      outdir=outdir, image_path=image_path, idx=0)
    aligned_image_list.append(align_image_path)

    if not os.path.exists(align_image_path):
      align_images.align_face(image_path=image_path,
                              outdir=outdir,
                              output_size=output_size,
                              landmark_model_bz2=landmark_model_bz2)

  return aligned_image_list


if __name__ == '__main__':

  parser = argparser_utils.get_parser()

  argparser_utils.add_argument_list_of_str(parser, name='raw_list', )
  argparser_utils.add_argument_str(parser, name='outdir', default='results')

  args, _ = parser.parse_known_args()
  argparser_utils.print_args(args)

  main(raw_list=args.raw_list, outdir=args.outdir)

