import os
import argparse

ROOT = '/mnt/res_nas/silvanweder/datasets/object-removal-custom'


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rotate', action='store_true')
    parser.add_argument('--resize', action='store_true')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--root_dir', type=str)
    parser.add_argument('--scene', default=None)

    return parser.parse_args()

def main(args):
 
    # extracts
    if args.scene is not None:
        scenes = [args.scene]
    else:
        scenes = os.listdir(args.root_dir)
        scenes = [sc for sc in scenes if os.path.isdir(os.path.join(args.root_dir, sc))]

    rotate = 'False' if not args.rotate else 'True'
    resize = 'False' if not args.resize else 'True'

    for sc in scenes:
        os.system(f'python3 bin/predict.py model.path=$(pwd)/big-lama indir={args.root_dir}/{sc}/lama_images_input{args.suffix} outdir={args.root_dir}/{sc}/lama_images_output{args.suffix} dataset.rotate={rotate} dataset.resize={resize}')


if __name__ == '__main__':
    main(arg_parser())