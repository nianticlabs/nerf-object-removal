import os
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

root = '/mnt/res_nas/silvanweder/datasets/object-removal-custom'

def arg_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--scene', default=None)
    return parser.parse_args()

def main(args):

    # extracts
    if args.scene is None:
        scenes = os.listdir(args.root_dir)
        scenes = [sc for sc in scenes if os.path.isdir(os.path.join(args.root_dir, sc))]
    else:
        scenes = [args.scene]

    for sc in scenes:
        if not os.path.isdir(os.path.join(args.root_dir)):
            continue
            
        print(f'Processing scene {sc}')

        output_dir = os.path.join(args.output_dir, sc, f'lama_images_input{args.suffix}')
        os.makedirs(output_dir, exist_ok=True)
        
        frames = [f.replace('.npy', '.jpg') for f in os.listdir(os.path.join(args.root_dir, f'masks{args.suffix}'))]
        for frame in tqdm(frames, total=len(frames)):
            try:
                image_file = os.path.join(args.root_dir, 'images', frame)
                target_file = os.path.join(output_dir, frame.replace('jpg', 'png'))
                mask_file = frame.replace('.jpg', '.npy')
                mask_file = os.path.join(args.root_dir, f'masks{args.suffix}', mask_file)

                # copy image
                Image.open(image_file).save(target_file)

                # preprocess and copy mask
                target_file = target_file.replace('.png', '_mask001.png')
                mask = np.load(mask_file)
                Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(target_file)
            except Exception as e:
                print(e)


if __name__ == '__main__':
    main(arg_parse())