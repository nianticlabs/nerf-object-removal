import os
import argparse

import numpy as np
from PIL import Image
from tqdm import tqdm

ROOT = '/mnt/res_nas/silvanweder/datasets/object-removal-custom'
MAX_DEPTH = 5.

def arg_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--scene')
    return parser.parse_args()

def main(args):
    
    # extracts
    if args.scene is None:
        scenes = os.listdir(args.root_dir)
        scenes = [sc for sc in scenes if os.path.isdir(os.path.join(args.root_dir, sc))]
    else:
        scenes = [args.scene]

    for sc in scenes:
            
        print(f'Processing scene {sc}')
        output_dir = os.path.join(args.output_dir, sc, f'lama_depth_input{args.suffix}')
        os.makedirs(output_dir, exist_ok=True)
        
        frames = [f.replace('.npy', '.bin') for f in os.listdir(os.path.join(args.root_dir, f'masks{args.suffix}'))]
        for frame in tqdm(frames, total=len(frames)):
            try:
                depth_file = os.path.join(args.root_dir, 'depth', 'depth_' + str(int(frame.split('.')[0])) + '.bin')
                target_file = os.path.join(output_dir, frame.replace('bin', 'png'))

                depth = np.fromfile(depth_file, dtype='float32')
                depth = depth.reshape(192, 256)
                depth = depth / MAX_DEPTH
                depth = (depth.clip(0, 1) * 255).astype(np.uint8)
                depth = np.stack((depth, depth, depth), axis=-1)

                mask_file = frame.replace('.bin', '.npy')
                mask_file = os.path.join(args.root_dir, f'masks{args.suffix}', mask_file)

                # copy image
                Image.fromarray(depth).save(target_file)
                # preprocess and copy mask
                target_file = target_file.replace('.png', '_mask001.png')
                mask = np.load(mask_file)
                Image.fromarray((mask * 255).astype(np.uint8), mode='L').save(target_file)

            except Exception as e:
                print(e)


if __name__ == '__main__':
    main(arg_parse())