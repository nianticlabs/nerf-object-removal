import os
import argparse
import json
import cv2
import lpips as lpips_lib
import torch

import numpy as np
import pandas as pd

from PIL import Image

from skimage.metrics import peak_signal_noise_ratio as psnr_
from skimage.metrics import structural_similarity as ssim_
from tqdm import tqdm

def arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_root_dir', default='/mnt/res_nas/silvanweder/experiments/object-removal-custom/synthetic-benchmark')
    parser.add_argument('--experiment', default=None)
    parser.add_argument('--data_path', default='/mnt/res_nas/silvanweder/datasets/object-removal-custom-clean')
    parser.add_argument('--benchmark', default='synthetic')
    parser.add_argument('--benchmark_name', default='syntethic_benchmark')
    parser.add_argument('--run', default='test')
    parser.add_argument('--multiview', action='store_true')

    args = parser.parse_args()
    return args

def load_transforms(file):
    with open(file, 'r') as file:
        transforms = json.load(file)
    return transforms


class LIPPS_FN(object):
    
    def __init__(self):
        self._device = torch.device('cuda:0')
        self._lpips = lpips_lib.LPIPS(net='vgg').to(self._device)

    def __call__(self, pred, target=None, key='image', mask=None):
        with torch.no_grad():
            pred_ = pred[key].copy()
            target_ = target[key]

            if mask is not None:
                pred_[mask == 0] = 0
                target_[mask == 0] = 0
            # normalization taken from here: https://github.com/richzhang/PerceptualSimilarity/blob/31bc1271ae6f13b7e281b9959ac24a5e8f2ed522/util/util.py#L45
            def normalize(x):
                x = torch.from_numpy(x)
                x = x / (255. / 2.) - 1
                x = x.permute(-1, 0, 1).unsqueeze(0)
                return x
            return self._lpips(normalize(pred_).to(self._device), normalize(target_).to(self._device)).detach().cpu().numpy()[0, 0, 0, 0]


def load_groundtruth(path, frame, args):

    frame_id = int(frame['file_path'].split('/')[-1].split('.')[0])
    image_file = os.path.join(path, frame['file_path'])
    depth_file = os.path.join(path, f'depth/depth_{frame_id}.bin')

    if args.benchmark == 'synthetic':
        mask_file = os.path.join(path, f'masks_synthetic/{str(frame_id).zfill(3)}.npy')
    elif args.benchmark == 'real':
        mask_file = os.path.join(path, f'masks_real/{str(frame_id).zfill(3)}.npy')


    image = np.asarray(Image.open(image_file))
    image = cv2.resize(image, (256, 192))
    depth = np.fromfile(depth_file, dtype=np.float32)
    depth = depth.reshape(192, 256)
    mask = np.load(mask_file)

    assert depth.shape[:2] == image.shape[:2]

    meta = {'gt_img_path': image_file,
            'gt_depth_path': depth_file,
            'gt_mask_path': mask_file}

    return {'image': image, 'depth': depth, 'mask': mask}, meta

def load_prediction(path, frame_idx, args):

    frame_id = str(frame_idx).zfill(3)

    if 'ours' in args.experiment:
        if args.multiview:
            image_file = os.path.join(path, f'color_multi_view_{frame_id}.png')
        else:
            image_file = os.path.join(path, f'color_view_dir_{frame_id}.png')
    else:
        image_file = os.path.join(path, f'color_{frame_id}.png')
    depth_file = os.path.join(path, f'distance_mean_{frame_id}.tiff')

    image = np.asarray(Image.open(image_file))
    depth = np.asarray(Image.open(depth_file))

    meta = {'pred_img_path': image_file,
            'pred_depth_path': depth_file,
            'frame_id': frame_id}

    return {'image': image, 'depth': depth}, meta

# metrics
def l1(pred, gt, key='depth', mask=None):
    if mask is None:
        mask = np.ones_like(gt)
    return np.mean(np.abs(pred[key][mask == 1] - gt[key][mask == 1]))

def l2(pred, gt, key='depth', mask=None):
    if mask is None:
        mask = np.ones_like(gt)
    return np.mean(np.power(pred[key][mask == 1] - gt[key][mask == 1], 2))

def ssim(pred, gt, key='image', mask=None):
    if mask is not None:
        pred_ = pred[key].copy()
        gt_ = gt[key]
        pred_[mask == 0] = 0
        gt_[mask == 0] = 0
        return ssim_(pred_, gt_, multichannel=True)

    return ssim_(pred[key], gt[key], multichannel=True)

def psnr(pred, gt, key='image', mask=None):
    if mask is not None:
        pred_ = pred[key].copy()
        gt_ = gt[key]
        pred_[mask == 0] = 0
        gt_[mask == 0] = 0
        return psnr_(gt_, pred_)
    return psnr_(gt[key], pred[key])


def get_metrics(args):
    return {'psnr': psnr,
            'ssim': ssim,
            'lpips': LIPPS_FN(),
            'l1': l1,
            'l2': l2,
            'psnr-mask': psnr,
            'ssim-mask': ssim,
            'lpips-mask': LIPPS_FN(),
            'l1-mask': l1,
            'l2-mask': l2,
            'psnr-mask-weighted': psnr,
            'ssim-mask-weighted': ssim,
            'lpips-mask-weighted': LIPPS_FN(),
            'l1-mask-weighted': l1,
            'l2-mask-weighted': l2, 
            }

def eval_experiment():
    raise NotImplementedError

def main(args):

    if args.experiment is None:
        print('Experiment not specified!')
        print('Available experiments are:')
        for exp in os.listdir(args.experiment_root_dir):
            print('--', exp)
        exit(1)
    experiment_path = os.path.join(args.experiment_root_dir, args.experiment)
    scenes = sorted([sc for sc in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, sc))])
    scenes = [str(i).zfill(3) for i in range(1, 17)]
    results = []
    metrics = get_metrics(args)

    for sc in scenes:
        print(f'Evaluating {sc} ...')
        data_path = os.path.join(args.data_path, sc)
        prediction_path = os.path.join(experiment_path, sc, f'{args.run}_preds')

        if not os.path.exists(prediction_path):
            print(prediction_path)
            print(f'Skipping scene {sc} - results not available')
            continue

        if args.run == 'test':
            transforms_path = os.path.join(experiment_path, sc, f'transforms_{args.run}.json')
        if args.run == 'train_test':
            transforms_path = os.path.join(experiment_path, sc, f'transforms_train.json')
    
        transforms = load_transforms(transforms_path)

        for frame_idx, frame in tqdm(enumerate(sorted(transforms['frames'], key=lambda d: d['file_path'])), total=len(transforms['frames'])):
            groundtruth, meta_gt = load_groundtruth(data_path, frame, args)
            predictions, meta_pred = load_prediction(prediction_path, frame_idx, args)
            results_ = {'scene': sc}
            results_.update(meta_gt)
            results_.update(meta_pred)
            for m_ in metrics.keys():
                if 'mask-weighted' in m_:
                    results_[m_] = np.sum(groundtruth['mask']) * metrics[m_](predictions, groundtruth, mask=groundtruth['mask'])
                    results_['mask_sum'] = np.sum(groundtruth['mask']) 
                elif 'mask' in m_:
                    results_[m_] = metrics[m_](predictions, groundtruth, mask=groundtruth['mask'])
                else:
                    results_[m_] = metrics[m_](predictions, groundtruth)
            results.append(results_)

    results = pd.DataFrame(results)
    results.to_json(os.path.join(experiment_path, f'results.json'))

if __name__ == '__main__':
    args = arg_parser()
    main(args)