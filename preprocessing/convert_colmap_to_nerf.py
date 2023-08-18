#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import argparse
import os
from pathlib import Path, PurePosixPath

import numpy as np
import json
import sys
import math
import cv2
import os
import shutil
import random
import open3d as o3d

from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="convert a text colmap export to nerf format transforms.json; optionally convert video to images, and optionally run colmap in the first place")

    parser.add_argument("--video", default="", help="input path to the video")
    parser.add_argument("--images", default="", help="input path to the images folder, ignored if --video is provided")
    parser.add_argument("--run_colmap", action="store_true", help="run colmap first on the image folder")

    parser.add_argument("--video_fps", default=3)
    parser.add_argument("--time_slice", default="", help="time (in seconds) in the format t1,t2 within which the images should be generated from the video. eg: \"--time_slice '10,300'\" will generate images only from 10th second to 300th second of the video")

    parser.add_argument("--colmap_matcher", default="exhaustive", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="select which matcher colmap should use. sequential for videos, exhaustive for adhoc images")
    parser.add_argument("--aabb_scale", default=2, choices=["1","2","4","8","16"], help="large scene scale factor. 1=scene fits in unit cube; power of 2 up to 16")
    parser.add_argument("--skip_early", default=0, help="skip this many images from the start")

    parser.add_argument("--colmap_text", default="colmap_text", help="input path to the colmap text files (set automatically if run_colmap is used)")
    parser.add_argument("--colmap_db", default="colmap.db", help="colmap database filename")
    parser.add_argument("--out", default="transforms.json", help="output path")
    parser.add_argument('--split', action='store_true')
    parser.add_argument('--split_fraction', default=0.1)
    parser.add_argument('--output_path')
    parser.add_argument('--scene')
    parser.add_argument('--sequence')
    parser.add_argument('--data_path')
    parser.add_argument('--depth_h', type=int, default=192)
    parser.add_argument('--depth_w', type=int, default=256)

    args = parser.parse_args()
    return args



def main(args):
    
    image_path = os.path.join(args.data_path, 'images')
    rel_path = ''
    sequence_path = os.path.join(args.data_path)
    args.colmap_text = os.path.join(sequence_path, 'sparse')
    args.out = os.path.join(sequence_path, args.out)

    AABB_SCALE = int(args.aabb_scale)
    TEXT_FOLDER = args.colmap_text
    OUT_PATH = args.out
    META_PATH = os.path.join(sequence_path, 'boxes/meta.json')

    # read metadata for scene
    with open(META_PATH, 'r') as file:
        meta = json.load(file)

    center = np.asarray(meta['center'])
    scale = 2. / meta['scale']
    scale_matrix = np.eye(4)
    np.fill_diagonal(scale_matrix[:3, :3], scale)

    START_STOP_PATH = os.path.join(sequence_path, 'meta.json')
    with open(START_STOP_PATH, 'r') as file:
        start_stop = json.load(file)
    split_point = start_stop['split_end'] 

    
    print(f"outputting to {OUT_PATH}...")
    with open(os.path.join(TEXT_FOLDER, "cameras.txt"), "r") as f:
        angle_x = math.pi / 2
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            w = float(els[2])
            h = float(els[3])
            fl_x = float(els[4])
            fl_y = float(els[4])
            k1 = 0
            k2 = 0
            p1 = 0
            p2 = 0
            cx = w / 2
            cy = h / 2
            
            scale_w = float(args.depth_w) / w
            scale_h = float(args.depth_h) / h

            if els[1] == "SIMPLE_PINHOLE":
                cx = float(els[5])
                cy = float(els[6])
            elif els[1] == "PINHOLE":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
            elif els[1] == "RADIAL":
                cx = float(els[5])
                cy = float(els[6])
                k1 = float(els[7])
                k2 = float(els[8])
            elif els[1] == "OPENCV":
                fl_y = float(els[5])
                cx = float(els[6])
                cy = float(els[7])
                k1 = float(els[8])
                k2 = float(els[9])
                p1 = float(els[10])
                p2 = float(els[11])
            else:
                print("unknown camera model ", els[1])
            

            fl_y = scale_h * fl_y
            fl_x = scale_w * fl_x
            
            cx = scale_w * cx
            cy = scale_h * cy
            
            w = args.depth_w
            h = args.depth_h

            # fl = 0.5 * w / tan(0.5 * angle_x);
            angle_x = math.atan(w / (fl_x * 2)) * 2
            angle_y = math.atan(h / (fl_y * 2)) * 2
            fovx = angle_x * 180 / math.pi
            fovy = angle_y * 180 / math.pi

    print(f"camera:\n\tres={w,h}\n\tcenter={cx,cy}\n\tfocal={fl_x,fl_y}\n\tfov={fovx,fovy}\n\tk={k1,k2} p={p1,p2} ")

    with open(os.path.join(TEXT_FOLDER, "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        out = {
            "camera_angle_x": angle_x,
            "camera_angle_y": angle_y,
            "scale_matrix": scale_matrix.tolist(),
            "center": center.tolist(),
            "fl_x": fl_x,
            "fl_y": fl_y,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "aabb_scale": AABB_SCALE,
            "frames": [],
        }

        up = np.zeros(3)
        for line in f:
            line = line.strip()

            # this is needed to skip the empty key points
            if len(line) == 0:
                i = i + 1
                continue
            if line[0] == "#":
                continue
            i = i + 1
            if  i % 2 == 1:
                elems = line.split(" ") # 1-4 is quat, 5-7 is trans, 9ff is filename (9, if filename contains no spaces)
                name = '_'.join(elems[9:])
                full_name = os.path.join(image_path, name)
                rel_name = rel_path + full_name[len(os.path.join(args.data_path)):] # +1 to remove the leading slash when working with relative paths
                b = sharpness(full_name)
                print(name, "sharpness =",b)

                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3,1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)

                c2w = np.linalg.inv(m)

                # center shift and scaling
                c2w[:3, 3] -= center
                c2w = scale_matrix @ c2w

                c2w[0:3, 2] *= -1 # flip the y and z axis
                c2w[0:3, 1] *= -1
                c2w = c2w[[1, 0, 2, 3],:] # swap y and z
                c2w[2, :] *= -1 # flip whole world upside down

                up += c2w[0:3, 1]

                frame = {"file_path" : rel_name, "sharpness" : b, "transform_matrix" : c2w}
                out["frames"].append(frame)
    
    # I would remove that
    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up, [0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R, [0, 1])
    R[-1, -1] = 1

    out['up_rotation'] = R.tolist()

    for f in out["frames"]:
        f["transform_matrix"] = np.matmul(R, f["transform_matrix"]) # rotate up to be the z axis

    for f in out["frames"]:
        f["transform_matrix"] = f["transform_matrix"].tolist()

    start_end_meta = json.load(open(sequence_path + '/meta.json', 'r'))


    # convert object bounding box
    box = o3d.io.read_triangle_mesh(os.path.join(sequence_path, 'boxes/bbox.ply'))
    vertices = np.asarray(box.vertices)
    rotation_up_matrix = np.asarray(out['up_rotation'])
    scale_matrix = np.asarray(out['scale_matrix'])
    center_shift = np.asarray(out['center'])

    # # transform box into nerf 
    vertices = vertices - center_shift
    vertices = (scale_matrix[:3, :3] @ vertices.T).T
    vertices = vertices[:, [1, 0, 2]]
    vertices[:, 2] *= -1
    vertices = (rotation_up_matrix[:3, :3] @ vertices.T).T
    np.savetxt(os.path.join(sequence_path, 'boxes/bbox_nerf.txt'), vertices)


    if args.split:
        test_split = {}
        train_split = {}

        for k, v in out.items():

            if k == 'frames':
                train_split[k] = v[:start_end_meta['split_start']]
                test_split[k] = v[start_end_meta['split_end']:]
            else:
                test_split[k] = v
                train_split[k] = v

        print(f"writing {len(train_split['frames'])} train frames to {OUT_PATH.replace('.json', '_train.json')}")
        with open(OUT_PATH.replace('.json', '_train.json'), "w") as outfile:
            json.dump(train_split, outfile, indent=2)
    
        print(f"writing {len(test_split['frames'])} test frames to {OUT_PATH.replace('.json', '_test.json')}")
        with open(OUT_PATH.replace('.json', '_test.json'), "w") as outfile:
            json.dump(test_split, outfile, indent=2)
    else:
        pass
        print(f"writing {OUT_PATH}")
        with open(OUT_PATH, "w") as outfile:
            json.dump(out, outfile, indent=2)


if __name__ == "__main__":
    args = parse_args()
    main(args)


    