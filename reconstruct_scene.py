#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Scene reconstruction using MASt3R and SfM pipeline
# --------------------------------------------------------
import argparse
import torch
import os
from pathlib import Path
from PIL import Image
from typing import List
import numpy as np

from mast3r.model import AsymmetricMASt3R
from mast3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from dust3r.utils.image import load_images

def get_argparser():
    parser = argparse.ArgumentParser(description='3D scene reconstruction using MASt3R')
    parser.add_argument('--image_dir', required=True, help='Directory containing input images')
    parser.add_argument('--output_dir', required=True, help='Directory for output reconstruction')
    parser.add_argument('--scene_graph', default='complete', help='Scene graph configuration')
    
    parser_weights = parser.add_mutually_exclusive_group(required=True)
    parser_weights.add_argument("--weights", type=str, help="path to the model weights", default=None)
    parser_weights.add_argument("--model_name", type=str, help="name of the model weights",
                              choices=["MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"])
    
    # Optimization parameters
    parser.add_argument('--lr1', type=float, default=0.07, help='Learning rate for coarse optimization')
    parser.add_argument('--niter1', type=int, default=300, help='Number of iterations for coarse optimization')
    parser.add_argument('--lr2', type=float, default=0.01, help='Learning rate for fine optimization')
    parser.add_argument('--niter2', type=int, default=300, help='Number of iterations for fine optimization')
    
    # Model parameters
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--image_size", type=int, default=512, help="input image size")
    parser.add_argument("--matching_conf_thr", type=float, default=0.0, help="Matching confidence threshold")
    parser.add_argument("--shared_intrinsics", action='store_true', help="Share camera intrinsics across images")
    return parser

def get_image_list(images_path):
    file_list = Path(images_path).rglob("*")
    file_list = sorted(list(file_list))

    image_list = []
    for file_path in file_list:
        try:
            with Image.open(file_path) as im:
                image_list.append(str(file_path))
        except:
            print(f'Skipping invalid image file {file_path}')
            continue
    return image_list

def compute_image_pairs(imgs_fp, scene_graph):
    """Compute image pairs using MASt3R"""
    
    pairs = make_pairs(imgs_fp, scene_graph, prefilter=None, symmetrize=True, sim_mat=None)
    return pairs

def save_depthmaps_to_npy(depthmaps: List[torch.Tensor], confs: List[torch.Tensor], out_files):
    for dm, conf, out_file in zip(depthmaps, confs, out_files):
        dm = dm.reshape(conf.shape)
        dm_arr = dm.detach().cpu().numpy()

        os.makedirs(Path(out_file).parent, exist_ok=True)
        np.save(out_file, dm_arr)
        

def main():
    parser = get_argparser()
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load MASt3R model
    if args.weights is not None:
        weights_path = args.weights
    else:
        weights_path = "naver/" + args.model_name

    print("Loading MASt3R model...")
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(args.device)

    # Load images
    print("Loading images")
    imgs_fp = get_image_list(args.image_dir)
    imgs = load_images(imgs_fp, size=args.image_size)

    # Compute image pairs
    print("Computing image pairs...")
    pairs = compute_image_pairs(imgs, args.scene_graph)

    # Create cache directory for sparse global alignment
    cache_dir = Path(args.output_dir) / 'cache'
    depthmaps_dir = Path(args.output_dir) / "depths"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(depthmaps_dir, exist_ok=True)

    # Run sparse global alignment
    print("Running sparse global alignment...")
    scene = sparse_global_alignment(
        imgs_fp, 
        pairs, 
        str(cache_dir),
        model,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        device=args.device,
        opt_depth=True,
        shared_intrinsics=args.shared_intrinsics,
        matching_conf_thr=args.matching_conf_thr
    )

    # Get the reconstruction results
    print("Extracting reconstruction results...")
    pts3d, depthmaps, confs = scene.get_dense_pts3d(clean_depth=True)
    #focals = scene.get_focals()
    #poses = scene.get_im_poses()

    dm_out_files = [depthmaps_dir / Path(f).relative_to(args.image_dir).with_suffix(".npy") for f in imgs_fp]
    save_depthmaps_to_npy(depthmaps, confs, dm_out_files)
    
    print("Reconstruction completed successfully!")
    print(f"Results cached in {cache_dir}")
    
    return scene

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
    main()
