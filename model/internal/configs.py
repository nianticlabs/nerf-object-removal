# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This is a heavily modified file from RegNeRF.
# Original file: https://github.com/google-research/google-research/blob/master/regnerf/internal/configs.py

"""Utility functions."""
import dataclasses
import os
from typing import Any, Callable

from absl import flags
import flax
import gin
import jax
import jax.numpy as jnp

gin.add_config_file_search_path('../')

gin.config.external_configurable(jnp.reciprocal, module='jnp')
gin.config.external_configurable(jnp.log, module='jnp')
gin.config.external_configurable(jnp.sqrt, module='jnp')
gin.config.external_configurable(jax.nn.relu, module='jax.nn')
gin.config.external_configurable(jax.nn.softplus, module='jax.nn')
gin.config.external_configurable(
    jax.nn.initializers.glorot_uniform(),
    module='jax.nn.initializers.glorot_uniform')
gin.config.external_configurable(
    jax.nn.initializers.he_uniform(), module='jax.nn.initializers.he_uniform')
gin.config.external_configurable(
    jax.nn.initializers.glorot_normal(),
    module='jax.nn.initializers.glorot_normal')
gin.config.external_configurable(
    jax.nn.initializers.he_normal(), module='jax.nn.initializers.he_normal')


@gin.configurable()
@dataclasses.dataclass
class Config:
  """Configuration flags for everything."""
  dataset_loader: str = 'dtu'  # The type of dataset loader to use.
  batching: str = 'single_image'  # Batch composition.
  batching_random: str = 'all_images'  # Batch composiiton for random views.
  batch_size: int = 4096  # The number of rays/pixels in each batch.
  batch_size_random: int = 4096  # The number of rays/pixels in each batch.
  factor: int = 0  # The downsample factor of images, 0 for no downsampling.
  render_factor: int = 0  # The factor for rendering.
  remap_to_hemisphere: bool = False  # Set to True for spherical 360 scenes.
  render_path: bool = False  # If True, render a path. Used only by LLFF.
  render_train: bool = False  # If True, renders train images instead.
  render_path_frames: int = 240  # Number of frames in path. Used only by LLFF.
  llffhold: int = 8   # Use every Nth image for the test set. Used only by LLFF.
  dtuhold: int = 8  # Use every Nth image for the test set. Used only by DTU.
  dtu_light_cond: int = 3  # Light condition. Used only by DTU.
  dtu_max_images: int = 49  # Whether to restrict the max number of images.
  dtu_split_type: str = 'pixelnerf'  # Which train/test split to use.
  use_tiffs: bool = False  # If True, use 32-bit TIFFs. Used only by Blender.
  compute_disp_metrics: bool = False  # If True, load and compute disparity MSE.
  compute_normal_metrics: bool = False  # If True, load and compute normal MAE.
  load_depth: bool = False # If True, depth is loaded and used to compute the midpoint for annealing
  load_inpainted_depth: bool = False
  load_masks: bool = False # If True, inpainting mask is loaded and available in batch
  lr_init: float = 5e-4  # The initial learning rate.
  lr_final: float = 5e-5  # The final learning rate.
  lr_delay_steps: int = 0  # The number of "warmup" learning steps.
  lr_delay_mult: float = 0.0  # How much sever the "warmup" should be.
  resample_padding_init: float = 0.01  # Padding when training starts, > 0.
  resample_padding_final: float = 0.01  # Padding when training ends, > 0.
  grad_max_norm: float = 0.0  # Gradient clipping magnitude, disabled == 0.
  grad_max_val: float = 0.0  # Gradient clipping value, disabled if == 0.
  gc_every: int = 10000  # The number of steps between garbage collections.
  disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
  randomized: bool = True  # Use randomized stratified sampling.
  near: float = 2.  # Near plane distance.
  far: float = 6.  # Far plane distance.
  near_origin: float = 0.0  # Near plane for origin adjustment (DTU).
  coarse_loss_mult: float = 0.1  # How much to downweight the coarse loss(es).
  weight_decay_mult: float = 0.  # The multiplier on weight decay.
  white_background: bool = True  # If True, use white as background, black ow.
  checkpoint_dir: str = None  # Where to store checkpoints and logs.
  render_dir: str = None  # Where to store output rendered path.
  data_dir: str = None  # Input data directory.
  render_chunk_size: int = 4096  # Chunk size for whole-image renderings.
  num_showcase_images: int = 5  # The number of test-set images to showcase.
  deterministic_showcase: bool = True  # If True, showcase the same images.
  vis_dist_curve_fn: Callable[Ellipsis, Any] = lambda x: x  # Curve for t_vals.
  vis_num_rays: int = 64  # The number of rays to visualize.
  dtu_scan: str = 'scan114'  # ID of considered scanID.
  llff_scan: str = 'fern'  # Which LLFF scan to use.
  blender_scene: str = 'lego'  # Which blender scene to use.
  dtu_mask_path: str = None  # DTU mask data directory.
  filter_masks: bool = False
  # scene: str = ''
  frame_list: list = None

  stop_gradient_density: bool = False

  train_list: str = 'transforms_train.json'
  test_list: str = 'transforms_test.json'
  
  feature_path: str = 'features'
  mask_path: str = 'masks'
  inpainting_path: str = '../lama_images_output'
  depth_path: str = 'depth'
  inpainted_depth_path: str = '../lama_depth_output'

  filter_invalid_masks: bool = False
  rgb_loss_mask: str = 'masked'
  rgb_loss_weight: float = 1.0

  train_inpainting: bool = False
  train_inpainting_no_density_grad: bool = False
  load_inpainting: bool = False
  inpainting_loss_mask: str = 'masked'
  inpainting_loss_weight: float = 0.1
  inpainting_sub_freq: int = 1
  inpainting_depth_factor: float = 0.1

  # confidence training
  train_confidence: bool = False
  use_confidence_for_depth: bool = False
  confidence_lambda: float = 0.01
  filter_views_every: int = 50000
  filter_view_conf: bool = False
  filter_view_conf_th: float = 0.
  filter_threshold_percentile: int = 50
  reset_lr: bool = False
  hard_filter: bool = False
  reset_variables: bool = False
  reset_variables_th: int = 100000
  filter_stop: int = 300000
  reset_variables_in_filter: bool = False

  # geometric supervision
  train_depth: bool = False # wether to supervise the depth
  depth_loss_weight: float = 1.
  depth_loss_mask: str = 'all'
  load_depth_confidence: bool = False
  filter_confidence: bool = False
  code_size: int = 1

  # label training
  train_label: bool = False
  label_loss_weight: float = 1.0
  train_label_noise: bool = False
  train_label_noise_p: float = 0.4
  train_label_3d: bool = False

  view_uncertainty_code: bool = True

  ray_constraint: bool = False
  hard_constraint: bool = True

  train_dist_reg: bool = False
  dist_reg_loss_weight: float = 0.01
  dist_reg_mask_weight_ratio: float = 0.1

  train_uncertainty_mask: bool = False
  train_uncertainty_inpainting: bool = False

  # New loss function weights
  depth_tvnorm_loss_mult: float = 0.0  # Loss weight of depth tv norm.
  depth_tvnorm_selector: str = 'distance_mean_save'  # Selector for tv depth.
  random_scales: int = 1  # Scales for random patch sampling (default is 1).
  random_scales_init: int = 0  # Init scale for random patch sampling.
  dietnerf_loss_mult: float = 0.0  # Loss mult for diet nerf regularizer.
  dietnerf_loss_resolution: int = 96  # Resolution for dietnerf loss.
  dietnerf_loss_every: int = 10  # Apply loss every x iteration.
  depth_tvnorm_decay: bool = False  # Whether to decay tvnorm.
  depth_tvnorm_maxstep: int = 0  # Max step for depth tv norm decay.
  depth_tvnorm_loss_mult_start: float = 0.0  # End loss weight for tv depth.
  depth_tvnorm_loss_mult_end: float = 0.0  # End loss weight for tv depth.
  # Weight for tvnorm mask (0.0 = disabled).
  depth_tvnorm_mask_weight: float = 0.0
  flow_loss_mult: float = 0.0  # Loss weight for flow-based loss.
  depth_tvnorm_type: str = 'l2'  # Type of depth tv norm loss.
  recon_loss_scales: int = 1  # How many scales to apply reconstruction loss.
  sample_reconscale_dist: str = 'uniform_scale'  # Type of recon scale dist.
  


  # Only used by train.py:
  max_steps: int = 250000  # The number of optimization steps.
  checkpoint_every: int = 25000  # The number of steps to save a checkpoint.
  print_every: int = 100  # The number of steps between reports to tensorboard.
  train_render_every: int = 10000  # Steps between test set renders for training
  n_input_views: int = 9  # Restrict the number of input views.
  n_random_poses: int = 10000  # How many random poses to use.
  load_random_rays: bool = True  # Whether to load random rays.
  anneal_nearfar: bool = False  # Whether to anneal near/far planes.
  anneal_nearfar_steps: int = 2000  # Steps for near/far annealing.
  anneal_nearfar_perc: float = 0.2  # Percentage for near/far annealing.
  anneal_mid_perc: float = 0.5  # Perc for near/far mid point.
  random_pose_type: str = 'renderpath'  # Type of random poses.
  random_pose_focusptjitter: bool = True  # Whether to use focus pt jitter.
  random_pose_radius: float = 1.0  # Radius of random pose sampling.
  random_pose_add_test_poses: bool = False  # Whether to add test poses.
  check_grad_for_nans: bool = False  # Whether to check grad or NaNs.
  maxdeg_val: int = 16  # Max positional encoding degree.
  maxdeg_steps: int = 0  # Steps for reaching max-value of deg-scaling.
  filter_good_inpaintings: bool = False

  # Only used by eval.py:
  eval_only_once: bool = True  # If True evaluate the model only once, ow loop.
  eval_save_output: bool = True  # If True save predicted images to disk.
  eval_render_interval: int = 1  # The interval between images saved to disk.
  eval_disable_lpips: bool = True  # If True, disable LPIPS computation.
  dtu_no_mask_eval: bool = False  # Set true for evaluation without masks.


def define_common_flags():
  # Define the flags used by both train.py and eval.py
  flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
  flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')
  flags.DEFINE_multi_string('scene', None, 'Scene name')
  flags.DEFINE_multi_string('data_dir', None, 'Data dir')
  flags.DEFINE_multi_string('checkpoint_dir', None, 'Checkpoint dir')


def load_config(save_config=True):
  """Loads config."""

  gin.parse_config_files_and_bindings(
      flags.FLAGS.gin_configs, flags.FLAGS.gin_bindings, flags.FLAGS.scene, skip_unknown=True)
  config = Config()

  if flags.FLAGS.checkpoint_dir is not None:
    config.checkpoint_dir = flags.FLAGS.checkpoint_dir[0]
  else:
    config.checkpoint_dir = config.checkpoint_dir + '/' + flags.FLAGS.scene[0]
  if flags.FLAGS.data_dir is not None:
    config.data_dir = flags.FLAGS.data_dir[0]
  else:
    config.data_dir = config.data_dir + '/' + flags.FLAGS.scene[0]

  if save_config and jax.host_id() == 0:
    os.makedirs(config.checkpoint_dir, exist_ok=True) # if it exists it's ok, load checkpoints from it
    with open(config.checkpoint_dir + '/config.gin', 'w') as f:
      f.write(gin.config_str())
  return config



