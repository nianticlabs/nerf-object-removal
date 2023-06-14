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
# Original file: https://github.com/google-research/google-research/blob/master/regnerf/train.py

import functools
import gc
import time
import pdb
import os
import glob

from absl import app
import flax

from internal import configs, datasets, math, models, utils, vis  # pylint: disable=g-multiple-import
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
from skimage.metrics import structural_similarity
from jax.experimental.host_callback import id_print
from jax.config import config
from flax.metrics import tensorboard
from flax.training import checkpoints

# config.update('jax_debug_nans', True)

configs.define_common_flags()
jax.config.parse_flags_with_absl()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.


@flax.struct.dataclass
class TrainStats:
  """Collection of stats for logging."""
  loss: float
  losses: float
  losses_georeg: float
  losses_depth: float
  losses_label: float
  losses_inpainting: float
  losses_dist_reg: float
  disp_mses: float
  normal_maes: float
  weight_l2: float
  psnr: float
  psnrs: float
  grad_norm: float
  grad_abs_max: float
  grad_norm_clipped: float
  conf_max: float
  conf_min: float
  alpha_min: float
  alpha_max: float
  # grad_norm_latent: float
  # grad_abs_max_latent: float
  # grad_norm_clipped_latent: float
  max_logit: float
  min_logit: float
  max_violation: float

def tree_sum(tree):
  return jax.tree_util.tree_reduce(lambda x, y: x + y, tree, initializer=0)


def tree_norm(tree):
  return jnp.sqrt(tree_sum(jax.tree_map(lambda x: jnp.sum(x**2), tree)))


def train_step(
    model,
    config,
    rng,
    state,
    batch,
    learning_rate,
    resample_padding,
    tvnorm_loss_weight,
    latent_codes,
):
  """One optimization step.

  Args:
    model: The linen model.
    config: The configuration.
    rng: jnp.ndarray, random number generator.
    state: utils.TrainState, state of the model/optimizer.
    batch: dict, a mini-batch of data for training.
    learning_rate: float, real-time learning rate.
    resample_padding: float, the histogram padding to use when resampling.
    tvnorm_loss_weight: float, tvnorm loss weight.

  Returns:
    A tuple (new_state, stats, rng) with
      new_state: utils.TrainState, new training state.
      stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
      rng: jnp.ndarray, updated random number generator.
  """
  rng, key, key2, key_noise = random.split(rng, 4)

  def loss_fn(variables, latent_codes, box=None, step=None):
    
    weight_l2 = (
        tree_sum(jax.tree_map(lambda z: jnp.sum(z**2), variables)) / tree_sum(
            jax.tree_map(lambda z: jnp.prod(jnp.array(z.shape)), variables)))

    renderings = model.apply(
        variables,
        key if config.randomized else None,
        batch['rays'],
        resample_padding=resample_padding,
        compute_extras=(config.compute_disp_metrics or
                        config.compute_normal_metrics or 
                        config.train_depth))
    lossmult = batch['rays'].lossmult
    if config.disable_multiscale_loss:
      lossmult = jnp.ones_like(lossmult)

    losses = []
    disp_mses = []
    normal_maes = []

    losses_mse = []
    losses_depth = []
    losses_data_unc = []
    losses_label = []
    losses_dist_reg = []
    losses_inpainting = []

    max_logit = []
    min_logit = []

    n_levels = len(renderings)

    for nl, rendering in enumerate(renderings):

      rgb_mask = batch['masks'] if config.rgb_loss_mask == 'masked' else jnp.zeros_like(batch['masks'])
      rgb_mask = rgb_mask == 0

      # here the gradient must flow through density as well
      mse_view_dir = (jnp.expand_dims(rgb_mask, axis=-1) * (rendering['rgb_view_dir'] - batch['rgb'][Ellipsis, :3])**2).sum() / jnp.clip(rgb_mask.sum(), a_min=1)
      mse_multi_view = (jnp.expand_dims(rgb_mask, axis=-1) * (rendering['rgb_multi_view'] - batch['rgb'][Ellipsis, :3])**2).sum() / jnp.clip(rgb_mask.sum(), a_min=1)

      mse = mse_view_dir + mse_multi_view

      numer_view_dir = (jnp.expand_dims(rgb_mask, axis=-1) * (rendering['rgb_view_dir'] - batch['rgb'][Ellipsis, :3])**2).sum() / jnp.clip(rgb_mask.sum(), a_min=1)
      numer_multi_view = (jnp.expand_dims(rgb_mask, axis=-1) * (rendering['rgb_multi_view'] - batch['rgb'][Ellipsis, :3])**2).sum() / jnp.clip(rgb_mask.sum(), a_min=1)

      numer = numer_view_dir + numer_multi_view

      if nl < n_levels - 1:
        losses.append(numer)
        losses_mse.append(mse)
      else:
        losses.append(numer)
        losses_mse.append(mse)

      if config.train_inpainting:

        # here density gradient must not flow through 
        inpainting_mask = batch['inpainting_mask'] if config.inpainting_loss_mask == 'masked' else jnp.ones_like(batch['masks'])
        inpainting_mask = inpainting_mask == 1

        mse_inp_view_dir = (jnp.expand_dims(inpainting_mask, axis=-1) * (rendering['rgb_view_dir_no_density_grad'] - batch['inpainting'][Ellipsis, :3])**2).sum(axis=-1)
        mse_inp_multi_view = (jnp.expand_dims(inpainting_mask, axis=-1) * (rendering['rgb_multi_view_no_density_grad'] - batch['inpainting'][Ellipsis, :3])**2).sum(axis=-1)

        latent_codes_ = latent_codes[:, 0]
        
        confidence_ = jnp.exp(-1 * latent_codes_)
        confidence_mask_ = jax.lax.stop_gradient(latent_codes[:, 1]) # we do not want to change the mask with the gradients
        confidence_ = confidence_.squeeze()

        assert mse_inp_multi_view.shape == confidence_.shape
        assert inpainting_mask.shape == latent_codes_.shape
        
        # update mask
        inpainting_mask = inpainting_mask & (confidence_mask_ == 1)

        # multi view training
        mse_inp_multi_view = confidence_ * mse_inp_multi_view
        mse_inp_multi_view = mse_inp_multi_view.sum() / jnp.clip(inpainting_mask.sum(), a_min=1) + config.confidence_lambda * (jax.nn.relu(latent_codes_) * inpainting_mask.squeeze()).sum() / jnp.clip(inpainting_mask.sum(), a_min=1)
        
        mse_inp_view_dir = (inpainting_mask * mse_inp_view_dir).sum() / jnp.clip(inpainting_mask.sum(), a_min=1)
        mse_inp = mse_inp_view_dir + mse_inp_multi_view 
        losses_inpainting.append(mse_inp)

      if config.train_depth:
        if config.depth_loss_mask == 'all':
          if config.filter_confidence:
            valid_mask = batch['confidence'] == 2

          depth_gt = jnp.where(batch['inpainting_mask'] == 1, batch['depth_inpainted'], batch['depth'])

          ld = (lossmult[:, 0] * jnp.abs((rendering['distance_mean'] - depth_gt[Ellipsis])))

          latent_codes_ = latent_codes[:, 0]
          confidence_ = jnp.exp(-1 * latent_codes_)
          confidence_mask_ = jax.lax.stop_gradient(latent_codes[:, 1])

          # have depth supervision either when not masked and good depth value or mask and view is selected
          non_masked_weight = jnp.ones_like(valid_mask)
          masked_weight = jnp.ones_like(valid_mask) * confidence_

          valid_mask = (valid_mask & (batch['inpainting_mask'] == 0)) | ((batch['inpainting_mask'] == 1) & (confidence_mask_ == 1))
          
          valid_mask_real = (valid_mask & (batch['inpainting_mask'] == 0))
          valid_mask_inpainting =  ((batch['inpainting_mask'] == 1) & (confidence_mask_ == 1))

          non_masked_weight = jnp.ones_like(valid_mask)
          masked_weight = config.inpainting_depth_factor * jnp.ones_like(valid_mask) * confidence_
          
          weights = jnp.where(valid_mask_real == 1, non_masked_weight, jnp.zeros_like(non_masked_weight))
          weights = jnp.where(valid_mask_inpainting == 1, masked_weight, weights)


          assert weights.shape == valid_mask.shape == ld.shape

          ld = weights * valid_mask * ld

          # ld = jnp.where(valid_mask == 1, ld, jnp.zeros_like(ld))
          denom_d = jnp.sum(valid_mask) + 1.e-06
          numer_d = jnp.sum(ld) 
          ld = numer_d / denom_d

        losses_depth.append(ld)

      # regularization from mipnerf 360
      if config.train_dist_reg:
        t_vals = rendering['t_vals_pointwise']
        weights = rendering['weights_pointwise']

        # convert t_vals to s values
        s_vals = ((1. / t_vals) - (1. / config.near)) / (1. / config.far - 1. / config.near)
        s_mids = 0.5 * (s_vals[Ellipsis, :-1] + s_vals[Ellipsis, 1:])
        s_dist = s_vals[Ellipsis, 1:] - s_vals[Ellipsis, :-1]

        # compute loss terms
        s_mids_diff = jnp.expand_dims(s_mids, axis=-1) - jnp.expand_dims(s_mids, axis=-2)
        s_mids_diff = jnp.abs(s_mids_diff)
        weights_prod = jnp.expand_dims(weights, axis=-1) * jnp.expand_dims(weights, axis=-2)
        weights_prod = s_mids_diff * weights_prod
        dist_reg_reg = 1./3. * ((weights ** 2) * s_dist)
        
        dist_reg_weight_mask = jnp.where(batch['inpainting_mask'] == 0, jnp.ones_like(batch['inpainting_mask']), config.dist_reg_mask_weight_ratio * jnp.ones_like(batch['inpainting_mask']))
        l_dist_reg = weights_prod.sum(axis=[-1, -2]) + dist_reg_reg.sum(axis=-1)
        assert l_dist_reg.shape == dist_reg_weight_mask.shape
        l_dist_reg = dist_reg_weight_mask * l_dist_reg
        l_dist_reg = l_dist_reg.mean()

        losses_dist_reg.append(l_dist_reg)

    losses_georeg = []

    losses = jnp.array(losses)
    losses_georeg = jnp.array(losses_georeg)
    losses_mse = jnp.array(losses_mse)
    losses_data_unc = jnp.array(losses_data_unc)
    disp_mses = jnp.array(disp_mses)
    normal_maes = jnp.array(normal_maes)

    min_logit = 0.
    max_logit = 0.
    
    loss = (
        losses[-1] + config.coarse_loss_mult * jnp.sum(losses[:-1]) +
        config.weight_decay_mult * weight_l2) # + losses_data_unc[-1])
    
    loss = config.rgb_loss_weight * loss

    if config.train_depth:
      losses_depth = config.depth_loss_weight *jnp.array(losses_depth)
      loss = loss + losses_depth[-1] + config.coarse_loss_mult * jnp.sum(losses_depth[:-1])

    if config.train_label:
      losses_label = config.label_loss_weight * jnp.array(losses_label)
      loss = loss +  losses_label[-1] + config.coarse_loss_mult * jnp.sum(losses_label[:-1])

    if config.train_dist_reg:
      losses_dist_reg = config.dist_reg_loss_weight * jnp.array(losses_dist_reg)
      loss = loss + (losses_dist_reg[-1] + config.coarse_loss_mult * jnp.sum(losses_dist_reg[:-1]))

    if config.train_inpainting:
      losses_inpainting = jnp.array(losses_inpainting)
      loss = loss + config.inpainting_loss_weight * (losses_inpainting[-1] + config.coarse_loss_mult * jnp.sum(losses_inpainting[:-1]))
   
    conf_min = 0
    conf_max = 0
    alpha_min = 0
    alpha_max = 0
    max_violation = 0

    return loss, (losses, losses_mse, losses_depth, losses_label, losses_inpainting, losses_dist_reg, disp_mses, normal_maes, weight_l2, losses_georeg, conf_min, conf_max, alpha_min, alpha_max, min_logit, max_logit, max_violation)

  (loss, loss_aux), (grad, grad_latent_codes) = (jax.value_and_grad(loss_fn, argnums=[0, 1], has_aux=True)(
      state.optimizer.target, state.optimizer_latent_codes.target[batch['image_ids']], step=state.optimizer.state.step))
  (losses, losses_mse, losses_depth, losses_label, losses_inpainting, losses_dist_reg, disp_mses, normal_maes, weight_l2, losses_georeg, conf_min, conf_max, alpha_min, alpha_max, min_logit, max_logit, max_violation) = loss_aux
  
  grad = jax.lax.pmean(grad, axis_name='batch')
  losses = jax.lax.pmean(losses, axis_name='batch')
  disp_mses = jax.lax.pmean(disp_mses, axis_name='batch')
  normal_maes = jax.lax.pmean(normal_maes, axis_name='batch')
  weight_l2 = jax.lax.pmean(weight_l2, axis_name='batch')
  losses_georeg = jax.lax.pmean(losses_georeg, axis_name='batch')

  def check_grad(grad):
    grad_abs_max = jax.tree_util.tree_reduce(
        lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), grad, initializer=0)

    if config.check_grad_for_nans:
      grad = jax.tree_map(jnp.nan_to_num, grad)

    if config.grad_max_val > 0:
      grad = jax.tree_map(
          lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val), grad)

    grad_abs_max = jax.tree_util.tree_reduce(
        lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), grad, initializer=0)

    grad_norm = tree_norm(grad)
    if config.grad_max_norm > 0:
      mult = jnp.minimum(
          1, config.grad_max_norm / (jnp.finfo(jnp.float32).eps + grad_norm))
      grad = jax.tree_map(lambda z: mult * z, grad)
    grad_norm_clipped = tree_norm(grad)

    return grad, (grad_norm_clipped, grad_abs_max, grad_norm)

  grad, (grad_norm_clipped, grad_abs_max, grad_norm) = check_grad(grad)
  grad_latent_codes, (grad_norm_clipped_latent, grad_abs_max_latent, grad_norm_latent) = check_grad(grad_latent_codes)
  
  # latent code optimization
  grad_latent_codes_aggregated = jnp.zeros_like(state.optimizer_latent_codes.target)

  for id in range(grad_latent_codes_aggregated.shape[0]):
    mask = jnp.expand_dims(batch['image_ids'] == id, axis=-1)
    mask = jnp.tile(mask, reps=(1, grad_latent_codes.shape[-1]))
    grad_agg = jnp.where(mask, grad_latent_codes, jnp.zeros_like(grad_latent_codes)).sum(axis=0)
    grad_count = mask.sum(axis=0)

    assert grad_agg.shape == grad_count.shape

    grad_agg = grad_agg / jnp.clip(grad_count, a_min=1.)
    grad_latent_codes_aggregated = grad_latent_codes_aggregated.at[id, :].set(grad_agg)  

  new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=learning_rate)
  new_state = state.replace(optimizer=new_optimizer)

  new_optimizer_latent_codes = state.optimizer_latent_codes.apply_gradient(grad_latent_codes_aggregated, learning_rate=learning_rate)
  new_state = new_state.replace(optimizer_latent_codes=new_optimizer_latent_codes)

  psnrs = math.mse_to_psnr(losses_mse)
  stats = TrainStats(
      loss=loss,
      losses=losses,
      losses_georeg=losses_georeg,
      losses_depth=losses_depth,
      losses_label=losses_label,
      losses_inpainting=losses_inpainting,
      losses_dist_reg=losses_dist_reg,
      disp_mses=disp_mses,
      normal_maes=normal_maes,
      weight_l2=weight_l2,
      psnr=psnrs[-1],
      psnrs=psnrs,
      grad_norm=grad_norm,
      grad_abs_max=grad_abs_max,
      grad_norm_clipped=grad_norm_clipped,
      conf_min=conf_min,
      conf_max=conf_max,
      alpha_min=alpha_min,
      alpha_max=alpha_max,
      max_logit=max_logit,
      min_logit=min_logit,
      max_violation=max_violation
  )

  return new_state, stats, rng


def main(unused_argv):

  rng = random.PRNGKey(20200823)

  # Shift the numpy random seed by host_id() to shuffle data loaded by different
  # hosts.
  np.random.seed(20201473 + jax.host_id())

  config = configs.load_config()

  if config.batch_size % jax.device_count() != 0:
    raise ValueError('Batch size must be divisible by the number of devices.')

  dataset = datasets.load_dataset('train', config.data_dir, config)
  test_dataset = datasets.load_dataset('test', config.data_dir, config)
  test_train_dataset = datasets.load_dataset('test_train', config.data_dir, config)


  rng, key = random.split(rng)
  model, variables = models.construct_mipnerf(
      key,
      dataset.peek()['rays'],
      config,
  )

  rng, key = random.split(rng)
  
  # intialize latent codes for all training images
  latent_codes = jnp.zeros((dataset.size, 1))
  latent_mask = jnp.ones_like(latent_codes)
  latent_codes = jnp.concatenate((latent_codes, latent_mask), axis=-1)

  # intialize latent codes for all test images
  rng, key = random.split(rng)
  latent_codes_test = 0.01 * jax.random.normal(key, (test_dataset.size, 8))

  # variables = variables.copy({"latent_codes": latent_codes})

  num_params = jax.tree_util.tree_reduce(
      lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0)
  print(f'Number of parameters being optimized: {num_params}')

  optimizer = flax.optim.Adam(config.lr_init).create(variables)
  optimizer_latent_codes = flax.optim.Adam(config.lr_init).create(latent_codes)

  state = utils.TrainState(optimizer=optimizer, optimizer_latent_codes=optimizer_latent_codes)
  del optimizer, variables, optimizer_latent_codes

  train_pstep = jax.pmap(
      functools.partial(train_step, model, config), axis_name='batch',
      in_axes=(0, 0, 0, None, None, None, None))

  # Because this is only used for test set rendering, we disable randomization
  # and use the "final" padding for resampling.
  def render_eval_fn(variables, _, rays):
    return jax.lax.all_gather(
        model.apply(
            variables,
            None,  # Deterministic.            
            rays,
            resample_padding=config.resample_padding_final,
            compute_extras=True), axis_name='batch')

  render_eval_pfn = jax.pmap(
      render_eval_fn,
      axis_name='batch',
      in_axes=(None, None, 0),  # Only distribute the data input.
      donate_argnums=(3,),
  )

  def ssim_fn(x, y):
    return structural_similarity(x, y, multichannel=True)

  if not utils.isdir(config.checkpoint_dir):
    utils.makedirs(config.checkpoint_dir)

  # get max checkpoint
  checkpoints_states = [int(ckpt.split('/')[-1].split('_')[-1]) for ckpt in glob.glob(os.path.join(config.checkpoint_dir, 'checkpoint_*'))]
  checkpoints_states = sorted(checkpoints_states)
  if len(checkpoints_states) > 0:
    if checkpoints_states[-1] == config.max_steps:
      return True

  state = checkpoints.restore_checkpoint(config.checkpoint_dir, state)
  # Resume training at the step of the last checkpoint.
  init_step = state.optimizer.state.step + 1
  state = flax.jax_utils.replicate(state)

  if jax.host_id() == 0:
    summary_writer = tensorboard.SummaryWriter(config.checkpoint_dir)
    summary_writer.text('config', f'<pre>{config}</pre>', step=0)

  # Prefetch_buffer_size = 3 x batch_size
  pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
  rng = rng + jax.host_id()  # Make random seed separate across hosts.
  rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
  gc.disable()  # Disable automatic garbage collection for efficiency.
  total_time = 0
  total_steps = 0
  avg_psnr_numer = 0.
  avg_psnr_denom = 0
  train_start_time = time.time()

  n_inactive = 0
  n_active = dataset.size

  lr_step = init_step
  lr_steps_max = config.max_steps
  
  jax.tree_util.tree_map(lambda x: print(x.shape, x) if len(x.shape) == 0 else False, state)

  for step, batch in zip(range(init_step, config.max_steps + 1), pdataset):
    learning_rate = math.learning_rate_decay(
        lr_step,
        config.lr_init,
        config.lr_final,
        lr_steps_max,
        config.lr_delay_steps,
        config.lr_delay_mult,
    )

    resample_padding = math.log_lerp(
        step / config.max_steps,
        config.resample_padding_init,
        config.resample_padding_final,
    )

    tvnorm_loss_weight = 0.
    

    state, stats, rngs = train_pstep(
        rngs,
        state,
        batch,
        learning_rate,
        resample_padding,
        tvnorm_loss_weight,
        latent_codes,
    )

    def reset_variables(rng, state, config):
      rng, key = random.split(rng, 2)

      model, variables = models.construct_mipnerf(key, dataset.peek()['rays'], config, )

      rng, key = random.split(rng, 2)

      new_optimizer = flax.optim.Adam(config.lr_init).create(variables)
      new_optimizer = flax.jax_utils.replicate(new_optimizer)

      state = state.replace(optimizer=new_optimizer)

      jax.tree_util.tree_map(lambda x: print(x.shape, x) if len(x.shape) == 0 else False, state)

      train_pstep = jax.pmap(functools.partial(train_step, model, config), axis_name='batch',
                             in_axes=(0, 0, 0, None, None, None, None))

      del variables

    

      rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.      
      pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)

      return state, train_pstep, rngs, pdataset, model

    # only do that on host id 0
    if (step + 1) % config.filter_views_every == 0 and config.filter_view_conf and jax.host_id() == 0 and (step + 1) <= config.filter_stop + 1:
      assert latent_codes.shape[-1] > 1
      # define new latent codes
      new_codes = state.optimizer_latent_codes.target[:, :, 0]
      new_confidences = jnp.exp(-1. * new_codes)
      old_mask = state.optimizer_latent_codes.target[:, :, 1] > 0
      
      threshold = min(jnp.percentile(new_confidences[old_mask == 1], config.filter_threshold_percentile), 0.999)
      new_mask = (new_confidences > threshold) & (state.optimizer_latent_codes.target[:, :, 1] == 1.)
      new_codes = jnp.where(new_mask, jnp.zeros_like(new_codes), new_codes) # resetting all kept masks to one
      
      n_active = int(new_mask.sum() / new_codes.shape[0])
      n_inactive = dataset.size - n_active
      
      if config.reset_variables_in_filter:
        del model, train_pstep
        state, train_pstep, rngs, pdataset, model = reset_variables(rng, state, config)
        learning_rate = config.lr_init
        lr_steps_max = lr_steps_max - lr_step 
        lr_step = 0

      if config.hard_filter:
        # remove views from batching
        views_removed = (new_mask == 0)[0]
        views_removed = jnp.nonzero(views_removed)[0]
        
        image_ids_ = dataset.image_ids
        mask_ = dataset.masks

        image_id_mask_ = jnp.isin(image_ids_, views_removed)
        new_ray_mask = image_id_mask_ & mask_
        new_ray_mask = new_ray_mask == 0
        
        dataset.ray_mask_for_batching = new_ray_mask.copy()

      print(f'{n_active} / {dataset.size} views currently activated in optimization')
      print(f'{n_inactive} / {new_mask.shape[1]} views currently deactivated in optimization')

      latent_codes = jnp.stack((new_codes, new_mask), axis=-1)

      # replace optimizer target
      optimizer_ = state.optimizer_latent_codes
      new_optimizer = optimizer_.replace(target=latent_codes, state=optimizer_.state)
      state = state.replace(optimizer_latent_codes=new_optimizer)

      if config.reset_lr:
        learning_rate = config.lr_init
        lr_steps_max = lr_steps_max - lr_step 
        lr_step = 0

    if config.reset_variables and step == config.reset_variables_th:
      del model, train_pstep
      state, train_pstep, rngs, pdataset, model = reset_variables(rng, state, config)
      learning_rate = config.lr_init
      lr_steps_max = lr_steps_max - lr_step 
      lr_step = 0

    lr_step += 1

    if step % config.gc_every == 0:
      gc.collect()  # Disable automatic garbage collection for efficiency.

    # Log training summaries. This is put behind a host_id check because in
    # multi-host evaluation, all hosts need to run inference even though we
    # only use host 0 to record results.
    if jax.host_id() == 0:
      avg_psnr_numer += stats.psnr[0]
      avg_psnr_denom += 1
      if step % config.print_every == 0:
        elapsed_time = time.time() - train_start_time
        steps_per_sec = config.print_every / elapsed_time
        rays_per_sec = config.batch_size * steps_per_sec

        # A robust approximation of total training time, in case of pre-emption.
        total_time += int(round(TIME_PRECISION * elapsed_time))
        total_steps += config.print_every
        approx_total_time = int(round(step * total_time / total_steps))

        avg_psnr = avg_psnr_numer / avg_psnr_denom
        avg_psnr_numer = 0.
        avg_psnr_denom = 0

        # For some reason, the `stats` object has a superfluous dimension.
        stats = jax.tree_map(lambda x: x[0], stats)
        summary_writer.scalar('num_params', num_params, step)
        summary_writer.scalar('train_loss', stats.loss, step)
        summary_writer.scalar('train_psnr', stats.psnr, step)
        if config.compute_disp_metrics:
          for i, disp_mse in enumerate(stats.disp_mses):
            summary_writer.scalar(f'train_disp_mse_{i}', disp_mse, step)
        if config.compute_normal_metrics:
          for i, normal_mae in enumerate(stats.normal_maes):
            summary_writer.scalar(f'train_normal_mae_{i}', normal_mae, step)
        summary_writer.scalar('train_avg_psnr', avg_psnr, step)
        summary_writer.scalar('train_avg_psnr_timed', avg_psnr,
                              total_time // TIME_PRECISION)
        summary_writer.scalar('train_avg_psnr_timed_approx', avg_psnr,
                              approx_total_time // TIME_PRECISION)

        active_mask = state.optimizer_latent_codes.target[:, :, 1]
        active_latent_codes = state.optimizer_latent_codes.target[:, :, 0][active_mask == 1]

        if active_mask.sum() == 0:
          active_latent_codes = 10. * jnp.ones_like(active_mask)

        summary_writer.scalar('max_latent_code', jnp.max(active_latent_codes), step)
        summary_writer.scalar('min_latent_code', jnp.min(active_latent_codes), step)
        summary_writer.scalar('median_latent_code', jnp.median(active_latent_codes), step)

        summary_writer.scalar('max_conf_code', jnp.max(jnp.exp(-1. * active_latent_codes)), step)
        summary_writer.scalar('min_conf_code', jnp.min(jnp.exp(-1. * active_latent_codes)), step)
        summary_writer.scalar('median_conf_code', jnp.median(jnp.exp(-1. * active_latent_codes)), step)

        summary_writer.histogram('conf_code', jnp.exp(-1. * active_latent_codes), step)

        for i, l in enumerate(stats.losses_depth):
          summary_writer.scalar(f'train_losses_depth_{i}', l, step)
        for i, l in enumerate(stats.losses_label):
          summary_writer.scalar(f'train_losses_label_{i}', l, step)
        for i, l in enumerate(stats.losses_inpainting):
          summary_writer.scalar(f'train_losses_inpainting_{i}', l, step)
        for i, l in enumerate(stats.losses_dist_reg):
          summary_writer.scalar(f'train_losses_dist_reg_{i}', l, step)
        for i, l in enumerate(stats.losses):
          summary_writer.scalar(f'train_losses_{i}', l, step)
        for i, l in enumerate(stats.losses_georeg):
          summary_writer.scalar(f'train_losses_depth_tv_norm{i}', l, step)
        for i, p in enumerate(stats.psnrs):
          summary_writer.scalar(f'train_psnrs_{i}', p, step)
        summary_writer.scalar('weight_l2', stats.weight_l2, step)
        summary_writer.scalar('train_grad_norm', stats.grad_norm, step)
        summary_writer.scalar('train_grad_norm_clipped',
                              stats.grad_norm_clipped, step)
        summary_writer.scalar('train_grad_abs_max', stats.grad_abs_max, step)
        summary_writer.scalar('learning_rate', learning_rate, step)
        summary_writer.scalar('tvnorm_loss_weight', tvnorm_loss_weight, step)
        summary_writer.scalar('resample_padding', resample_padding, step)
        summary_writer.scalar('train_steps_per_sec', steps_per_sec, step)
        summary_writer.scalar('train_rays_per_sec', rays_per_sec, step)
        summary_writer.scalar('train_inactive', n_inactive, step)
        summary_writer.scalar('train_active', n_active, step)

        precision = int(np.ceil(np.log10(config.max_steps))) + 1
        print(f'{step:{precision}d}' + f'/{config.max_steps:d}: ' +
              f'loss={stats.loss:0.4f}, ' + f'avg_psnr={avg_psnr:0.2f}, ' +
              f'weight_l2={stats.weight_l2:0.2e}, ' +
              f'lr={learning_rate:0.2e}, '
              f'pad={resample_padding:0.2e}, ' +
              f'{rays_per_sec:0.0f} rays/sec')
        summary_writer.scalar('train_conf_min', stats.conf_min, step)
        summary_writer.scalar('train_conf_max', stats.conf_max, step)
        summary_writer.scalar('train_alpha_min', stats.alpha_min, step)
        summary_writer.scalar('train_alpha_max', stats.alpha_max, step)
        # summary_writer.embedding('latent_codes', state.optimizer_latent_codes.target, step)
        # summary_writer.scalar('train_grad_norm_latent', stats.grad_norm_latent, step)
        # summary_writer.scalar('train_grad_norm_clipped_latent',
        #                       stats.grad_norm_clipped_latent, step)
        # summary_writer.scalar('train_grad_abs_max_latent', stats.grad_abs_max_latent, step)
        summary_writer.scalar('train_min_logit', stats.min_logit, step)
        summary_writer.scalar('train_max_logit', stats.max_logit, step)
        summary_writer.scalar('train_max_violation', stats.max_violation, step)

        train_start_time = time.time()

      if step % config.checkpoint_every == 0:
        state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            config.checkpoint_dir, state_to_save, int(step), keep=100)

    # Test-set evaluation.
    if config.train_render_every > 0 and step % config.train_render_every == 0:
      jnp.save(config.checkpoint_dir + '/latent_codes.npy', state.optimizer_latent_codes.target)


      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.
      eval_start_time = time.time()
      eval_variables = jax.device_get(jax.tree_map(lambda x: x[0],
                                                   state)).optimizer.target
      test_case = next(test_dataset)

      rendering = models.render_image(
          functools.partial(render_eval_pfn, eval_variables),
          test_case['rays'],
          rngs[0],
          config,)

      vis_start_time = time.time()
      vis_suite = vis.visualize_suite(jax.lax.stop_gradient(rendering), jax.lax.stop_gradient(test_case['rays']), config)
      print(f'Visualized in {(time.time() - vis_start_time):0.3f}s')

      # Log eval summaries on host 0.
      if jax.host_id() == 0:
        if not config.render_path:
          psnr_multi_view = float(
              math.mse_to_psnr(((
                  rendering['rgb_multi_view'] - test_case['rgb'])**2).mean()))
          ssim_multi_view = float(ssim_fn(np.asarray(rendering['rgb_multi_view']), np.asarray(test_case['rgb'])))
          psnr_view_dir = float(
              math.mse_to_psnr(((
                  rendering['rgb_view_dir'] - test_case['rgb'])**2).mean()))
          ssim_view_dir = float(ssim_fn(np.asarray(rendering['rgb_view_dir']), np.asarray(test_case['rgb'])))
        eval_time = time.time() - eval_start_time
        num_rays = jnp.prod(jnp.array(test_case['rays'].directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
        print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')
        if not config.render_path:
          print(f'PSNR={psnr_multi_view:.4f} SSIM={ssim_multi_view:.4f}')
          summary_writer.scalar('test_psnr_multi_view', psnr_multi_view, step)
          summary_writer.scalar('test_ssim_multi_view', ssim_multi_view, step)
          summary_writer.scalar('test_psnr_view_dir', psnr_view_dir, step)
          summary_writer.scalar('test_ssim_view_dir', ssim_view_dir, step)
          summary_writer.image('test_target', test_case['rgb'], step)

          label_vis = test_case['mask'].copy()
          label_vis = label_vis * 255
          label_vis = jnp.stack((label_vis, label_vis, label_vis), axis=-1)
          summary_writer.image('test_label', label_vis, step)
         
        for k, v in vis_suite.items():
          summary_writer.image('test_pred_' + k, v, step)

    # Train-set evaluation.
    if config.train_render_every > 0 and step % config.train_render_every == 0:

      # We reuse the same random number generator from the optimization step
      # here on purpose so that the visualization matches what happened in
      # training.

      eval_start_time = time.time()
      eval_variables = jax.device_get(jax.tree_map(lambda x: x[0],
                                                   state)).optimizer.target
      test_case = next(test_train_dataset)

      rendering = models.render_image(
          functools.partial(render_eval_pfn, eval_variables),
          test_case['rays'],
          rngs[0],
          config)

      vis_start_time = time.time()
      vis_suite = vis.visualize_suite(rendering, test_case['rays'], config)
      print(f'Visualized in {(time.time() - vis_start_time):0.3f}s')

      # Log eval summaries on host 0.
      if jax.host_id() == 0:
        if not config.render_path:
          psnr_multi_view = float(
              math.mse_to_psnr(((
                  rendering['rgb_multi_view'] - test_case['rgb'])**2).mean()))
          ssim_multi_view = float(ssim_fn(np.asarray(rendering['rgb_multi_view']), np.asarray(test_case['rgb'])))
          psnr_view_dir = float(
              math.mse_to_psnr(((
                  rendering['rgb_view_dir'] - test_case['rgb'])**2).mean()))
          ssim_view_dir = float(ssim_fn(np.asarray(rendering['rgb_view_dir']), np.asarray(test_case['rgb'])))
        eval_time = time.time() - eval_start_time
        num_rays = jnp.prod(jnp.array(test_case['rays'].directions.shape[:-1]))
        rays_per_sec = num_rays / eval_time
        summary_writer.scalar('test_train_rays_per_sec', rays_per_sec, step)
        print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')
        if not config.render_path:        

          if 'inpainting' in test_case.keys():
            summary_writer.image('test_train_inpainting', test_case['inpainting'], step)

          print(f'PSNR={psnr_multi_view:.4f} SSIM={ssim_multi_view:.4f}')
          summary_writer.scalar('test_train_psnr_multi_view', psnr_multi_view, step)
          summary_writer.scalar('test_train_ssim_multi_view', ssim_multi_view, step)
          summary_writer.scalar('test_train_psnr_view_dir', psnr_view_dir, step)
          summary_writer.scalar('test_train_ssim_view_dir', ssim_view_dir, step)
          summary_writer.image('test_train_target', test_case['rgb'], step)

          label_vis = test_case['mask'].copy()
          label_vis = label_vis * 255
          label_vis = jnp.stack((label_vis, label_vis, label_vis), axis=-1)
          summary_writer.image('test_train_label', label_vis, step)
  
        for k, v in vis_suite.items():
          summary_writer.image('test_train_pred_' + k, v, step)


  

  if config.max_steps % config.checkpoint_every != 0:
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    checkpoints.save_checkpoint(
        config.checkpoint_dir, state, int(config.max_steps), keep=100)


if __name__ == '__main__':
  app.run(main)
