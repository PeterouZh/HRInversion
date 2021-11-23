import os
import json
import collections
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as trans

from tl2.proj.fvcore import MODEL_REGISTRY
from tl2.proj.fvcore.checkpoint import Checkpointer
from tl2.proj.pytorch.pytorch_hook import FeatureExtractor

from torch_utils import misc
from torch_utils import persistence
from ada_lib.training.networks import (SynthesisBlock, MappingNetwork, FullyConnectedLayer,
                                       normalize_2nd_moment)
from ada_lib.training.networks import Discriminator as Discriminator_base


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
  def __init__(self,
               w_dim,  # Intermediate latent (W) dimensionality.
               img_resolution,  # Output image resolution.
               img_channels,  # Number of color channels.
               shape,
               inr_net_kwargs={},
               tanh=False,
               channel_base=32768,  # Overall multiplier for the number of channels.
               channel_max=512,  # Maximum number of channels in any layer.
               num_fp16_res=0,  # Use FP16 for the N highest resolutions.
               **block_kwargs,  # Arguments for SynthesisBlock.
               ):
    assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
    super().__init__()
    self.w_dim = w_dim
    self.img_resolution = img_resolution
    self.img_resolution_log2 = int(np.log2(img_resolution))
    self.img_channels = img_channels
    self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
    channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
    fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
    self.shape = shape
    self.tanh = tanh

    self.num_ws = 0
    for res in self.block_resolutions:
      in_channels = channels_dict[res // 2] if res > 4 else 0
      out_channels = channels_dict[res]
      use_fp16 = (res >= fp16_resolution)
      is_last = (res == self.img_resolution)
      block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                             img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
      self.num_ws += block.num_conv
      if is_last:
        self.num_ws += block.num_torgb
      setattr(self, f'b{res}', block)

    self.inr_net = INRNetwork(**inr_net_kwargs)
    pass

  def forward(self,
              ws,
              shape=None,
              **block_kwargs):
    block_ws = []
    with torch.autograd.profiler.record_function('split_ws'):
      misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
      ws = ws.to(torch.float32)
      w_idx = 0
      for res in self.block_resolutions:
        block = getattr(self, f'b{res}')
        block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
        w_idx += block.num_conv

    x = img = None
    for res, cur_ws in zip(self.block_resolutions, block_ws):
      block = getattr(self, f'b{res}')
      x, img = block(x, img, cur_ws, **block_kwargs)

    if shape is None:
      shape = self.shape
    if shape is not None:
      x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
      img = self.inr_net(x, shape=shape)

    if self.tanh:
      img = torch.tanh(img)
    return img


@persistence.persistent_class
class INRNetwork(torch.nn.Module):
  def __init__(self,
               in_dim,
               out_dim,
               hidden_list,
               max_points_every=65536,
               activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
               lr_multiplier=1.,  # Learning rate multiplier for the mapping layers.
               **kwargs
               ):
    super().__init__()
    self.hidden_list = hidden_list
    self.max_points_every = max_points_every

    lastv = in_dim
    for idx, hidden in enumerate(hidden_list):
      name = f"inr_fc_{idx}"
      layer = FullyConnectedLayer(lastv, hidden, activation=activation, lr_multiplier=lr_multiplier)
      setattr(self, name, layer)
      lastv = hidden

    name = f"inr_fc_out"
    out_layer = FullyConnectedLayer(lastv, out_dim, activation="linear", lr_multiplier=lr_multiplier)
    setattr(self, name, out_layer)

    pass

  def forward(self,
              x,
              shape,
              max_points_every=None):
    if max_points_every is None:
      max_points_every = self.max_points_every
    coord = self.make_coord(shape=shape, flatten=True, device=x.device)
    coord = coord.unsqueeze(0).expand(x.shape[0], *coord.shape[-2:]).requires_grad_(False)

    # feat = F.unfold(h, 3, padding=1).view(h.shape[0], h.shape[1] * 9, h.shape[2], h.shape[3])
    feat = x

    num_iters = (coord.size(1) + max_points_every - 1) // max_points_every

    pred = []
    for idx in range(num_iters):
      part_coord = coord[:, idx * max_points_every:(idx + 1) * max_points_every]

      part_pred = self._sample_rgb(feat=feat, coord=part_coord)
      pred.append(part_pred)
    pred = torch.cat(pred, dim=1)

    out = pred.permute(0, 2, 1).view(pred.shape[0], -1, *shape)
    del coord
    return out

  def _sample_rgb(self,
                  feat,
                  coord):
    sampled_feat = F.grid_sample(
      feat, coord.flip(-1).unsqueeze(1), mode='bilinear', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

    inp = torch.cat([sampled_feat, coord.detach()], dim=-1)

    bs, q = inp.shape[:2]
    points = inp.view(bs * q, -1)

    pred = self.inr_net(points).view(bs, q, -1)
    return pred

  def inr_net(self, x):
    for idx in range(len(self.hidden_list)):
      name = f"inr_fc_{idx}"
      layer = getattr(self, name)
      x = layer(x)

    name = f"inr_fc_out"
    layer = getattr(self, name)
    out = layer(x)
    return out

  def make_coord(self, shape, device, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
      if ranges is None:
        v0, v1 = -1, 1
      else:
        v0, v1 = ranges[i]
      r = (v1 - v0) / (2 * n)
      # fix memory leaky bug
      seq = v0 + r + (2 * r) * torch.arange(n, dtype=torch.float32, device=device)
      coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
      ret = ret.view(-1, ret.shape[-1])
    return ret


@MODEL_REGISTRY.register(name_prefix=__name__)
@persistence.persistent_class
class GeneratorUltra(torch.nn.Module):
  def __init__(self,
               z_dim,  # Input latent (Z) dimensionality.
               c_dim,  # Conditioning label (C) dimensionality.
               w_dim,  # Intermediate latent (W) dimensionality.
               img_resolution,  # Output resolution.
               img_channels,  # Number of output color channels.
               shape,
               tanh=False,
               mapping_kwargs={},  # Arguments for MappingNetwork.
               synthesis_kwargs={},  # Arguments for SynthesisNetwork.
               inr_net_kwargs={},
               **kwargs):
    super().__init__()
    self.z_dim = z_dim
    self.c_dim = c_dim
    self.w_dim = w_dim
    self.img_resolution = img_resolution
    self.img_channels = img_channels
    self.shape = shape

    self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                      shape=shape, inr_net_kwargs=inr_net_kwargs, tanh=tanh,
                                      **synthesis_kwargs)
    self.num_ws = self.synthesis.num_ws
    self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    pass

  def forward(self,
              z,
              c,
              shape=None,
              truncation_psi=1,
              truncation_cutoff=None,
              **synthesis_kwargs):
    ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    if shape is None:
      shape = self.shape
    img = self.synthesis(ws, shape=shape, **synthesis_kwargs)

    return img


@persistence.persistent_class
class SynthesisNetworkPathReg(torch.nn.Module):
  def __init__(self,
               w_dim,  # Intermediate latent (W) dimensionality.
               img_resolution,  # Output image resolution.
               img_channels,  # Number of color channels.
               shape,
               inr_net_kwargs={},
               tanh=False,
               channel_base=32768,  # Overall multiplier for the number of channels.
               channel_max=512,  # Maximum number of channels in any layer.
               num_fp16_res=0,  # Use FP16 for the N highest resolutions.
               freeze_blocks=(),
               **block_kwargs,  # Arguments for SynthesisBlock.
               ):
    assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
    super().__init__()
    self.w_dim = w_dim
    self.img_resolution = img_resolution
    self.img_resolution_log2 = int(np.log2(img_resolution))
    self.img_channels = img_channels
    self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
    channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
    fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
    self.shape = shape
    self.tanh = tanh
    self.freeze_blocks = freeze_blocks

    self.num_ws = 0
    for res in self.block_resolutions:
      in_channels = channels_dict[res // 2] if res > 4 else 0
      out_channels = channels_dict[res]
      use_fp16 = (res >= fp16_resolution)
      is_last = (res == self.img_resolution)
      block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                             img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
      self.num_ws += block.num_conv
      if is_last:
        self.num_ws += block.num_torgb
      setattr(self, f'b{res}', block)

    if inr_net_kwargs['disable']:
      self.inr_net = None
    else:
      self.inr_net = INRNetwork(**inr_net_kwargs)
    pass

  def forward(self,
              ws,
              return_ori_out=False,
              shape=None,
              **block_kwargs):
    block_ws = []
    with torch.autograd.profiler.record_function('split_ws'):
      misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
      ws = ws.to(torch.float32)
      w_idx = 0
      for res in self.block_resolutions:
        block = getattr(self, f'b{res}')
        block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
        w_idx += block.num_conv

    x = img = None
    for res, cur_ws in zip(self.block_resolutions, block_ws):
      block = getattr(self, f'b{res}')
      if res in self.freeze_blocks:
        with torch.no_grad():
          x, img = block(x, img, cur_ws, **block_kwargs)
      else:
        x, img = block(x, img, cur_ws, **block_kwargs)
    if return_ori_out:
      return img

    if self.inr_net is None:
      return img

    if shape is None:
      shape = self.shape
    if shape is not None:
      x = x.to(dtype=torch.float32, memory_format=torch.contiguous_format)
      img = self.inr_net(x, shape=shape)

    if self.tanh:
      img = torch.tanh(img)
    return img

@MODEL_REGISTRY.register(name_prefix=__name__)
@persistence.persistent_class
class GeneratorUltraPathReg(torch.nn.Module):
  def __init__(self,
               z_dim,  # Input latent (Z) dimensionality.
               c_dim,  # Conditioning label (C) dimensionality.
               w_dim,  # Intermediate latent (W) dimensionality.
               img_resolution,  # Output resolution.
               img_channels,  # Number of output color channels.
               shape,
               return_ori_out=False,
               tanh=False,
               mapping_kwargs={},  # Arguments for MappingNetwork.
               synthesis_kwargs={},  # Arguments for SynthesisNetwork.
               inr_net_kwargs={},
               **kwargs):
    super().__init__()
    self.z_dim = z_dim
    self.c_dim = c_dim
    self.w_dim = w_dim
    self.img_resolution = img_resolution
    self.img_channels = img_channels
    self.shape = shape
    self.return_ori_out = return_ori_out

    self.synthesis = SynthesisNetworkPathReg(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                             shape=shape, inr_net_kwargs=inr_net_kwargs, tanh=tanh,
                                             **synthesis_kwargs)
    self.num_ws = self.synthesis.num_ws
    self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    pass

  def forward(self,
              z,
              c,
              shape=None,
              truncation_psi=1,
              truncation_cutoff=None,
              return_ori_out=None,
              **synthesis_kwargs):
    ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    if shape is None:
      shape = self.shape
    if return_ori_out is None:
      return_ori_out = self.return_ori_out
    img = self.synthesis(ws, shape=shape, return_ori_out=return_ori_out, **synthesis_kwargs)

    return img


@persistence.persistent_class
class MappingNetworkZPC(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        # if c_dim == 0:
        #     embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        # features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]
        assert z_dim == embed_features
        features_list = [z_dim] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))
            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                # x = torch.cat([x, y], dim=1) if x is not None else y
                x = x + y if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

@MODEL_REGISTRY.register(name_prefix=__name__)
@persistence.persistent_class
class GeneratorUltraZPC(torch.nn.Module):
  def __init__(self,
               z_dim,  # Input latent (Z) dimensionality.
               c_dim,  # Conditioning label (C) dimensionality.
               w_dim,  # Intermediate latent (W) dimensionality.
               img_resolution,  # Output resolution.
               img_channels,  # Number of output color channels.
               shape,
               return_ori_out=False,
               tanh=False,
               mapping_kwargs={},  # Arguments for MappingNetwork.
               synthesis_kwargs={},  # Arguments for SynthesisNetwork.
               inr_net_kwargs={},
               **kwargs):
    super().__init__()
    self.z_dim = z_dim
    self.c_dim = c_dim
    self.w_dim = w_dim
    self.img_resolution = img_resolution
    self.img_channels = img_channels
    self.shape = shape
    self.return_ori_out = return_ori_out

    self.synthesis = SynthesisNetworkPathReg(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
                                             shape=shape, inr_net_kwargs=inr_net_kwargs, tanh=tanh,
                                             **synthesis_kwargs)
    self.num_ws = self.synthesis.num_ws
    self.mapping = MappingNetworkZPC(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    pass

  def forward(self,
              z,
              c,
              shape=None,
              truncation_psi=1,
              truncation_cutoff=None,
              return_ori_out=None,
              **synthesis_kwargs):
    ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    if shape is None:
      shape = self.shape
    if return_ori_out is None:
      return_ori_out = self.return_ori_out
    img = self.synthesis(ws, shape=shape, return_ori_out=return_ori_out, **synthesis_kwargs)

    return img


@MODEL_REGISTRY.register(name_prefix=__name__)
@persistence.persistent_class
class GeneratorUltraZPC_FreezeMapping(GeneratorUltraZPC):

  def forward(self,
              z,
              c,
              shape=None,
              truncation_psi=1,
              truncation_cutoff=None,
              return_ori_out=None,
              **synthesis_kwargs):
    with torch.no_grad():
      ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    if shape is None:
      shape = self.shape
    if return_ori_out is None:
      return_ori_out = self.return_ori_out
    img = self.synthesis(ws, shape=shape, return_ori_out=return_ori_out, **synthesis_kwargs)

    return img

@persistence.persistent_class
class INR_Bicubic_Network(INRNetwork):
  def __init__(self,
               sample_mode="bicubic",
               **kwargs
               ):
    """
    sample_mode: ['bilinear', 'bicubic']
    """
    super().__init__(**kwargs)
    self.sample_mode = sample_mode
    pass

  def _sample_rgb(self,
                  feat,
                  coord):
    sampled_feat = F.grid_sample(
      feat, coord.flip(-1).unsqueeze(1), mode=self.sample_mode, align_corners=False)[:, :, 0, :].permute(0, 2, 1)

    inp = torch.cat([sampled_feat, coord.detach()], dim=-1)

    bs, q = inp.shape[:2]
    points = inp.view(bs * q, -1)

    pred = self.inr_net(points).view(bs, q, -1)
    return pred


@persistence.persistent_class
class SynthesisNetworkPathReg_INRBicubic(SynthesisNetworkPathReg):
  def __init__(self,
               inr_net_kwargs={},
               **kwargs
               ):
    super().__init__(inr_net_kwargs=inr_net_kwargs, **kwargs)

    del self.inr_net
    self.inr_net = INR_Bicubic_Network(**inr_net_kwargs)
    pass


@MODEL_REGISTRY.register(name_prefix=__name__)
@persistence.persistent_class
class GeneratorUltraZPCBicubic(torch.nn.Module):
  def __init__(self,
               z_dim,  # Input latent (Z) dimensionality.
               c_dim,  # Conditioning label (C) dimensionality.
               w_dim,  # Intermediate latent (W) dimensionality.
               img_resolution,  # Output resolution.
               img_channels,  # Number of output color channels.
               shape,
               return_ori_out=False,
               tanh=False,
               mapping_kwargs={},  # Arguments for MappingNetwork.
               synthesis_kwargs={},  # Arguments for SynthesisNetwork.
               inr_net_kwargs={},
               **kwargs):
    super().__init__()
    self.z_dim = z_dim
    self.c_dim = c_dim
    self.w_dim = w_dim
    self.img_resolution = img_resolution
    self.img_channels = img_channels
    self.shape = shape
    self.return_ori_out = return_ori_out

    self.synthesis = SynthesisNetworkPathReg_INRBicubic(
      w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
      shape=shape, inr_net_kwargs=inr_net_kwargs, tanh=tanh,
      **synthesis_kwargs)
    self.num_ws = self.synthesis.num_ws
    self.mapping = MappingNetworkZPC(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    pass

  def forward(self,
              z,
              c,
              shape=None,
              truncation_psi=1,
              truncation_cutoff=None,
              return_ori_out=None,
              **synthesis_kwargs):
    ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)

    if shape is None:
      shape = self.shape
    if return_ori_out is None:
      return_ori_out = self.return_ori_out
    img = self.synthesis(ws, shape=shape, return_ori_out=return_ori_out, **synthesis_kwargs)

    return img



@MODEL_REGISTRY.register(name_prefix=__name__)
@persistence.persistent_class
class Discriminator(Discriminator_base):
  def __init__(self,
               c_dim,                          # Conditioning label (C) dimensionality.
               img_resolution,                 # Input resolution.
               img_channels,                   # Number of input color channels.
               architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
               channel_base        = 32768,    # Overall multiplier for the number of channels.
               channel_max         = 512,      # Maximum number of channels in any layer.
               num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
               conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
               cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
               block_kwargs        = {},       # Arguments for DiscriminatorBlock.
               mapping_kwargs      = {},       # Arguments for MappingNetwork.
               epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
               **kwargs
               ):
    super().__init__(c_dim=c_dim,
                     img_resolution=img_resolution,
                     img_channels=img_channels,
                     architecture=architecture,
                     channel_base=channel_base,
                     channel_max=channel_max,
                     num_fp16_res=num_fp16_res,
                     conv_clamp=conv_clamp,
                     cmap_dim=cmap_dim,
                     block_kwargs=block_kwargs,
                     mapping_kwargs=mapping_kwargs,
                     epilogue_kwargs=epilogue_kwargs)
    pass


@MODEL_REGISTRY.register(name_prefix=__name__)
class DiscriminatorLoss(torch.nn.Module):
  def __init__(self,
               D,
               use_stat_loss=False,
               layers=None,
               loss_w_dict=None,
               **kwargs):
    super().__init__()

    self.use_stat_loss = use_stat_loss

    if layers is None:
      layers = self.layers
    print(f"DiscriminatorLoss layers: {layers}")
    self.D = FeatureExtractor(D, layers=layers)

    self.loss_w_dict = loss_w_dict
    if loss_w_dict is None:
      self.loss_w_dict = self.loss_weight
    print(f"DiscriminatorLoss loss_w_dict: {json.dumps(self.loss_w_dict)}")
    pass

  @property
  def layers(self):
    layers = ['b1024', 'b512', 'b256', 'b128', 'b64', 'b32', 'b16', 'b8', ]
    return layers

  @property
  def loss_weight(self):
    layers = ['b1024', 'b512', 'b256', 'b128', 'b64', 'b32', 'b16', 'b8', ]
    weights = [1, 1, 1, 1, 1, 1, 1, 1]
    assert len(layers) == len(weights)
    loss_w_dict = {}
    for layer, w in zip(layers, weights):
      loss_w_dict[layer] = w
    return loss_w_dict


  def forward(self, *args, loss_w_dict=None, use_stat_loss=None, **kwargs):
    self.D.eval()

    if use_stat_loss is None:
      use_stat_loss = self.use_stat_loss

    feas_dict = self.D(*args, **kwargs)
    feas = []
    for k, v in feas_dict.items():
      fea = v[0].to(torch.float32)
      b, c, h, w = fea.shape
      if use_stat_loss:
        fea = self.stat_loss(fea)
      else:
        fea = fea.flatten(start_dim=1)

      if loss_w_dict is None:
        fea = fea * self.loss_w_dict[k]
      else:
        fea = fea * loss_w_dict[k]
      feas.append(fea)
    feas = torch.cat(feas, dim=1)
    return feas

  def stat_loss(self, fea):
    fea = fea.flatten(start_dim=2)
    var_mean = torch.var_mean(fea, dim=2)
    fea = torch.cat(var_mean, dim=1)
    return fea

@MODEL_REGISTRY.register(name_prefix=__name__)
class DiscriminatorStatLoss(DiscriminatorLoss):
  def forward(self, *args, loss_w_dict=None, **kwargs):
    self.D.eval()
    feas_dict = self.D(*args, **kwargs)
    feas = []
    for k, v in feas_dict.items():
      fea = v[0].to(torch.float32)
      # b, c, h, w = fea.shape
      fea = fea.flatten(start_dim=2)
      var_mean = torch.var_mean(fea, dim=2)
      fea = torch.cat(var_mean, dim=1)
      if loss_w_dict is None:
        fea = fea * self.loss_w_dict[k]
      else:
        fea = fea * loss_w_dict[k]
      feas.append(fea)
    feas = torch.cat(feas, dim=1)
    return feas


@MODEL_REGISTRY.register(name_prefix=__name__)
class Swin22kLoss(torch.nn.Module):
  def __init__(self,
               use_stat_loss=False,
               layers=None,
               loss_w_dict=None,
               **kwargs):
    super().__init__()
    import timm

    self.use_stat_loss = use_stat_loss

    if layers is None:
      layers = self.layers
    print(f"Swin layers: {layers}")
    net = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=True)
    self.net = FeatureExtractor(net, layers=layers)

    self.loss_w_dict = loss_w_dict
    if loss_w_dict is None:
      self.loss_w_dict = self.loss_weight
    print(f"Swin loss_w_dict: {json.dumps(self.loss_w_dict)}")
    pass

  @property
  def layers(self):
    layers = ['patch_embed', 'layers.0', 'layers.1', 'layers.2', 'layers.3', 'avgpool']
    return layers

  @property
  def loss_weight(self):
    layers = ['patch_embed', 'layers.0', 'layers.1', 'layers.2', 'layers.3', 'avgpool']
    weights = [1, 1, 1, 1, 1, 1]
    assert len(layers) == len(weights)
    loss_w_dict = {}
    for layer, w in zip(layers, weights):
      loss_w_dict[layer] = w
    return loss_w_dict


  def forward(self, x, *args, loss_w_dict=None, use_stat_loss=None, **kwargs):
    self.net.eval()

    if use_stat_loss is None:
      use_stat_loss = self.use_stat_loss

    x = F.interpolate(x, size=(384, 384), mode='area')
    feas_dict = self.net(x)
    feas = []
    for k, v in feas_dict.items():
      fea = v
      # b, c, h, w = fea.shape
      if use_stat_loss:
        assert 0
        fea = self.stat_loss(fea)
      else:
        fea = fea.flatten(start_dim=1)

      if loss_w_dict is None:
        fea = fea * self.loss_w_dict[k]
      else:
        fea = fea * loss_w_dict[k]
      feas.append(fea)
    feas = torch.cat(feas, dim=1)
    return feas

  def stat_loss(self, fea):
    fea = fea.flatten(start_dim=2)
    var_mean = torch.var_mean(fea, dim=2)
    fea = torch.cat(var_mean, dim=1)
    return fea


@MODEL_REGISTRY.register(name_prefix=__name__)
class HRNetW64Loss(torch.nn.Module):
  def __init__(self,
               use_stat_loss=False,
               layers=None,
               loss_w_dict=None,
               **kwargs):
    super().__init__()
    import timm

    self.use_stat_loss = use_stat_loss

    if layers is None:
      layers = self.layers
    print(f"hrnet_w64 layers: {layers}")
    net = timm.create_model('hrnet_w64', pretrained=True)
    self.net = FeatureExtractor(net, layers=layers)

    self.loss_w_dict = loss_w_dict
    if loss_w_dict is None:
      self.loss_w_dict = self.loss_weight
    print(f"Swin loss_w_dict: {json.dumps(self.loss_w_dict)}")
    pass

  @property
  def layers(self):
    layers = ['act1', 'act2', 'layer1', 'stage2', 'stage3', 'stage4', 'final_layer']
    return layers

  @property
  def loss_weight(self):
    layers = ['act1', 'act2', 'layer1', 'stage2', 'stage3', 'stage4', 'final_layer']
    weights = [1, 1, 1, 1, 1, 1, 1]
    assert len(layers) == len(weights)
    loss_w_dict = {}
    for layer, w in zip(layers, weights):
      loss_w_dict[layer] = w
    return loss_w_dict


  def forward(self, x, *args, loss_w_dict=None, use_stat_loss=None, **kwargs):
    self.net.eval()

    if use_stat_loss is None:
      use_stat_loss = self.use_stat_loss

    feas_dict = self.net(x)
    feas = []
    for k, v in feas_dict.items():
      if k.startswith('stage'):
        fea = v[0]
      else:
        fea = v
      # b, c, h, w = fea.shape
      if use_stat_loss:
        fea = self.stat_loss(fea)
      else:
        fea = fea.flatten(start_dim=1)

      if loss_w_dict is None:
        fea = fea * self.loss_w_dict[k]
      else:
        fea = fea * loss_w_dict[k]
      feas.append(fea)
    feas = torch.cat(feas, dim=1)
    return feas

  def stat_loss(self, fea):
    fea = fea.flatten(start_dim=2)
    var_mean = torch.var_mean(fea, dim=2)
    fea = torch.cat(var_mean, dim=1)
    return fea


@MODEL_REGISTRY.register(name_prefix=__name__)
class SSLResNext101Loss(torch.nn.Module):
  def __init__(self,
               use_stat_loss=False,
               layers=None,
               loss_w_dict=None,
               **kwargs):
    super().__init__()
    import timm

    self.use_stat_loss = use_stat_loss

    if layers is None:
      layers = self.layers
    print(f"ssl_resnet50 layers: {layers}")
    net = timm.create_model('ssl_resnet50', pretrained=True)
    self.net = FeatureExtractor(net, layers=layers)

    self.loss_w_dict = loss_w_dict
    if loss_w_dict is None:
      self.loss_w_dict = self.loss_weight
    print(f"ssl_resnet50 loss_w_dict: {json.dumps(self.loss_w_dict)}")
    pass

  @property
  def layers(self):
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'global_pool']
    return layers

  @property
  def loss_weight(self):
    layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'global_pool']
    weights = [1, 1, 1, 1, 1, 1]
    assert len(layers) == len(weights)
    loss_w_dict = {}
    for layer, w in zip(layers, weights):
      loss_w_dict[layer] = w
    return loss_w_dict


  def forward(self, x, *args, loss_w_dict=None, use_stat_loss=None, **kwargs):
    self.net.eval()

    if use_stat_loss is None:
      use_stat_loss = self.use_stat_loss

    feas_dict = self.net(x)
    feas = []
    for k, v in feas_dict.items():
      fea = v
      # b, c, h, w = fea.shape
      if use_stat_loss:
        fea = self.stat_loss(fea)
      else:
        fea = fea.flatten(start_dim=1)

      if loss_w_dict is None:
        fea = fea * self.loss_w_dict[k]
      else:
        fea = fea * loss_w_dict[k]
      feas.append(fea)
    feas = torch.cat(feas, dim=1)
    return feas

  def stat_loss(self, fea):
    fea = fea.flatten(start_dim=2)
    var_mean = torch.var_mean(fea, dim=2)
    fea = torch.cat(var_mean, dim=1)
    return fea



@MODEL_REGISTRY.register(name_prefix=__name__)
class DeitSmallPatch16Loss(torch.nn.Module):
  def __init__(self,
               model_name='vit_deit_small_patch16_224',
               resize_input=False,
               use_stat_loss=False,
               layers=None,
               loss_w_dict=None,
               **kwargs):
    super().__init__()
    import timm
    from timm.models.vision_transformer import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    assert model_name in ['vit_deit_small_patch16_224',
                          'vit_deit_small_distilled_patch16_224',
                          'dino_deitsmall16']

    self.mean = IMAGENET_DEFAULT_MEAN
    self.std = IMAGENET_DEFAULT_STD
    self.transform = trans.Normalize(mean=self.mean, std=self.std)

    self.resize_input = resize_input
    self.use_stat_loss = use_stat_loss

    if layers is None:
      layers = self.layers
    print(f"{model_name} layers: {layers}")

    if model_name in ['vit_deit_small_patch16_224', 'vit_deit_small_distilled_patch16_224']:
      net = timm.create_model(model_name, pretrained=True, features_only=False)
    elif model_name == 'dino_deitsmall16':
      dino_model = os.path.expanduser("~/.cache/torch/hub/checkpoints/dino_deitsmall16_pretrain.pth")
      net = timm.create_model('vit_deit_small_patch16_224', pretrained=False, features_only=False)
      net_ckpt = Checkpointer(net)
      net_ckpt.load_state_dict_from_file(dino_model)
      del net_ckpt

    self.net = FeatureExtractor(net, layers=layers)

    self.loss_w_dict = loss_w_dict
    if loss_w_dict is None:
      self.loss_w_dict = self.loss_weight
    print(f"{model_name} loss_w_dict: {json.dumps(self.loss_w_dict)}")
    pass

  @property
  def layers(self):
    layers = ['patch_embed', 'blocks.0', 'blocks.3', 'blocks.7', 'blocks.11']
    return layers

  @property
  def loss_weight(self):
    layers = self.layers
    weights = [1, 1, 1, 1, 1]
    assert len(layers) == len(weights)
    loss_w_dict = {}
    for layer, w in zip(layers, weights):
      loss_w_dict[layer] = w
    return loss_w_dict


  def forward(self, x, *args, loss_w_dict=None, use_stat_loss=None, **kwargs):
    """
    x: [-1 , 1]
    """
    self.net.eval()

    if use_stat_loss is None:
      use_stat_loss = self.use_stat_loss

    x = (x + 1) / 2.
    x = self.transform(x)
    x = F.interpolate(x, size=(224, 224), mode='area')

    feas_dict = self.net(x)
    feas = []
    for k, v in feas_dict.items():
      fea = v
      # b, c, h, w = fea.shape
      if use_stat_loss:
        assert 0
        fea = self.stat_loss(fea)
      else:
        fea = fea.flatten(start_dim=1)

      if loss_w_dict is None:
        fea = fea * self.loss_w_dict[k]
      else:
        fea = fea * loss_w_dict[k]
      feas.append(fea)
    feas = torch.cat(feas, dim=1)
    return feas

  def stat_loss(self, fea):
    fea = fea.flatten(start_dim=2)
    var_mean = torch.var_mean(fea, dim=2)
    fea = torch.cat(var_mean, dim=1)
    return fea


@MODEL_REGISTRY.register(name_prefix=__name__)
class ResNet50Loss(torch.nn.Module):
  def __init__(self,
               model_name='resnet50_conv',
               resize_input=False,
               use_stat_loss=False,
               layers=None,
               loss_w_dict=None,
               downsample_scale=None,
               **kwargs):
    super().__init__()
    import timm
    from timm.models.resnet import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

    assert model_name in ['resnet50_conv',
                          'dino_resnet50_conv',
                          'random_resnet50_conv',
                          'resnet50_relu']

    self.mean = IMAGENET_DEFAULT_MEAN
    self.std = IMAGENET_DEFAULT_STD
    self.transform = trans.Normalize(mean=self.mean, std=self.std)

    self.resize_input = resize_input
    self.downsample_scale = downsample_scale
    self.use_stat_loss = use_stat_loss

    if layers is None:
      layers = self.layers

    if model_name in ['resnet50_conv']:
      net = timm.create_model('resnet50', pretrained=True, features_only=False)
    elif model_name in ['random_resnet50_conv']:
      print("random initializing resnet50")
      net = timm.create_model('resnet50', pretrained=False, features_only=False)
    elif model_name == 'dino_resnet50_conv':
      dino_model = os.path.expanduser("~/.cache/torch/hub/checkpoints/dino_resnet50_pretrain.pth")
      net = timm.create_model('resnet50', pretrained=False, features_only=False)
      net_ckpt = Checkpointer(net)
      net_ckpt.load_state_dict_from_file(dino_model)
      del net_ckpt
    elif model_name in ['resnet50_relu']:
      net = timm.create_model('resnet50', pretrained=True, features_only=False)
      layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4']

    self.loss_w_dict = loss_w_dict
    if loss_w_dict is None:
      self.loss_w_dict = self.loss_weight
    print(f"{model_name} loss_w_dict: {json.dumps(self.loss_w_dict)}")

    print(f"{model_name} layers: {layers}")
    self.net = FeatureExtractor(net, layers=layers)
    pass

  @property
  def layers(self):
    layers = ['conv1', 'layer1.2.conv1', 'layer2.3.conv1', 'layer3.5.conv1', 'layer4.2.conv1']
    return layers

  @property
  def loss_weight(self):
    layers = self.layers
    weights = [1, 1, 1, 1, 1]
    assert len(layers) == len(weights)
    loss_w_dict = {}
    for layer, w in zip(layers, weights):
      loss_w_dict[layer] = w
    return loss_w_dict


  def forward(self, x, *args, loss_w_dict=None, use_stat_loss=None, **kwargs):
    """
    x: [-1 , 1]
    """
    self.net.eval()

    if use_stat_loss is None:
      use_stat_loss = self.use_stat_loss

    x = (x + 1) / 2.
    x = self.transform(x)
    if self.resize_input:
      if self.downsample_scale is not None:
        downsample_size = int(x.shape[-2] // self.downsample_scale), int(x.shape[-1] // self.downsample_scale)
      else:
        downsample_size = (256, 256)

      x = F.interpolate(x, size=downsample_size, mode='area')

    feas_dict = self.net(x)
    feas = []
    for k, v in feas_dict.items():
      fea = v
      # b, c, h, w = fea.shape
      if use_stat_loss:
        fea = self.stat_loss(fea)
      else:
        fea = fea.flatten(start_dim=1)

      if loss_w_dict is None:
        fea = fea * self.loss_w_dict[k]
      else:
        fea = fea * loss_w_dict[k]
      feas.append(fea)
    feas = torch.cat(feas, dim=1)
    return feas

  def stat_loss(self, fea):
    fea = fea.flatten(start_dim=2)
    var_mean = torch.var_mean(fea, dim=2)
    fea = torch.cat(var_mean, dim=1)
    return fea




