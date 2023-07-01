# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# pylint: skip-file

# Adapted from score_sde_pytorch: https://github.com/yang-song/score_sde_pytorch
# Removed progressive module and Fourier timestep embeddings and FIR kernel

from . import layers, normalization, utils
import torch.nn as nn
import functools
import torch
from model.attention import SpatialTransformer
from abc import abstractmethod

ResnetBlockDDPM = layers.ResnetBlockDDPMpp
ResnetBlockBigGAN = layers.ResnetBlockBigGANpp
# Combine = layers.Combine
conv3x3 = layers.conv3x3
conv1x1 = layers.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


class TimestepBlock(nn.Module):
  """
  Any module where forward() takes timestep embeddings as a second argument.
  """

  @abstractmethod
  def forward(self, x, emb):
    """
    Apply the module to `x` given `emb` timestep embeddings.
    """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, context=None):
        for layer in self:
          try:
            if type(layer) in (layers.ResnetBlockDDPMpp, layers.ResnetBlockBigGANpp):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
          except Exception as e:
            print('In timestep embed sequential, layer:', layer)
            print('h shape:', x.shape)
            print('emb shape:', emb.shape)
            print('context shape:', context.shape)
            raise e
        return x

class UNetModel(nn.Module):
  """UNet model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.max_res_num // (2 ** i) for i in range(num_resolutions)]

    self.skip_rescale = skip_rescale = config.model.skip_rescale
    self.resblock_type = resblock_type = config.model.resblock_type.lower()
    init_scale = config.model.init_scale

    self.embedding_type = embedding_type = config.model.embedding_type.lower()
    self.n_heads = n_heads = config.model.n_heads
    self.context_dim = context_dim = config.model.context_dim

    assert embedding_type in ['fourier', 'positional']

    modules = []
    embed_dim = nf
    modules.append(nn.Linear(embed_dim, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    modules.append(nn.Linear(nf * 4, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    self.pre_blocks = nn.ModuleList(modules)

    AttnBlock = functools.partial(layers.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layers.Upsample,
                                 with_conv=resamp_with_conv)

    Downsample = functools.partial(layers.Downsample,
                                   with_conv=resamp_with_conv)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')
    channels = config.data.num_channels
    modules.append(conv3x3(channels, nf))
    self.pre_conv = conv3x3(channels, nf)

    # Downsampling block
    self.input_blocks = nn.ModuleList([])
    hs_c = [nf]
    in_ch = nf
    self.input_channels = [nf]

    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        cur_layers = [ResnetBlock(in_ch=in_ch, out_ch=out_ch)]

        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          cur_layers.append(AttnBlock(channels=in_ch))
          cur_layers.append(SpatialTransformer(in_channels=in_ch,
                                               n_heads=n_heads,
                                               d_head=in_ch // n_heads,
                                               context_dim=context_dim))
        self.input_blocks.append(TimestepEmbedSequential(*cur_layers))
        hs_c.append(in_ch)
        self.input_channels.append(in_ch)

      if i_level != num_resolutions - 1:
        self.input_blocks.append(
          TimestepEmbedSequential(
            Downsample(in_ch=in_ch) if resblock_type == 'ddpm'
            else ResnetBlock(down=True, in_ch=in_ch)
          )
        )
        hs_c.append(in_ch)
        self.input_channels.append(in_ch)

    in_ch = hs_c[-1]
    self.mid_channel = self.input_channels[-1]
    self.mid_blocks = TimestepEmbedSequential(
      ResnetBlock(in_ch=self.mid_channel),
      AttnBlock(channels=self.mid_channel),
      SpatialTransformer(in_channels=in_ch,
                         n_heads=n_heads,
                         d_head=in_ch // n_heads,
                         context_dim=context_dim),
      ResnetBlock(in_ch=self.mid_channel)
    )

    # Upsampling block
    self.out_blocks = nn.ModuleList([])
    self.out_channels = []
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        cur_layers = [ResnetBlock(in_ch=in_ch + self.input_channels.pop(), out_ch=out_ch)]
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          cur_layers.append(AttnBlock(channels=in_ch))
          cur_layers.append(SpatialTransformer(in_channels=in_ch,
                                               n_heads=n_heads,
                                               d_head=in_ch // n_heads,
                                               context_dim=context_dim))

        if i_level != 0 and i_block == num_res_blocks:
          if resblock_type == 'ddpm':
            cur_layers.append(Upsample(in_ch=in_ch))
          else:
            cur_layers.append(ResnetBlock(in_ch=in_ch, up=True))
        self.out_blocks.append(TimestepEmbedSequential(*cur_layers))


    self.out = nn.ModuleList([])


    self.out.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch, eps=1e-6))
    self.out.append(self.act)
    self.out.append(conv3x3(in_ch, channels, init_scale=init_scale))


  def forward(self, x, time_cond, text_emb=None):

    # Sinusoidal positional embeddings.
    timesteps = time_cond
    used_sigmas = self.sigmas[time_cond.long()]
    temb = layers.get_timestep_embedding(timesteps, self.nf)

    # pre blocks
    for module in self.pre_blocks:
      temb = module(temb)
    x = x.float()
    h = self.pre_conv(x)
    hs = [h]

    # Down Sampling blocks
    for i, module in enumerate(self.input_blocks):
      try:
        h = module(h, temb, text_emb)
        # print('in down block', i, 'h shape:', h.shape)
        hs.append(h)
      except Exception as e:
        print(f'in down block {i}, module is:', module)
        print('h shape:', h.shape)
        print('temb shape:', temb.shape)
        print('text_emb shape:', text_emb.shape)

    # Mid Block
    h = self.mid_blocks(h, temb, text_emb)

    # Up Sampling blocks
    for module in self.out_blocks:
        h = torch.cat([h, hs.pop()], dim=1)
        h = module(h, temb, text_emb)


    assert not hs

    for out in self.out:
      h = out(h)

    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h

class NCSNpp(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = get_act(config)
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.model.nf
    ch_mult = config.model.ch_mult
    self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
    dropout = config.model.dropout
    resamp_with_conv = config.model.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.data.max_res_num // (2 ** i) for i in range(num_resolutions)]

    self.skip_rescale = skip_rescale = config.model.skip_rescale
    self.resblock_type = resblock_type = config.model.resblock_type.lower()
    init_scale = config.model.init_scale

    self.embedding_type = embedding_type = config.model.embedding_type.lower()

    # self.self_attn = torch.nn.MultiheadAttention(nf, 8, dropout=0.1)
    # self.cross_attn = torch.nn.MultiheadAttention(nf, 8, dropout=0.1)

    assert embedding_type in ['fourier', 'positional']

    modules = []
    embed_dim = nf
    modules.append(nn.Linear(embed_dim, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)
    modules.append(nn.Linear(nf * 4, nf * 4))
    modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
    nn.init.zeros_(modules[-1].bias)

    AttnBlock = functools.partial(layers.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layers.Upsample,
                                 with_conv=resamp_with_conv)

    Downsample = functools.partial(layers.Downsample,
                                   with_conv=resamp_with_conv)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    channels = config.data.num_channels
    modules.append(conv3x3(channels, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        # print('add resblock', i_level, len(modules))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
          # print('add attnblock', i_level, len(modules))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))
          # print('add resblock non_last level', i_level, len(modules))
        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                num_channels=in_ch, eps=1e-6))
    modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond, text_emb=None):
    batch_size = x.shape[0]
    text_emb_size = 1024
    text_emb = torch.randn(batch_size, text_emb_size).to(x.device)

    modules = self.all_modules
    m_idx = 0
    # Sinusoidal positional embeddings.
    timesteps = time_cond
    used_sigmas = self.sigmas[time_cond.long()]
    temb = layers.get_timestep_embedding(timesteps, self.nf)

    temb = modules[m_idx](temb)
    m_idx += 1
    temb = modules[m_idx](self.act(temb))
    m_idx += 1

    # Downsampling block
    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
        # if True:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb)
          m_idx += 1


        hs.append(h)


    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1


    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
      # if True:
        h = modules[m_idx](h)
        m_idx += 1

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb)
          m_idx += 1

    assert not hs

    h = self.act(modules[m_idx](h))
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1

    assert m_idx == len(modules)
    if self.config.model.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h