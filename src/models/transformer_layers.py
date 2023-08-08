"""
* Copyright (c) 2023 OPPO. All rights reserved.
*
*
* Licensed under the Apache License, Version 2.0 (the "License"):
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and 
* limitations under the License.
*
* NOTICE:
* Code slightly modified from original repo: https://github.com/stevenlsw/hoi-forecast/tree/master/networks,
* which is protected under the MIT License: https://github.com/stevenlsw/hoi-forecast/blob/master/LICENSE.
* We sincerely thank the contributors of the original repo.
*
"""

import torch
import torch.nn as nn

import os, sys
HoiForecast_ROOT = os.path.join(os.path.dirname(__file__), '../../third_party/hoi-forecast')
sys.path.append(HoiForecast_ROOT)
""" 
The followings are hoi-forecast classes
"""
from networks.net_utils import DropPath
from networks.embedding import PositionalEncoding
from networks.layer import MultiHeadAttention, Mlp



def get_pad_mask(seq, pad_idx=0):
    if not isinstance(pad_idx, int):
        raise TypeError("<pad_index> has to be an int!")
    if seq.dim() == 3:
        return seq != pad_idx  # (B, T, T)
    return (seq != pad_idx).unsqueeze(1) # (B, 1, T)


class EncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(EncoderBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, mask=None):
        src_mask = None
        if mask is not None:
            src_mask = get_pad_mask(mask, pad_idx=0)  # (B, 1, T)
        x_norm = self.norm1(x)
        x = x + self.drop_path(self.attn(q=x_norm, k=x_norm, v=x_norm, mask=src_mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Encoder(nn.Module):
    """ Transformer Encoder """
    def __init__(self, num_frames, embed_dim=512, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, dropout=0.):
        super(Encoder, self).__init__()
        # dropout layer
        self.dropout = nn.Dropout(dropout)
        # temporal positional encoding
        self.time_embed = PositionalEncoding(embed_dim, max_len=num_frames)
        self.time_drop = nn.Dropout(p=drop_rate)
        
        # construct encoder blocks with increasing drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.encoder_blocks = nn.ModuleList([EncoderBlock(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer
        ) for i in range(depth)])
        
        # layer normalization
        self.norm = norm_layer(embed_dim)
        
    
    def forward(self, x, mask=None):
        """ x: (B, T, D)
            mask: # (B, T)
        """
        x = self.time_embed(x)
        x = self.time_drop(x)
        
        for blk in self.encoder_blocks:
            x = blk(x, mask=mask)  # (B, T, D)
        
        x = self.norm(x)
        return x
