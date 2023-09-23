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
"""

import torch
import torch.nn as nn
import os, sys
HoiForecast_ROOT = os.path.join(os.path.dirname(__file__), '../../third_party/hoi-forecast')
sys.path.append(HoiForecast_ROOT)
""" 
The followings are hoi-forecast classes
"""
from networks.embedding import PositionalEncoding
from networks.layer import MultiHeadAttention



class ProbTranV3(nn.Module):
    def __init__(self, vis_dim, loc_dim, modalities, seq_len=40, h_dim=128, z_dim=16, pred_link=False):
        super(ProbTranV3, self).__init__()
        self.modalities = modalities
        assert any(mod in modalities for mod in ['rgb', 'loc'])  # at least one of modalitties should exist
        # input dimension
        self.x_dim = vis_dim + loc_dim
        if 'rgb' not in modalities:
            self.x_dim = loc_dim
        if 'loc' not in modalities:
            self.x_dim = vis_dim
        # predictive link dimension
        self.loc_dim = loc_dim
        self.h_dim = h_dim
        self.pred_link = pred_link

        # xt --> ht
        self.phi_x = nn.Sequential(
            nn.Linear(self.x_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim))
        self.time_embed_h = PositionalEncoding(h_dim, max_len=seq_len)
        self.layer_norm_h = nn.LayerNorm(h_dim)
        # w(1:t-1) --> w_bar_t
        self.attn_w = MultiHeadAttention(h_dim // 2, num_heads=8, qkv_bias=False,
                                       qk_scale=None, attn_drop=0, proj_drop=0)
        self.layer_norm_wb = nn.LayerNorm(h_dim)
        # h(1:C) --> w_hat_t
        self.attn_h = MultiHeadAttention(h_dim, num_heads=8, qkv_bias=False,
                                       qk_scale=None, attn_drop=0, proj_drop=0)
        self.layer_norm_wh = nn.LayerNorm(h_dim * 2)
        # Prior: w_hat_t --> zt
        self.prior_mlp = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim))
        self.prior_mean = nn.Sequential(
            nn.Linear(h_dim, z_dim),)
        # zt --> wt
        self.phi_z = nn.Sequential(
            nn.Linear(h_dim * 2 + z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim // 2))
        self.time_embed_w = PositionalEncoding(h_dim // 2, max_len=seq_len)
        self.layer_norm_w = nn.LayerNorm(h_dim // 2)
        # w(1:T) --> x(1:T)
        dim_phi_w = h_dim // 2 + self.loc_dim if self.pred_link and 'loc' in self.modalities else h_dim // 2
        self.phi_w = nn.Sequential(
            nn.Linear(dim_phi_w, self.loc_dim),
            nn.ReLU(),
            nn.Linear(self.loc_dim, self.loc_dim))
    

    def forward(self, mask, vis_obs=None, traj_obs=None):
        """
            \log p(z1:T | x1:C) = \sum\log pθ(zt | z1:t-1, x1:C )
            vis_obs: (B*N, T, D)
            traj_obs: (B*N, T, D)
            mask: (B*N, T), mask of the observed time steps
        """
        batch_size, seq_len = mask.size()[:2]
        device = mask.device

        wt = torch.zeros([batch_size, 1, self.h_dim // 2], device=device)
        w_accum = [wt]  # [(B*N, 1, d/2)]

        x_obs = torch.cat([obs for obs in [vis_obs, traj_obs] if obs is not None], dim=-1)
        # ht = LayerNorm(MLP(xt) + Position(t))
        h = self.layer_norm_h(self.time_embed_h(self.phi_x(x_obs)))

        for t in range(seq_len):
            # w¯ t = LayerNorm([wt−1; Attention(wt−1, w1:t−1, w1:t−1)])
            attn_w = self.attn_w(q=wt, k=torch.cat(w_accum, dim=1), v=torch.cat(w_accum, dim=1), mask=mask[:, :(t+1)].unsqueeze(1))
            wbar_t = self.layer_norm_wb(torch.cat([wt, attn_w], dim=-1))  # (B*N, 1, d)

            # wˆ t = LayerNorm([w¯ t; Attention(w¯ t, h1:C , h1:C )])
            attn_h = self.attn_h(q=wbar_t, k=h, v=h, mask=mask.unsqueeze(1))
            what_t = self.layer_norm_wh(torch.cat([wbar_t, attn_h], dim=-1))  # (B*N, 1, 2d)

            # zt = MLP(wˆ t)
            prior_shared = self.prior_mlp(what_t)
            zt = self.prior_mean(prior_shared)  # (B*N, 1, dz)

            # wt = LayerNorm(MLP([wˆ t; zt]) + Position(t))
            fuse_wz = self.phi_z(torch.cat([what_t, zt], dim=-1))
            wt = self.layer_norm_w(fuse_wz + self.time_embed_w.pe[:, t:(t+1)])
            w_accum.append(wt)
        
        phi_w_in = torch.cat(w_accum[1:], dim=1)  # (B*N, T, d/2)
        if self.pred_link and 'loc' in self.modalities:
            loc_tm1 = torch.cat((torch.zeros(batch_size, 1, self.loc_dim).to(device), traj_obs[:, :-1, :]), dim=1)  # (B*N, T, d)
            phi_w_in = torch.cat([phi_w_in, loc_tm1], dim=-1)
        x_pred = self.phi_w(phi_w_in)  # (B*N, T, D)

        return x_pred, dict()
