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
from einops import rearrange
from src.models.Heads import MlpHead, UncertaintyHead
from src.models.EmbedTrajectory import MlpModel
from src.models.EmbedFrame import FrameEmbedding
from src.models.transformer_layers import Encoder
from src.models.utils import get_masks, get_rgb_observed, get_traj_observed, gather_results
from src.models.ProTranV3 import ProbTranV3


class TransformerSSM(nn.Module):
    def __init__(self, cfg, seq_len=64, input_size=224):
        super(TransformerSSM, self).__init__()
        self.embed_dim = cfg.feat_dim
        self.loc_dim = cfg.loc_dim
        self.modalities = cfg.modalities
        assert any(mod in cfg.modalities for mod in ['rgb', 'loc'])  # at least one of modalitties should exist
        self.transformer_params = vars(cfg.transformer) if hasattr(cfg, 'transformer') else dict()
        self.use_odom = getattr(cfg, 'use_odom', False)
        self.sup_obs = getattr(cfg, 'sup_obs', False)
        self.ssm_params = vars(cfg.ssm)
        self.ssm_params.update(seq_len=seq_len)
        self.aleatoric_uncertainty = getattr(cfg, 'aleatoric_uncertainty', None)
        self.pred_velocity = getattr(cfg, 'pred_velocity', False)
        
        # output config
        self.target = getattr(cfg, 'target', '3d')  # ['2d', '3d']
        self.ignore_depth = getattr(cfg, 'ignore_depth', False)
        self.output_size = 3 if self.target == '3d' and (not self.ignore_depth) else 2
        self.centralize = getattr(cfg, 'centralize', False)
        
        # input feature extraction
        input_dim = 0
        if 'rgb' in self.modalities:
            # frame-level feature extraction
            self.global_feature = FrameEmbedding(cfg, input_size=input_size)
            input_dim += self.embed_dim
        if 'loc' in self.modalities:
            # location to feature
            self.location_feature = MlpModel(self.output_size, self.loc_dim // 2, self.loc_dim)
            input_dim += self.loc_dim
        
        # state transition model: pθ(z1:T | x1:C )
        self.transition = ProbTranV3(self.embed_dim, self.loc_dim, self.modalities, **self.ssm_params)
        # transformer encoders
        self.encoder_vis = Encoder(num_frames=seq_len, embed_dim=self.embed_dim, **self.transformer_params)
        self.encoder_traj = Encoder(num_frames=seq_len, embed_dim=self.loc_dim, **self.transformer_params)

        out_layer = nn.Tanh if self.centralize else nn.Sigmoid
        dim_pred = self.loc_dim
        self.predict = MlpHead(dim_pred, dim_pred // 2, self.output_size, out_layer=out_layer)

        if self.aleatoric_uncertainty:
            self.predict_uncertainty = UncertaintyHead(dim_pred, dim_pred //2, cfg=cfg.aleatoric_uncertainty)
        
        if self.pred_velocity:
            self.predict_velo = MlpHead(dim_pred, dim_pred // 2, self.output_size, bn_layer=nn.LayerNorm, out_layer=nn.Tanh)
    
    
    def _get_rgb_features(self, x, num_ratios, mask):
        """
            x: (B, C, T, H, W)
            num_ratios: ()
            mask: (B*N, T)
        """
        rgb_input = get_rgb_observed(x, num_ratios, mask)  # (B*N, C, T, H, W)
        batch_size, max_frames = rgb_input.size(0), x.size(2)
        # get global appearance features
        rgb_input = rearrange(rgb_input, 'b c t h w -> (b t) c h w')
        rgb_feat = self.global_feature(rgb_input)  # (B*T, D)
        rgb_feat = rearrange(rgb_feat, '(b t) d -> b t d', b=batch_size, t=max_frames)  # (B*N, T, D)
        return rgb_feat


    def _get_loc_features(self, traj, num_ratios, mask):
        """
            traj: (B, T, 3)
            num_ratios: ()
            mask: (B*N, T)
        """
        traj_input = get_traj_observed(traj, num_ratios, mask)  # (B*N, T, 3)
        loc_feat = self.location_feature(traj_input)  # (B*N, T, D)
        return loc_feat


    def get_input_features(self, x, traj_all, num_ratios, mask):
        # get rgb & trajectory features of full observation
        x_feat = self._get_rgb_features(x, num_ratios, mask)  # (B*N, T, D)
        if 'loc' in self.modalities:
            y_feat = self._get_loc_features(traj_all, num_ratios, mask)  # (B*N, T, D)
            x_feat = torch.cat([x_feat, y_feat], dim=-1)
        return x_feat
    

    def forward(self, x, nframes, ratios, traj):
        """
            x: (B, C, T, H, W)
            nframes: (B,)
            ratios: (B, N)
            traj: (B, T, 3) or (B, T*(T+1)/2, 3)
        """
        batch_size, max_frames, num_ratios = x.size(0), traj.size(1), ratios.size(1)
        device = x.device
        # get trajectory target
        traj_all = traj[:, :, :self.output_size]

        # get masks of observed and unobserved frames
        mask_o, mask_u, last_obs_frames = get_masks(batch_size, ratios, max_frames, nframes, device)  # (B, N, T)
        mask_o = rearrange(mask_o, 'b n t -> (b n) t')  # (B*N, T)

        vis_feat, traj_feat = None, None
        if 'rgb' in self.modalities:
            # visual feature
            vis_feat = self._get_rgb_features(x, num_ratios, mask_o)  # (B*N, T, D)
            vis_feat = self.encoder_vis(vis_feat, mask_o)
        if 'loc' in self.modalities:
            # trajectory feature
            traj_feat = self._get_loc_features(traj_all, num_ratios, mask_o)
            traj_feat = self.encoder_traj(traj_feat, mask_o)
        # state transition: x_obs (+ y_obs) --> x_future
        future_pred, pred_dict = self.transition(mask_o, vis_obs=vis_feat, traj_obs=traj_feat)

        # trajectory prediction
        preds = self.predict(future_pred)  # (B*N, T, 3)
        pred_dict.update({'traj': preds})

        # uncertainty prediction
        if self.aleatoric_uncertainty:
            uncts = self.predict_uncertainty(future_pred)  # (B*N, T, 3)
            pred_dict.update({'unct': uncts})
        
        # velocity prediction
        if self.pred_velocity:
            velos = self.predict_velo(future_pred.detach())  # (B*N, T, 3)
            pred_dict.update({'velo': velos})
        
        """ ----------- parse prediction results ----------- """
        outputs, outputs_obs = gather_results(pred_dict, nframes, ratios, last_obs_frames, self.sup_obs)
        
        return outputs, outputs_obs
    

    def inference(self, x, nframes, ratios, traj=None):
        return self(x, nframes, ratios, traj)

    
