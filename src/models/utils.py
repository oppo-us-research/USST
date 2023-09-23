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
from einops import rearrange



def get_masks(batch_size, ratios, max_frames, nframes, device):
    """Get the multi-hot masks for observed and unobserved frames """
    num_ratios = ratios.size(1)
    mask_o = torch.zeros((batch_size, num_ratios, max_frames)).to(device, non_blocking=True)  # (B, N, T)
    mask_u = torch.zeros((batch_size, num_ratios, max_frames)).to(device, non_blocking=True)  # (B, N, T)
    last_frames = torch.zeros((batch_size, num_ratios)).long()
    for b in range(batch_size):  # loop for each video clip
        # get the number observed frames and full frames
        num_full = int(nframes[b])
        num_obs = torch.floor(num_full * ratios[b]).to(torch.long)  # (2,)
        last_frames[b] = num_obs - 1  # index
        # get masks
        for i, n_o in enumerate(num_obs):
            mask_o[b, i, :n_o] = 1
            mask_u[b, i, n_o: num_full] = 1
    return mask_o, mask_u, last_frames


def get_rgb_observed(rgb, num_ratios, mask_o):
    """ get the observed part of RGB frames"""
    rgb_input = rgb[:, None].repeat(1, num_ratios, 1, 1, 1, 1)
    rgb_input = rearrange(rgb_input, 'b n c t h w -> (b n) c t h w')  # (B*N, C, T, H, W)
    rgb_mask = mask_o[:, None, :].repeat(1, rgb.size(1), 1).unsqueeze(-1).unsqueeze(-1)  # (B*N, C, T, 1, 1)
    rgb_input = rgb_input * rgb_mask
    return rgb_input  # (B*N, C, T, H, W)


def get_traj_observed(traj_all, num_ratios, mask_o):
    """ get the observed part of trajectory"""
    traj_input = traj_all[:, None].repeat(1, num_ratios, 1, 1)  # (B, N, T, 3)
    traj_input = rearrange(traj_input, 'b n t c-> (b n) t c')  # (B*N, T, 3)
    traj_mask = mask_o.unsqueeze(-1)  # (B*N, T, 1)
    traj_input = traj_input * traj_mask
    return traj_input  # (B*N, T, 3)


def gather_results(pred_dict, nframes, ratios, last_obs_frames, sup_obs=False):
    """ pred_dict: 'traj': (B*N, T, 3), 'kld': (B*N, T)
        nframes: (B,)
        ratios: (B, N)
        last_obs_frames: (B, N)
    """
    def split_dict(pred_dict, start, end):
        result = dict()
        for k, v in pred_dict.items():
            result.update({k: v[p, start: end]})
        return result
        
    batch_size, num_ratios = ratios.size()
    outputs = [dict() for _ in range(batch_size)]
    outputs_obs = [dict() for _ in range(batch_size)] if sup_obs else None
    for b in range(batch_size):
        for n in range(num_ratios):
            p = b * num_ratios + n
            key = '{:.1f}'.format(float(ratios[b, n]))  # we use ratio string as the key
            # unobserved part
            outputs[b][key] = split_dict(pred_dict, start=last_obs_frames[b, n]+1, end=int(nframes[b]))
            # observed part
            if sup_obs:
                outputs_obs[b][key] = split_dict(pred_dict, start=0, end=last_obs_frames[b, n]+1)
    return outputs, outputs_obs