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
from src.H2OLoader import normalize_traj, denormalize_traj, world_to_camera, XYZ_to_uv, INTRINSICS
import numpy as np


def gather_eval_results(outputs, nframes, trajectories, cam_poses, ignore_depth=False, use_global=False, centralize=False, return_unct=False):
    """ gather predictions and gts for evaluation purpose
        return: all_preds (N, 9, 3)
                all_gts (N, 9, 3)
    """
    trajectories = trajectories.cpu().numpy()
    cam_poses = cam_poses.numpy()
    all_preds, all_gts, all_poses = {}, {}, {}
    if return_unct: all_uncts = {}
    target = '3d' if trajectories.shape[-1] == 3 else '2d'
    # parse results
    for b, ratio_preds in enumerate(outputs):
        # get the number observed frames and full frames
        num_full = nframes[b]
        traj_gt = trajectories[b]  # (T,3)
        campose = cam_poses[b]
        if ignore_depth: traj_gt = traj_gt[:, :2]

        # de-centralize and de-normalize
        if centralize: traj_gt = (traj_gt + 1.0) * 0.5  # in [0, 1]
        traj_gt = denormalize_traj(traj_gt, target=target, use_global=use_global)
        
        for i, (r, results) in enumerate(ratio_preds.items()):
            num_obs = torch.floor(num_full * float(r)).to(torch.long)
            preds = results['traj'].cpu().numpy()
            gts = traj_gt[num_obs:num_full, :]
            poses = campose[num_obs:num_full]  # (N,4,4)

            # de-centralize and de-normalize
            if centralize: preds = (preds + 1.0) * 0.5  # in [0, 1]
            preds = denormalize_traj(preds, target=target, use_global=use_global)
            # get uncertainty if any
            if return_unct:
                logvars = results['unct'].cpu().numpy() if 'unct' in results else np.ones_like(preds)
                uncts = np.exp(logvars)
            
            # gather results
            if r in all_preds and r in all_gts:
                all_preds[r].append(preds)  # (M, 3)
                all_gts[r].append(gts)  # (M, 3)
                all_poses[r].append(poses)
                if return_unct: all_uncts[r].append(uncts)
            else:
                all_preds[r] = [preds]
                all_gts[r] = [gts]
                all_poses[r] = [poses]
                if return_unct: all_uncts[r] = [uncts]
    if return_unct: 
        return all_preds, all_uncts, all_gts, all_poses
    return all_preds, all_gts, all_poses


def traj_transform(traj, cam2world=None, target='3d', eval_space='3d', use_global=True):
    """ transform trajectory space
    """
    if target == '3d' and eval_space in ['2d', 'norm2d']:
        if use_global:
            traj = world_to_camera(traj, cam2world)
        # camera 3D to 2D
        traj = XYZ_to_uv(traj, INTRINSICS)

        if eval_space == 'norm2d':
            traj = normalize_traj(traj, target='2d')
    
    if target == '2d' and eval_space == 'norm2d':
        traj = normalize_traj(traj, target='2d')
    
    return traj



def compute_displacement_errors(all_preds, all_gts, all_camposes, target='3d', eval_space='3d', use_global=True):
    """Compute the Displacement Errors (ADE and FDE)"""
    all_ades, all_fdes = dict(), dict()
    for r in list(all_gts.keys()):
        preds, gts, cam2world = all_preds[r], all_gts[r], all_camposes[r]
        
        # transform trajectory
        preds = traj_transform(preds, cam2world, target, eval_space, use_global)
        gts = traj_transform(gts, cam2world, target, eval_space, use_global)

        displace_errors = np.sqrt(np.sum((preds - gts)**2, axis=-1))  # (Tu,)
        # ADE score
        all_ades[r] = np.mean(displace_errors)
        # FDE score
        all_fdes[r] = displace_errors[-1]
    
    return all_ades, all_fdes


def compute_block_distances(all_preds, all_gts, all_camposes, target='3d', eval_space='3d', use_global=True):
    """Compute the block distances along x, y, and z dimensions"""
    all_dxs, all_dys, all_dzs = dict(), dict(), dict()
    for r in list(all_gts.keys()):
        preds, gts, cam2world = all_preds[r], all_gts[r], all_camposes[r]
        
        # transform trajectory
        preds = traj_transform(preds, cam2world, target, eval_space, use_global)
        gts = traj_transform(gts, cam2world, target, eval_space, use_global)
        
        # delta X
        all_dxs[r] = np.mean(np.fabs(preds[:, 0] - gts[:, 0]))
        # delta Y
        all_dys[r] = np.mean(np.fabs(preds[:, 1] - gts[:, 1]))
        if preds.shape[-1] == 3:
            # delta Z
            all_dzs[r] = np.mean(np.fabs(preds[:, 2] - gts[:, 2]))
    return all_dxs, all_dys, all_dzs