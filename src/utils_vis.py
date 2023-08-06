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

import os
import numpy as np
import torch
import cv2
import imageio
from src.utils import output_transform, read_video, draw_trajectory_3d_with_gt



def draw_uv_points(video, preds, start=0, color=(0, 255, 0)):
    """ video: (T, H, W, C)
        preds: (Tm, 2/3)
    """
    height, width = video.shape[1:3]
    for t, loc in enumerate(preds):
        u, v = int(loc[0] * width), int(loc[1] * height)  # normalized loc to real loc
        cv2.circle(video[start + t], (u, v), radius=5, color=color, thickness=-1)
    return video


def video_to_gif(video, giffile, toBGR=False):
    assert giffile.endswith('.gif')
    with imageio.get_writer(giffile, mode='I', duration=0.2) as writer:
        for frame in video:
            if toBGR:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.append_data(frame)


def remove_the_old(result_path, topk=5, ignores=[]):
    files = [name for name in os.listdir(result_path) if name.endswith('.gif')]
    if len(files) >= topk + len(ignores):
        all_epochs = [int(filename.split('_')[0].split('epoch')[-1]) for filename in files]  # epoch20_bathroomCabinet_7_c40r0.6.gif
        fids = np.argsort(all_epochs)
        # iteratively remove
        for i in fids[:(len(files) - topk + 1)]:
            if all_epochs[i] in ignores:
                continue
            file_to_remove = os.path.join(result_path, files[i])
            if os.path.isfile(file_to_remove):
                os.remove(file_to_remove)


def vis_demo(video, outputs, nframes, trajectories, result_prefix=None, obs_ratios=['0.2', '0.5', '0.8'], toBGR=True, return_video=False):
    outputs_unobs, outputs_obs = outputs
    vis_video = {}
    for r in obs_ratios:
        num_full = int(nframes[0])  # only one video
        traj_gt = trajectories[0, :num_full]  # (Tm,3)
        video_vis = np.copy(video[:num_full])
        
        # visualize ground truth
        video_vis = draw_uv_points(video_vis, traj_gt, start=0, color=(0, 0, 255))  # blue
        
        # visualize predictions of the observed part
        if outputs_obs is not None:
            preds = outputs_obs[0][str(r)]['traj']
            video_vis = draw_uv_points(video_vis, preds, start=0, color=(255, 255, 0))  # yellow
        
        # visualize predictions of the future part
        preds = outputs_unobs[0][str(r)]['traj']
        num_obs = int(num_full * float(r))
        video_vis = draw_uv_points(video_vis, preds, start=num_obs-1, color=(0, 0, 0))  # black
        
        if not return_video:
            # save as GIF file
            video_to_gif(video_vis, result_prefix + 'r{}.gif'.format(r), toBGR=toBGR)
        else:
            vis_video[r] = video_vis
    return vis_video


def vis_traj3d(outputs, traj3d_gt, obs_ratios, nframes, canvas_size, result_prefix=None, show_past_pred=False, return_video=False):
    """ Visualize results in 3D space
    """
    outputs_unobs, outputs_obs = outputs
    vis_video = {}
    result_dir = os.path.dirname(result_prefix) if result_prefix is not None else './'
    for r in obs_ratios:
        preds_future = outputs_unobs[0][str(r)]['traj']
        # get the observed part
        num_full = int(nframes[0])  # only one video
        num_obs = int(num_full * float(r))
        # preds_past = outputs_obs[0][r]['traj'] if outputs_obs is not None else traj3d_gt[0, :num_obs]
        preds_past = outputs_obs[0][str(r)]['traj'] if outputs_obs is not None and show_past_pred else traj3d_gt[0, :num_obs]
        # complete prediction
        traj3d_pred = np.concatenate([preds_past, preds_future], axis=0)

        # draw 3D
        temp_file = os.path.join(result_dir, 'temp_pred.mp4')
        draw_trajectory_3d_with_gt(traj3d_pred, traj3d_gt[0, :num_full], canvas_size, ratio=float(r),
            fps=5, dpi=90, views=[30, -120], minmax=None, savefile=temp_file)
        video_traj3d = read_video(temp_file)
        
        if not return_video:
            video_to_gif(video_traj3d, result_prefix + 'r{}.gif'.format(r))
        else:
            vis_video[r] = video_traj3d

        if os.path.exists(temp_file):
            os.remove(temp_file)
    return vis_video


def visualize_example(train_loader, result_dir, epoch, num=1, target='3d'):
    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}
    means = np.array([[[0.5, 0.5, 0.5]]])  # (1,1,3)
    stds = np.array([[[0.5, 0.5, 0.5]]])  # (1,1,3)
    
    for i, (clip, nf, traj) in enumerate(train_loader):
        if i >= num:
            break  # only visualize 1 batches (clips)
        u = torch.floor(traj[0, :, 0] * clip.size(-1)).to(torch.long)  # (T,)
        v = torch.floor(traj[0, :, 1] * clip.size(-2)).to(torch.long)  # (T,)
        # visualize
        vis_mat = []
        for t in range(nf[0]):
            frame = clip[0, :, t].permute(1, 2, 0).contiguous().numpy()  # (224, 224, 3)
            frame = ((frame * stds + means) * 255).astype(np.uint8)  # broadcast add & mul 
            cv2.circle(frame, (int(u[t]), int(v[t])), radius=5, color=(0, 255, 0), thickness=-1)
            vis_mat.append(frame)
        vis_mat = np.array(vis_mat)  # (T, 224, 224, 3)
        # save to GIF figure
        video_to_gif(vis_mat, os.path.join(result_dir, 'vis_epoch{}_clip{}.gif'.format(epoch, i)))


def run_runtime_demo(cfg, model, data_batch, obs_ratios=['0.2', '0.5', '0.8'], result_folder='demo_train', prefix_name='epoch0_', topk=5):
    # input data (a training sample)
    filename, input_data, odometry, nframes, gt_traj = data_batch[0][0], data_batch[1][0:1], data_batch[2][0:1], data_batch[3][0:1], data_batch[4][0:1]
    scene_record = filename[:-4].split('/')[-2]
    clip_name = filename[:-4].split('/')[-1].split('_')[0]
    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}  # from EgoPAT3D preprocessing code
    vis_ratio = 0.25
    max_depth = 3
    
    # directly read the original video
    video = read_video(filename)
    
    # result path
    result_path = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, result_folder)
    os.makedirs(result_path, exist_ok=True)
    # remove the old figs
    remove_the_old(result_path, topk=topk, ignores=getattr(cfg.TRAIN.scheduler, 'lr_decay_epoch', []))  # only save demos from the latest topK models
    
    # inference
    with torch.no_grad():
        ratios = torch.tensor([[float(r) for r in obs_ratios]]).repeat(nframes.size(0), 1)  # (B, M)
        outputs = model.inference(input_data.cuda(), nframes, ratios, traj=gt_traj.cuda())
    
    # transform from global to local coordinate frame
    outputs, gt_traj = output_transform(outputs, gt_traj.numpy(), nframes.numpy(), odometry.numpy(), 
                                        intrinsics=intrinsics, vis_ratio=vis_ratio, max_depth=max_depth, 
                                        centralize=model.centralize, use_odom=model.use_odom, ignore_depth=model.ignore_depth)
    
    # visualize
    result_prefix = os.path.join(result_path, prefix_name + '{}_{}'.format(scene_record, clip_name))
    vis_demo(video, outputs, nframes, gt_traj, result_prefix, obs_ratios=obs_ratios)

    if model.target == '3d':
        # visualize 3D trajectory
        result_3d_path = os.path.join(result_path, 'traj3d')
        os.makedirs(result_3d_path, exist_ok=True)
        remove_the_old(result_3d_path, topk=topk, ignores=getattr(cfg.TRAIN.scheduler, 'lr_decay_epoch', []))  # only save demos from the latest topK models

        canvas_size = (intrinsics['h'] * vis_ratio * 0.5, intrinsics['w'] * vis_ratio * 0.5)
        result_prefix = os.path.join(result_3d_path, prefix_name + '{}_{}'.format(scene_record, clip_name))
        vis_traj3d(outputs, gt_traj, obs_ratios, nframes, canvas_size, result_prefix)