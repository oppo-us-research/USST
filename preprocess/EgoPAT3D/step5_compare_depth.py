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

""" 
    In this file, we compare the visualized 3D trajectories amon the true depth (from RGB-D camera recordings),
    the repaired depth by least square fitting, and the estimated depth (from NewCRFs model)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
from matplotlib.animation import FFMpegWriter
from step4_repair_invalid_depth import read_traj_file, remove_tail_outliers, repair_depth
from depth_estimation import init_depth_model, monocular_depth_estimation



def read_video(video_file):
    assert os.path.exists(video_file), "File does not exist! {}".format(video_file)
    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    video = []
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
        success, frame = cap.read()
    video = np.array(video)
    return video


def repair_depth_by_model(traj3d, traj2d, invalids, video, model, cfg, vis_ratio=0.25):
    """ Replace with the estimated depth for invalid depth frames """
    # compute depth scale
    all_scales = []
    for ids in range(len(traj3d)):
        if ids in invalids:
            continue
        # depth estimation
        depth_map = monocular_depth_estimation([video[ids]], model, cfg)
        # repaire 3D coordinates
        u, v = int(traj2d[ids, 0]), int(traj2d[ids, 1])  # (u,v) in (540, 960) frame
        Z = depth_map[0][v, u]
        # compute scale
        s = traj3d[ids, 2] / Z
        all_scales.append(s)
    scale = np.mean(np.array(all_scales))
    
    # repair depth with the scale and depth estimation model
    traj3d_repaire = np.copy(traj3d)
    for ids in invalids:
        # depth estimation
        depth_map = monocular_depth_estimation([video[ids]], model, cfg)
        # repaire 3D coordinates
        u, v = int(traj2d[ids, 0]), int(traj2d[ids, 1])  # (u,v) in (540, 960) frame
        Z = depth_map[0][v, u] * scale
        X = (u / vis_ratio - 1.94228662e+03) * Z / 1.80820276e+03
        Y = (v / vis_ratio - 1.12382178e+03) * Z / 1.80794556e+03
        traj3d_repaire[ids] = np.array([X, Y, Z])
    
    return traj3d_repaire


def find_vis_samples(model=None, cfg=None):
    # for each scene, find a complete trajectory as an example for visualization
    examples = dict()
    for scene_id in sorted(os.listdir(traj_dir)):
        found = False
        
        for record_name in sorted(os.listdir(os.path.join(traj_dir, scene_id))):
            # loop all trajectory files
            all_trajfiles = os.listdir(os.path.join(traj_dir, scene_id, record_name))
            all_trajfiles = list(filter(lambda x: x.endswith('.pkl'), all_trajfiles))
            clip_ids = np.argsort([int(filename.split('_')[0][1:]) for filename in all_trajfiles])
            all_trajfiles = [all_trajfiles[i] for i in clip_ids]
            
            for traj_filename in all_trajfiles:
                # read trajectory file
                traj_file = os.path.join(traj_dir, scene_id, record_name, traj_filename)
                traj2d, traj3d = read_traj_file(traj_file)  # (N,2), (N,3)
                
                # trim the trajectory rail part
                num_preserve = remove_tail_outliers(traj3d[:, 2])
                traj2d, traj3d = traj2d[:num_preserve], traj3d[:num_preserve]
                
                invalids = np.where(traj3d[:, 2] <= MIN_DEPTH)[0]
                if len(invalids) > 3 and len(invalids) < len(traj3d) - MIN_VALID_POINTS:
                    # repair by least square fitting
                    traj3d_lsq = repair_depth(traj3d, traj2d, invalids, vis_ratio)
                    
                    # read video
                    video_file = os.path.join(rgb_dir, scene_id, record_name, traj_filename[:-4] + '.mp4')
                    video = read_video(video_file)  # rgb(T, H, W, C)
                    video = video[:num_preserve]
                    
                    # repair by depth estimation model
                    traj3d_est = repair_depth_by_model(traj3d, traj2d, invalids, video, model, cfg)
                    
                    examples[scene_id] = {'scene_id': scene_id, 'record_name': record_name, 'clip_name': traj_filename[:-4],
                                          'traj2d': traj2d, 'traj3d_true': traj3d, 'traj3d_lsq': traj3d_lsq, 'traj3d_est': traj3d_est, 'num_preserve': num_preserve}
                    found = True
                    print('Found one video clip (len={}): {}/{}/{}'.format(len(traj3d), scene_id, record_name, traj_filename.split('_')[0]))
                
                if found:
                    break  # break the clip loop
            if found:
                break  # break the record loop
    
    return examples


def draw_uv_points(video, traj2d):
    """ draw uv location on rgb video """
    video_vis = np.copy(video)
    for frame, coord in zip(video_vis, traj2d):
        u, v = int(coord[0]), int(coord[1])
        cv2.circle(frame, (u, v), radius=5, color=(0, 255, 0), thickness=-1)
    return video_vis


def draw_depth_comparison(traj3d_true, traj3d_lsq, traj3d_est, canvas_height=540, fig_size=5, fps=5, savefile='temp.mp4'):
    
    # initialize a canvas for visualization
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot()
    ax.set_xlabel('frame id')
    ax.set_ylabel('hand depth (m)')
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    
    Z_true, Z_lsq, Z_est = traj3d_true[:, 2], traj3d_lsq[:, 2], traj3d_est[:, 2]
    max_depth = max(np.max(Z_true), np.max(Z_est))
    plt.ylim(0, max_depth + 0.01)
    plt.xlim(0, traj3d_true.shape[0] + 1)
    plt.grid(axis='y')
    
    # draw with plt canvas
    traj_writer = FFMpegWriter(fps=fps, metadata=dict(title='depth comparison', artist='Matplotlib'))
    with traj_writer.saving(fig, savefile, dpi=int(canvas_height / fig_size)):
        for i in range(len(Z_true)):
            ax.scatter(np.arange(1, i+2), Z_est[:(i+1)], color='b', label='Repair by NewCRFs')
            ax.scatter(np.arange(1, i+2), Z_lsq[:(i+1)], color='k', label='Repair by LSF')
            ax.scatter(np.arange(1, i+2), Z_true[:(i+1)], color='r', label='RGB-D Camera')
            if i == 0:
                plt.legend()
            traj_writer.grab_frame()
    plt.close()
    
    # read the cached video
    traj_video = read_video(savefile)
    
    return traj_video


def write_video(mat, video_file, fps=5):
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mat.shape[2], mat.shape[1]))
    for frame in mat:
        video_writer.write(frame)


def vis_compare(examples):
    # vis
    for scene_id, data in examples.items():
        # read rgb video
        video_file = os.path.join(rgb_dir, data['scene_id'], data['record_name'], data['clip_name'] + '.mp4')
        rgb = read_video(video_file)  # rgb(T, H, W, C)
        rgb = rgb[:data['num_preserve']]
        
        # draw 2D hand locations on RGB
        vis_mat = draw_uv_points(rgb, data['traj2d'])

        # draw 3D trajectory on XoZ plane
        compare_vis = draw_depth_comparison(data['traj3d_true'], data['traj3d_lsq'], data['traj3d_est'], canvas_height=rgb.shape[1], fps=VIS_FPS)
        vis_mat = np.concatenate([vis_mat, compare_vis], axis=2) 
        
        # save visualized video
        filename = 'compare_{}_{}_{}'.format(data['scene_id'], data['record_name'], data['clip_name'].split("_")[0])
        vis_file = os.path.join(vis_path, filename + '.mp4')
        write_video(vis_mat, vis_file, fps=VIS_FPS)


def plot_compare(examples, fig_size=5, fontsize=20):
    
    for scene_id, data in examples.items():
        
        traj3d_true, traj3d_lsq, traj3d_est = data['traj3d_true'], data['traj3d_lsq'], data['traj3d_est']
        
        # draw on figure
        fig = plt.figure(figsize=(fig_size, fig_size))
        ax = fig.add_subplot()
        ax.set_xlabel('frame id', fontsize=fontsize)
        ax.set_ylabel('hand depth (m)', fontsize=fontsize)
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        
        Z_true, Z_lsq, Z_est = traj3d_true[:, 2], traj3d_lsq[:, 2], traj3d_est[:, 2]
        max_depth = max(np.max(Z_true), np.max(Z_est))
        plt.ylim(0, max_depth + 0.01)
        plt.xlim(0, traj3d_true.shape[0] + 1)
        plt.grid(axis='y')
        
        ax.scatter(np.arange(1, len(Z_true) + 1), Z_est[:len(Z_true)], color='b', label='Repair by NewCRFs')
        ax.scatter(np.arange(1, len(Z_true) + 1), Z_lsq[:len(Z_true)], color='k', label='Repair by LSF')
        ax.scatter(np.arange(1, len(Z_true) + 1), Z_true[:len(Z_true)], color='r', label='RGB-D Camera')
        plt.legend(fontsize=fontsize)
        plt.tight_layout()
        
        # save
        fig_dir = os.path.join(vis_path, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        filename = 'compare_{}_{}_{}'.format(data['scene_id'], data['record_name'], data['clip_name'].split("_")[0])
        plt.savefig(os.path.join(fig_dir, filename + '.png'))
        plt.savefig(os.path.join(fig_dir, filename + '.pdf'))


if __name__ == '__main__':
    
    # root path
    root_path = os.path.join(os.path.dirname(__file__), '../../data/EgoPAT3D')
    MIN_DEPTH = 0.001
    MIN_VALID_POINTS = 10
    NUM_TAIL_FRAMES = 6
    VIS_FPS = 5
    vis_ratio = 0.25
    
    # trajectory path
    traj_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'trajectory_warpped')
    assert os.path.exists(traj_dir), 'Path does not exist!'
    
    # video path
    rgb_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'video_clips_hand')
    assert os.path.exists(rgb_dir), 'Path does not exist!'
    
    # result path
    vis_path = os.path.join(os.path.dirname(__file__), '../../output/EgoPAT3D/compare_gt_depth')
    os.makedirs(vis_path, exist_ok=True)
    vis_data_file = os.path.join(vis_path, 'vis_data.pkl')
    
    if not os.path.exists(vis_data_file):
        # initialize depth estimation model
        model, cfg = init_depth_model()
        
        # find examples for visualization
        examples = find_vis_samples(model, cfg)
        
        # save
        with open(vis_data_file, 'wb') as f:
            pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(vis_data_file, 'rb') as f:
            examples = pickle.load(f)
    
    print("Visualizing the comparison...")
    # vis_compare(examples)
    
    plot_compare(examples)