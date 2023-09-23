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
import random
from tqdm import tqdm
import numpy as np
import cv2
from demo import read_traj_file, read_video, read_odometry
from src.utils import video_to_gif
from illustrate_task import visualize_transformed_traj, local_to_global, draw_trajectory_3d


scene_splits = {'train': {'1': ['1', '2', '3', '4', '5', '6', '7'], 
                          '2': ['1', '2', '3', '4', '5', '6', '7'], 
                          '3': ['1', '2', '3', '4', '5', '6'],
                          '4': ['1', '2', '3', '4', '5', '6', '7'],
                          '5': ['1', '2', '3', '4', '5', '6'], 
                          '6': ['1', '2', '3', '4', '5', '6'], 
                          '7': ['1', '2', '3', '4', '5', '6', '7'],
                          '9': ['1', '2', '3', '4', '5', '6', '7'],
                          '10': ['1', '2', '3', '4', '5', '6', '7'],
                          '11': ['1', '2', '3', '4', '5', '6', '7'],
                          '12': ['1', '2', '3', '4', '5', '6', '7']},
                'val': {'1': ['8'], 
                        '2': ['8'], '3': ['7'], '4': ['8'], '5': ['7'], '6': ['7'], 
                        '7': ['8'], '9': ['8'], '10': ['8'], '11': ['8'], '12': ['8']},
                'test': {'1': ['9', '10'], 
                        '2': ['9', '10'], '3': ['9', '10'], '4': ['9', '10'], '5': ['8', '9'], '6': ['8', '9'], 
                        '7': ['9', '10'], '9': ['9', '10'], '10': ['9', '10'], '11': ['9', '10'], '12': ['9', '10']},
                'test_novel': {'13': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 
                            '14': ['2', '3', '4', '5', '6', '7', '8', '9', '10'], 
                            '15': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}}
scene_names = {'1': 'bathroomCabinet', '2': 'bathroomCounter', '3': 'bin', '4': 'desk', '5': 'drawer', '6': 'kitchenCounter', '7': 'kitchenCupboard',
               '9': 'microwave', '10': 'nightstand', '11': 'pantryShelf', '12': 'smallBins', '13': 'stoveTop', '14': 'windowsillAC', '15': 'woodenTable'}



if __name__ == '__main__':

    data_root = os.path.join(os.path.dirname(__file__), '../data', 'EgoPAT3D')
    video_dir = os.path.join(data_root, 'EgoPAT3D-postproc', 'video_clips_hand')
    traj_dir = os.path.join(data_root, 'EgoPAT3D-postproc', 'trajectory_repair')
    odometry_dir = os.path.join(data_root, 'EgoPAT3D-postproc', 'odometry')
    # intrinsics and extra info
    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}  # from EgoPAT3D preprocessing code
    vis_ratio = 0.25
    max_depth = 3
    for k, v in intrinsics.items():
        v *= vis_ratio  # scaling the camera intrinsics
        intrinsics[k] = v
    obs_ratio = 0.5
    
    num_record_vis = 3  # number of records to visualize for each scene (<= 3)
    num_clip_vis = 3  # number of samples to visualize for each record (<= 6)
    subset = 'test_novel'
    random.seed(42)
    
    # result path
    result_dir = os.path.join(data_root, 'EgoPAT3D-postproc', 'visualize', subset)
    os.makedirs(result_dir, exist_ok=True)
    temp_file = os.path.join(result_dir, 'temp.mp4')
    
    for scene_id, record_splits in tqdm(scene_splits[subset].items(), ncols=0, desc=subset):
        
        scene_rgb_path = os.path.join(video_dir, scene_id)
        scene_traj_path = os.path.join(traj_dir, scene_id)
        scene_odom_path = os.path.join(odometry_dir, scene_id)
        
        record_names = list(filter(lambda x: x.split('_')[-1] in record_splits, os.listdir(scene_traj_path)))  # get the splits of records
        record_names_selected = random.choices(record_names, k=num_record_vis)
        for record in record_names_selected:
            record_rgb_path = os.path.join(scene_rgb_path, record)
            record_traj_path = os.path.join(scene_traj_path, record)
            record_odom_path = os.path.join(scene_odom_path, record)
            
            traj_files = list(filter(lambda x: x.endswith('.pkl'), os.listdir(record_traj_path)))  # a list of '*.pkl'
            traj_files_selected = random.choices(traj_files, k=num_clip_vis)
            for traj_name in traj_files_selected:
                # input files
                video_file = os.path.join(record_rgb_path, traj_name[:-4] + '.mp4')
                traj_file = os.path.join(record_traj_path, traj_name)
                odom_file = os.path.join(record_odom_path, traj_name[:-4] + '.npy')
                # read trajectory
                traj2d, traj3d, num_preserve = read_traj_file(traj_file)
                # read video 
                video = read_video(video_file, toRGB=True)  # RGB video
                video = video[:num_preserve]
                # read odometry
                odometry = read_odometry(odom_file)
                odometry = odometry[:num_preserve]
                
                # visualize 2D trajectory on video
                visualize_transformed_traj(video, traj3d, odometry, intrinsics, ratio=obs_ratio)
                
                # local 3D to global 3D
                traj3d_global = local_to_global(traj3d, odometry)
                traj3d = traj3d_global[0]  # only refer to the camera of the first frame 
                
                # visualize 3D trajectory
                canvas_size = min(intrinsics['h'], intrinsics['w']) * 0.82
                draw_trajectory_3d(traj3d, (canvas_size, canvas_size), ratio=obs_ratio,
                    fps=5, dpi=90, views=[30, -120], minmax=[(-0.1, -0.2, 0.1), (0.3, 0.1, 0.7)], savefile=temp_file)
                video_traj3d = read_video(temp_file, toRGB=True)
                
                # overlap the two videos
                video_vis = np.copy(video)
                for i, (traj, rgb) in enumerate(zip(video_traj3d, video)):
                    bottom_left = rgb[rgb.shape[0] - traj.shape[0]:, :traj.shape[1], :]
                    rgb[rgb.shape[0] - traj.shape[0]:, :traj.shape[1], :] = cv2.addWeighted(bottom_left, 0.4, traj, 0.6, 0)  # bottom left
                    video_vis[i] = rgb
                
                # save to GIF file
                gif_name = '{}_{}.gif'.format(record, traj_name.split('_')[0])
                video_to_gif(video_vis, os.path.join(result_dir, gif_name), toBGR=True)
                
                if os.path.exists(temp_file):
                    os.remove(temp_file)