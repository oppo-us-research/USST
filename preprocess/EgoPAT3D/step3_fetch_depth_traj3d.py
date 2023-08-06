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
import pickle
import numpy as np
from tqdm import tqdm
import pyk4a
from pyk4a import PyK4APlayback
import argparse



def get_depth(playback, start, end, pointer):
    """get depth video clip"""
    depth_clip = np.zeros([end - start + 1] + resolution)  # (N, H, W)
    has_invalid = False
    # move pointer to the clip start frame
    while pointer < start:
        capture = playback.get_next_capture()
        pointer += 1
    # get the depth of the clip
    num_valid = 0
    while pointer < end + 1:
        capture = playback.get_next_capture()
        if (capture.color is None) or (capture.depth is None):
            pointer += 1  # move pointer
            has_invalid = True
            continue
        # assert (capture.color is not None) and (capture.depth is not None)
        depth_mat = pyk4a.depth_image_to_color_camera(capture.depth, playback.calibration, playback.thread_safe)
        depth_clip[pointer - start] = depth_mat
        pointer += 1  # move pointer
        num_valid += 1
    if num_valid < end - start + 1:
        has_invalid = True
    return depth_clip, pointer, has_invalid


def get_traj3d(traj2d, depth_clip):
    """ traj2d: (T, 2)
        depth_clip: (T, H, W)
    """
    traj3d = np.zeros((traj2d.shape[0], 3))
    for t, (coord, depth) in enumerate(zip(traj2d, depth_clip)):
        # scaling the 2D coordinates to raw 4K resolution
        u, v = int(coord[0] / vis_ratio), int(coord[1] / vis_ratio)
        # from 2D (u,v) to 3D (X, Y, Z) using the constant camera instrinsics
        Z = depth[v, u] / 1000  # mm to m
        X = (u - 1.94228662e+03) * Z / 1.80820276e+03
        Y = (v - 1.12382178e+03) * Z / 1.80794556e+03
        traj3d[t] = np.array([X, Y, Z])
    return traj3d
    


def fetch_trajectory_depth():
    
    for scene_id in sorted(args.scenes):
        for record_name in sorted(os.listdir(os.path.join(traj_dir, scene_id))):
            
            # get depth from raw MKV videos
            mkv_filepath = os.path.join(mkv_dir, scene_id, record_name + '.mkv')
            playback = PyK4APlayback(mkv_filepath)
            playback.open()
            pointer = 0
            
            # sort the filenames of clip trajectories in an ascending order
            all_trajfiles = os.listdir(os.path.join(traj_dir, scene_id, record_name))
            all_trajfiles = list(filter(lambda x: x.endswith('.pkl'), all_trajfiles))
            clip_ids = np.argsort([int(filename.split('_')[0][1:]) for filename in all_trajfiles])
            all_trajfiles = [all_trajfiles[i] for i in clip_ids]
            
            for traj_filename in tqdm(all_trajfiles, desc=f'Record {record_name}', total=len(all_trajfiles)):
                # read trajectory file
                traj_file = os.path.join(traj_dir, scene_id, record_name, traj_filename)
                with open(traj_file, 'rb') as f:
                    data = pickle.load(f)
                traj2d = data['traj2d']  # (T, 2)
                
                # get depth clip
                start_frame = int(traj_filename.split(".pkl")[0].split('_')[1][1:])
                end_frame = int(traj_filename.split(".pkl")[0].split('_')[2][1:])
                depth_clip, pointer, has_invalid = get_depth(playback, start_frame, end_frame, pointer)
                
                if has_invalid:
                    print("Invalid clip! Record: {}, Clip: {}".format(record_name, traj_filename))
                
                # get 3D trajectory (camera coordinate system)
                traj3d = get_traj3d(traj2d, depth_clip)
                data.update({'traj3d': traj3d})
                
                # write results to the pickle file
                with open(traj_file, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            playback.close()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../data/EgoPAT3D')
    parser.add_argument('--scenes', nargs='+', default=[])
    args = parser.parse_args()
    
    root_path = os.path.join(os.path.dirname(__file__), args.data_root)
    resolution = [2160, 3840]  # height, width
    vis_ratio = 0.25
    
    # result path
    traj_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'trajectory_warpped')
    assert os.path.exists(traj_dir), 'Path does not exist!'
    
    # mkv video path which contains depth
    mkv_dir = os.path.join(root_path, 'EgoPAT3D-mkv')
    
    if len(args.scenes) == 0:
        args.scenes = os.listdir(traj_dir)
    print("Start processing scenes: {}".format(args.scenes))
    
    fetch_trajectory_depth()
    
    print("Finished scenes: {}".format(args.scenes))