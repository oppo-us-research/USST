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
import pickle
import cv2
from tqdm import tqdm


def read_trajfile(trajfile):
    assert os.path.exists(trajfile), "File does not exist! {}".format(trajfile)
    with open(trajfile, 'rb') as f:
        traj_clips = pickle.load(f)
    return traj_clips


def get_all_clips(traj3d_all):
    all_clips = []
    for traj3d in traj3d_all:
        if len(traj3d) < 2: # at least two time points 
            continue
        start = int(list(traj3d.keys())[0])  # the start frame may not valid
        end = int(list(traj3d.keys())[-1])
        # find the first valid landmark
        for fid, landmarks in traj3d.items():
            pts3d = np.array(list(landmarks.values()))
            if len(pts3d[pts3d[:, 2] > 0]) > 0:
                start = fid  # move start forward
                break
        if start >= end:
            continue
        all_clips.append((start, end))
    return all_clips


def read_video(video_file, ratio=0.5):
    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    videos = []
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dst_size = (int(frame.shape[1] * ratio), int(frame.shape[0] * ratio))
        frame = cv2.resize(frame, dst_size)
        videos.append(frame)
        success, frame = cap.read()
    videos = np.array(videos)
    fps = cap.get(cv2.CAP_PROP_FPS) if len(videos) > 0 else 30
    return videos, fps


def write_video(mat, video_file, fps=30):
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mat.shape[2], mat.shape[1]))
    for frame in mat:
        video_writer.write(frame)


if __name__ == '__main__':
    
    ratio = 0.25  # we down-sampling the rgb video into 1/16 size for each clip (540 x 960)
    
    # trajectory directory
    root_dir = os.path.join(os.path.dirname(__file__), '../../data', 'EgoPAT3D')
    traj_dir = os.path.join(root_dir, 'EgoPAT3D-postproc', 'trajectory')
    
    # result directory
    videoclip_path = os.path.join(root_dir, 'EgoPAT3D-postproc', 'clip_videos')
    os.makedirs(videoclip_path, exist_ok=True)
    
    # get clip from trajectory annotation
    for scene_id in sorted(os.listdir(traj_dir)):
        scene_dir = os.path.join(traj_dir, scene_id)
        recordings = sorted(os.listdir(scene_dir))
        for filename in tqdm(recordings, total=len(recordings), desc='Process scene {}'.format(scene_id)):
            if not filename.endswith('pkl'):
                continue
            # read pkl trajectory file
            traj_file = os.path.join(scene_dir, filename)
            traj3d_all = read_trajfile(traj_file)
            
            # gather valid clips
            all_clips = get_all_clips(traj3d_all)
            
            record_name = filename.split('_')[0] + '_' + filename.split('_')[1]
            # create saving directory
            clip_savedir = os.path.join(videoclip_path, scene_id, record_name)
            os.makedirs(clip_savedir, exist_ok=True)
            
            # if some result clip files exist, check if there is a need to continue on the following codes
            need_continue = False
            for start, end in all_clips:
                # save each video clip
                clip_file = os.path.join(clip_savedir, 'clip_s{}_e{}.mp4'.format(start, end))
                if not os.path.exists(clip_file):
                    need_continue = True
                    break
                else:
                    clip_mat, _ = read_video(clip_file, ratio=1)
                    if len(clip_mat) == 0:  # Re-generate video clip
                        need_continue = True
                        break
            
            if need_continue:
                # read the whole mp4 video
                video_file = os.path.join(root_dir, 'EgoPAT3D-complete', scene_id, record_name, 'rgb_video.mp4')
                rgb_video, FPS = read_video(video_file, ratio=ratio)
                
                for start, end in all_clips:
                    # save each video clip
                    clip_file = os.path.join(clip_savedir, 'clip_s{}_e{}.mp4'.format(start, end))
                    if not os.path.exists(clip_file):
                        write_video(rgb_video[start: end+1], clip_file, fps=FPS)