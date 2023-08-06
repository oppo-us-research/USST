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

""" In this step, we use RAFT model to get the forward and backward optical flows.
    The flows are used to warp the initial & the last hand location to get the forward & backward trajectories.
    Finally, the hand trajectory is the mean of the forward & backward trajectories.
"""
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch
import argparse
import sys
import pickle

RAFT_ROOT = os.path.join(os.path.dirname(__file__), '../../third_party/RAFT')
sys.path.append(os.path.join(RAFT_ROOT, 'core'))
from raft import RAFT
from utils.utils import InputPadder



def init_raft(args, raft_root='./RAFT'):
    # class cfg: pass
    args.model = os.path.join(raft_root, 'models', 'raft-things.pth')
    args.small = False  # do not use small model
    args.mixed_precision = False  # do not use mixed precision inference
    args.alternate_corr = False  # do not use efficent correlation implementation
    args.iters = 32
    args.device = torch.device('cuda')
    # cfg = vars(cfg)  # class to dict
    # for k, v in cfg.items():
    #     parser.add_argument('--' + k, default=v)
    # args = parser.parse_args()  # dict to parser
    
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(args.device)
    model.eval()
    return model, args


def read_video(video_file):
    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    video = []
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
        success, frame = cap.read()
    video = np.array(video)
    return video


def write_video(mat, video_file, fps=30):
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mat.shape[2], mat.shape[1]))
    for frame in mat:
        video_writer.write(frame)
        
        
def coord_check(px, py, img_size=(540, 960)):
    px_new = max(min(px, img_size[1] - 1), 0)
    py_new = max(min(py, img_size[0] - 1), 0)
    return px_new, py_new


@torch.no_grad()
def flow_warpping(video, px, py, model, iters=120, device='cuda'):
    """ video: (T, H, W, C) """
    # toTensor() and toGPU()
    video_tensor = torch.from_numpy(video.astype(np.float32)).permute(0, 3, 1, 2).to(device)  # (T, C, H, W)
    padder = InputPadder(video_tensor.shape[1:])
    traj2d = [[px, py]]
    for image1, image2 in zip(video_tensor[:-1], video_tensor[1:]):
        # pad the frames
        image1, image2 = padder.pad(image1[None], image2[None])
        # flow inference
        _, flow_up = model(image1, image2, iters=iters, test_mode=True)  # (1, 2, 544, 960)
        flow_up = padder.unpad(flow_up)
        # warpping
        new_px = traj2d[-1][0] + float(flow_up[0, 0, int(traj2d[-1][1]), int(traj2d[-1][0])].cpu())  # u = u + du
        new_py = traj2d[-1][1] + float(flow_up[0, 1, int(traj2d[-1][1]), int(traj2d[-1][0])].cpu())  # v = v + dv
        # clipping
        new_px, new_py = coord_check(new_px, new_py, img_size=video_tensor.shape[2:])
        traj2d.append([new_px, new_py])
    traj2d = np.array(traj2d, dtype=np.float32)
    return traj2d


def run_warping():
    
    for scene_id in args.scenes:
        # output path
        result_scene = os.path.join(output_dir, scene_id)
        os.makedirs(result_scene, exist_ok=True)
        
        for record_name in sorted(os.listdir(os.path.join(clip_path, scene_id))):
            # input video folder
            video_path = os.path.join(clip_path, scene_id, record_name)
            
            # output path
            result_record = os.path.join(result_scene, record_name)
            os.makedirs(result_record, exist_ok=True)
            
            all_clips = os.listdir(video_path)
            for clip_name in tqdm(sorted(all_clips), desc=f'Record {record_name}', total=len(all_clips)):                
                # read video with hand location
                sx = int(clip_name.split('.mp4')[0].split('_')[-4][2:])
                sy = int(clip_name.split('.mp4')[0].split('_')[-3][2:])
                ex = int(clip_name.split('.mp4')[0].split('_')[-2][2:])
                ey = int(clip_name.split('.mp4')[0].split('_')[-1][2:])
                # read video
                video = read_video(os.path.join(video_path, clip_name))  # (N, H, W, C)
                # coordinate check
                sx, sy = coord_check(sx, sy, img_size=video.shape[1:3])
                ex, ey = coord_check(ex, ey, img_size=video.shape[1:3])
                
                with torch.no_grad():
                    # forward pass: warp the first hand location to the future frames by RAFT
                    traj2d_forward = flow_warpping(video, sx, sy, model, iters=args.iters, device=args.device)
                    # backward pass: warp the last hand location to the initial frames by RAFT
                    traj2d_backward = flow_warpping(video[::-1], ex, ey, model, iters=args.iters, device=args.device)
                    # take weighted mean
                    weights = args.conf + (1.0 - args.conf) / (1 + np.exp(np.arange(1, video.shape[0] + 1) - video.shape[0] / 2))
                    traj2d = traj2d_forward * weights[:, None] + traj2d_backward[::-1] * (1 - weights[:, None])
                
                # save the 2D trajectory
                result_traj_file = os.path.join(result_record, '{}.pkl'.format(clip_name.split('.mp4')[0]))
                with open(result_traj_file, 'wb') as f:
                    pickle.dump({'traj2d': traj2d}, f, protocol=pickle.HIGHEST_PROTOCOL)


def visualize(scene_id='1', record_name='bathroomCabinet_1', clip_name='c9_s439_e483_x438_y189'):
    
    # read trajectory file
    input_traj_file = os.path.join(output_dir, scene_id, record_name, clip_name + '.pkl')
    assert os.path.exists(input_traj_file), 'trajectory file does not exist!'
    with open(input_traj_file, 'rb') as f:
        data = pickle.load(f)
    traj2d = data['traj2d']  # (T, 2)
    
    # read video file
    video_file = os.path.join(clip_path, scene_id, record_name, clip_name + '.mp4')
    assert os.path.exists(video_file), 'video clip file does not exist!'
    video = read_video(video_file)  # (T, H, W, C)
    
    # visualize
    vis_video = []
    for hand_point, img in zip(traj2d, video):
        img_vis = cv2.circle(img, (int(hand_point[0]), int(hand_point[1])), radius=5, color=(0, 0, 255), thickness=-1)
        vis_video.append(img_vis)
    vis_video = np.array(vis_video)
    
    result_vis_file = os.path.join(output_dir, scene_id, record_name, 'vis_' + clip_name + '.mp4')
    write_video(vis_video, result_vis_file, fps=5)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../data/EgoPAT3D')
    parser.add_argument("--scenes", nargs='+', default=[], help='the scene ids.')
    parser.add_argument("--vis", action='store_true', help='visualize an example')
    parser.add_argument('--conf', type=float, default=0.3, help='the least confidence of forward warpping')
    args = parser.parse_args()
    
    root_path = os.path.join(os.path.dirname(__file__), args.data_root)
    
    # path to video clips with hand annotation of the first frame
    clip_path = os.path.join(root_path, 'EgoPAT3D-postproc', 'video_clips_hand')
    
    # result path
    output_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'trajectory_warpped')
    os.makedirs(output_dir, exist_ok=True)
    
    if args.vis:
        assert os.path.exists(output_dir)
        print('Visualize 2D trajectory...')
        visualize(scene_id='1', record_name='bathroomCabinet_1', clip_name='c100_s5241_e5270_sx563_sy454_ex722_ey355')
    else:
        # get input scenes
        all_scenes = os.listdir(clip_path)
        if len(args.scenes) > 0:
            for s in args.scenes:
                assert s in all_scenes
        else:
            args.scenes = all_scenes
        # initialize a RAFT flow model
        model, args = init_raft(args, raft_root=RAFT_ROOT)
    
        run_warping()
    