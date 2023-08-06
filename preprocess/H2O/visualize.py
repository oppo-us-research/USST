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

import pickle
import cv2
import numpy as np
import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 

TRAJ_COLOR = (255, 0, 0)    # red
PAST_COLOR = (255, 255, 0)  # yellow
FUTURE_COLOR = (0, 0, 255)  # blue



def read_traj(traj_file):
    with open(traj_file, 'rb') as f:
        traj_data = pickle.load(f)
        trajdata_left = traj_data['left_hand']
        trajdata_right = traj_data['right_hand']
        intrinsics = traj_data['intrinsics']
    return trajdata_left, trajdata_right, intrinsics


def read_video(video_file, toRGB=True):
    assert os.path.exists(video_file), "File does not exist! {}".format(video_file)
    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    video = []
    while success:
        if toRGB:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video.append(frame)
        success, frame = cap.read()
    video = np.array(video)
    return video


def XYZ_to_uv(traj3d, intrinsics):
    """ traj3d: (T, 3), a list of 3D (X, Y,Z) points in local coordinate system
    """
    # transform the (X, Y, Z) into the (u,v) by PINHOLE camera model
    traj2d = np.zeros((traj3d.shape[0], 2), dtype=np.float32)
    traj2d[:, 0] = (traj3d[:, 0] * intrinsics['fx'] / traj3d[:, 2] + intrinsics['cx'])
    traj2d[:, 1] = (traj3d[:, 1] * intrinsics['fy'] / traj3d[:, 2] + intrinsics['cy'])
    # clip the coordinates 
    traj2d[:, 0] = np.clip(traj2d[:, 0], 0, intrinsics['width'])
    traj2d[:, 1] = np.clip(traj2d[:, 1], 0, intrinsics['height'])
    traj2d = np.floor(traj2d).astype(np.int32)
    return traj2d


def resize_coords(traj2d, width, height, intrinsics):
    traj2d_new = np.copy(traj2d)
    traj2d_new[:, 0] = np.floor(traj2d[:, 0] / intrinsics['width'] * width)
    traj2d_new[:, 1] = np.floor(traj2d[:, 1] / intrinsics['height'] * height)
    return traj2d_new


def draw_traj2d_frame(frame, traj2d, traj_start=0, future_start=15, radius=5):
    cv2.polylines(frame, [traj2d], False, TRAJ_COLOR, thickness=2)
    for t, uv in enumerate(traj2d):
        color = PAST_COLOR if traj_start + t <= future_start else FUTURE_COLOR
        cv2.circle(frame, tuple(uv), radius=radius, color=color, thickness=-1)


def draw_traj3d_frame(frame, traj3d, temp_file='temp.mp4', compose='concat'):
    # draw 3D plot
    height, width = frame.shape[:2]
    scale = 0.8 if compose == 'overlap' else 1.0
    canvas_size = min(height, width) * scale
    traj3d_frame = draw_trajectory_3d(traj3d, (canvas_size, canvas_size), dpi=90, views=[30, -120], savefile=temp_file)
    if compose == 'overlap':
        # overlap the traj3d frame on the the original frame
        bottom_left = frame[height - traj3d_frame.shape[0]:, :traj3d_frame.shape[1], :]
        frame[height - traj3d_frame.shape[0]:, :traj3d_frame.shape[1], :] = cv2.addWeighted(bottom_left, 0.4, traj3d_frame, 0.6, 0)  # bottom left
    else:
        # concatenate horizontally
        traj3d_frame = cv2.resize(traj3d_frame, (int(height / traj3d_frame.shape[0] * traj3d_frame.shape[1]), height))
        frame = cv2.hconcat([frame, traj3d_frame])
    return frame


def draw_trajectory_3d(traj3d, video_size, dpi=90, views=[30, -120], minmax=None, savefile='temp.mp4'):

    if minmax is not None:
        xmin, ymin, zmin = minmax[0]
        xmax, ymax, zmax = minmax[1]
    else:
        xmin, xmax = np.min(traj3d[:, 0]) - 0.1, np.max(traj3d[:, 0]) + 0.1
        ymin, ymax = np.min(traj3d[:, 1]) - 0.1, np.max(traj3d[:, 1]) + 0.1
        zmin, zmax = np.min(traj3d[:, 2]) - 0.1, np.max(traj3d[:, 2]) + 0.1

    fig_w, fig_h = int(video_size[1] // dpi), int(video_size[0] // dpi)

    # init a 3D figure
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(views[0], views[1])  # elev: 30, azim: -120
    # Make a 3D quiver plot
    ox, oy, oz = np.zeros((3,3))
    dx, dy, dz = np.array([[xmax-xmin, 0, 0], [0, zmax-zmin, 0], [0, 0, ymax-ymin]]) * 0.2
    ax.quiver(ox, oy, oz, dx, dy, dz)
    # Setup coordinate system
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, zmax)
    ax.set_zlim(ymin, ymax)
    ax.invert_zaxis()
    ax.set_xlabel('X')  # x: rightward
    ax.set_ylabel('Z')  # z: innerward
    ax.set_zlabel('Y')  # y: downward
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='z', nbins=5)
    plt.tight_layout()

    traj3d_writer = FFMpegWriter(fps=30, metadata=dict(title='3D Trajectory', artist='Matplotlib'))
    with traj3d_writer.saving(fig, savefile, dpi=dpi):
        xs, ys, zs = np.array(traj3d)[:, 0], np.array(traj3d)[:, 1], np.array(traj3d)[:, 2]
        ax.plot(xs, zs, ys, '-o', color='red', markersize=2)
        # grab a single frame
        traj3d_writer.grab_frame()
    plt.close()

    frame_traj3d = read_video(savefile, toRGB=True)
    # remoe temporary file
    if os.path.exists(savefile):
        os.remove(savefile)
    
    return frame_traj3d[0]



def video_to_gif(video, giffile, fps=5.0, toBGR=False):
    assert giffile.endswith('.gif')
    with imageio.get_writer(giffile, mode='I', duration=1.0/fps) as writer:
        for frame in video:
            frame_vis = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if toBGR else np.copy(frame)
            writer.append_data(frame_vis)


def visualize(trajdata, video, intrinsics, n_future=30, obs_ratio=1.0, radius=5, with_3d=False, fps=30):
    """ Visualize trajectory
    """
    height, width = video.shape[1:3]
    temp_file = os.path.join(result_dir, 'temp.mp4')
    result = []
    # visualize 2D trajectory
    for traj_data in trajdata:
        start, end = traj_data['start'], traj_data['end']
        clip = np.copy(video[start: end+1])
        traj3d = traj_data['traj3d']  # suppose this is in world coordinate system
        cam2world = traj_data['cam2world']
        future_start = int(len(traj3d) * obs_ratio)
        horizon = min(n_future, len(traj3d))

        video_vis = []
        for i, (frame, Rc2w) in enumerate(zip(clip, cam2world)):
            # get the global trajectory
            horizon_end = min(i+horizon, len(traj3d))
            traj3d_global = traj3d[i:horizon_end]

            # world to cam
            extrinsics = np.linalg.inv(Rc2w)
            global_homo = np.concatenate([traj3d_global, np.ones((horizon_end-i, 1))], axis=1)  # (30, 4)
            traj3d_local = extrinsics.dot(global_homo.T)[:3, :].T  # (30, 3)

            # project 3D coordinates onto pixel 2D coordinates
            traj2d_vis = XYZ_to_uv(traj3d_local, intrinsics)
            
            # scaling due to video resize
            traj2d_vis = resize_coords(traj2d_vis, width, height, intrinsics)

            # visualize 2D trajectory on the RGB frame
            draw_traj2d_frame(frame, traj2d_vis, traj_start=i, future_start=future_start, radius=radius)

            if with_3d:
                # visualize 3D trajectory
                frame = draw_traj3d_frame(frame, traj3d_global, temp_file='temp.mp4')

            video_vis.append(frame)
        result.append(np.array(video_vis))
    
    return result
    

def main(traj_file, vid_file, result_dir, longest=False):

    # read trajectory data
    trajdata_left, trajdata_right, intrinsics = read_traj(traj_file)
    
    # read video
    video = read_video(vid_file, toRGB=True)
    fps = 30

    # visualize left & right hand trajectories
    vis_left = visualize(trajdata_left, video, intrinsics, n_future=30, radius=3, with_3d=True, fps=fps)
    vis_right = visualize(trajdata_right, video, intrinsics, n_future=30, radius=3, with_3d=True, fps=fps)

    if longest:
        lens = [len(vis) for vis in vis_left]
        vis_left = vis_left[np.argmax(lens)]  # the longest left hand
        lens = [len(vis) for vis in vis_right]
        vis_right = vis_right[np.argmax(lens)]  # the longest left hand
        vis = vis_left if len(vis_left) > len(vis_right) else vis_right
        # save to GIF file
        video_to_gif(vis, os.path.join(result_dir, '{}.gif'.format(sample)), fps=fps)
    else:
        # save to GIF file
        for n, vis in enumerate(vis_left):
            video_to_gif(vis, os.path.join(result_dir, '{}_left{}.gif'.format(sample, n+1)), fps=fps)
        for n, vis in enumerate(vis_right):
            video_to_gif(vis, os.path.join(result_dir, '{}_right{}.gif'.format(sample, n+1)), fps=fps)



if __name__ == '__main__':

    data_root = '../../data/H2O/Ego3DTraj'
    samples = ['sub1_h1_4', 'sub1_h2_4', 'sub1_k1_4', 'sub1_k2_4', 'sub1_o1_4', 'sub1_o2_4',
               'sub2_h1_4', 'sub2_h2_4', 'sub2_k1_4', 'sub2_k2_4', 'sub2_o1_4', 'sub2_o2_4',
               'sub3_h1_4', 'sub3_h2_4', 'sub3_k1_4', 'sub3_k2_4', 'sub3_o1_4', 'sub3_o2_4',
               'sub4_h1_4', 'sub4_h2_4', 'sub4_k1_4', 'sub4_k2_4', 'sub4_o1_4', 'sub4_o2_4']

    for sample in samples:
        
        print("Visualizing sample: {}".format(sample))
        traj_file = os.path.join(data_root, 'traj/{}.pkl'.format(sample))
        vid_file = os.path.join(data_root, 'video/{}.mp4'.format(sample))

        result_dir = os.path.join(data_root, 'vis')
        os.makedirs(result_dir, exist_ok=True)

        main(traj_file, vid_file, result_dir, longest=True)

