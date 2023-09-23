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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import pyk4a
from pyk4a import PyK4APlayback
from tqdm import tqdm
import pickle
import argparse

from google.protobuf import text_format
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils
import mediapipe.python.solutions.drawing_styles as mp_style
import mediapipe.python.solutions.hands as mp_hands
from typing import Mapping, Optional, Tuple

_VISIBILITY_THRESHOLD = 0.5
_PRESENCE_THRESHOLD = 0.5
WHITE_COLOR = (224, 224, 224)



def read_landmarks(landmark_file, flip=False):
    landmarks = dict()
    with open(landmark_file, 'r') as f:
        pid = 0
        for line in f.readlines():
            line = line.strip()
            if line.split(':')[0].isnumeric():
                frame_id = int(line.split(':')[0])
                landmarks[frame_id] = np.zeros((21, 3))  # a new frame
                assert pid == 0 or pid == 20, 'invalid frame!'
                pid = -1
                continue
            if 'landmark' in line: 
                pid += 1  # a new 3D point
                continue
            if 'x:' in line: 
                xval = float(line.split(': ')[1])
                landmarks[frame_id][pid, 0] = 1.0 - xval if flip else xval
                continue
            if 'y:' in line:
                landmarks[frame_id][pid, 1] = float(line.split(': ')[1])
                continue
            if 'z:' in line:
                landmarks[frame_id][pid, 2] = float(line.split(': ')[1])
                continue
    return landmarks


def read_landmarks_text(landmark_file, flip=False):
    landmarks = dict()
    with open(landmark_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line.split(':')[0].isnumeric():
                frame_id = int(line.split(':')[0])
                landmarks[frame_id] = ''
                pointer = 0
                continue
            if flip and 'x:' in line:
                xval = 1.0 - float(line.split(': ')[1])  # flip x coordinates horizontally
                line = '  x: {}'.format(xval)
            if pointer <= 105:  # 1 + 5 * 21 = 105 rows
                if ('x:' in line) or ('y:' in line) or ('}' in line):
                    line += ' '
                landmarks[frame_id] += line
                pointer += 1
    return landmarks


def read_clip_ranges(rng_file):
    ranges = dict()
    with open(rng_file, 'r') as f:
        for line in f.readlines():
            cid = int(line.strip().split('. ')[0])
            start = int(line.strip().split('. ')[1].split(', ')[0])
            end = int(line.strip().split('. ')[1].split(', ')[1])
            ranges[cid] = (start, end)
    return ranges


def get_2d_coordinates(landmarks_text, start, end, resolution):
    image_cols, image_rows = resolution[1], resolution[0]
    results = {}
    for i in range(start, end):
        if i not in landmarks_text:
            continue
        # parse landmarks
        landmark_list = text_format.Parse(landmarks_text[i], landmark_pb2.NormalizedLandmarkList())
        # get the 2D coordinates
        idx_to_coordinates = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                landmark.visibility < _VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            landmark_px = drawing_utils._normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
            if landmark_px:
                idx_to_coordinates[idx] = landmark_px  # (x, y)
        # save to dict
        results[i] = idx_to_coordinates
    return results


def get_3d_coordinates(landmarks, landmarks_2d, frame_start, depth_clip=None):
    results = {}
    for frame_id, idx_to_coords in landmarks_2d.items():
        pxyz = np.copy(landmarks[frame_id])  # (21, 3)
        pxyz[1:, 2] += pxyz[0, 2]  # z values of top 20 are relative offsets to the 21-th z value
        idx_to_coords3d = {}
        # iterate each 2D point (u,v) to fetch the depth (z)
        for idx, puv in idx_to_coords.items():
            x_norm, y_norm, Z = pxyz[idx]
            # retrieve z from (u,v) location of depth map
            if depth_clip is not None:
                u, v = int(puv[1]), int(puv[0])
                Z = depth_clip[frame_id - frame_start, u, v]
            # from 2D (u,v) to 3D (X, Y, Z)
            Z = Z / 1000  # mm to m
            X = (x_norm * resolution[1] - 1.94228662e+03) * Z / 1.80820276e+03
            Y = (y_norm * resolution[0] - 1.12382178e+03) * Z / 1.80794556e+03
            idx_to_coords3d[idx] = [X, Y, Z]  # (m, m, m)
        results[frame_id] = idx_to_coords3d
    return results


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


def get_depth_vis(depth_mat, depth_range=[0, 1200], ratio=1.0):
    if np.max(depth_mat) > depth_range[1]:
        depth_range[1] = np.max(depth_mat)
    depth_vis = colorize(depth_mat, tuple(depth_range))
    # resize into smaller size
    dst_size = (int(depth_vis.shape[1] * ratio), int(depth_vis.shape[0] * ratio))
    depth_vis = cv2.resize(depth_vis, dst_size)
    return depth_vis


def get_depth(playback, start, end, pointer, ratio=1.0, vis_depth=False):
    
    depth_clip = np.zeros([end - start] + resolution)  # (N, H, W, 3)
    depth_vis_all = []
    # move pointer to the clip start frame
    while pointer < start:
        capture = playback.get_next_capture()
        pointer += 1
    # get the depth of the clip
    while pointer < end:
        capture = playback.get_next_capture()
        if (capture.color is None) or (capture.depth is None):
            pointer += 1  # move pointer
            continue
        # rgb = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
        depth_mat = pyk4a.depth_image_to_color_camera(capture.depth, playback.calibration, playback.thread_safe)
        depth_clip[pointer - start] = depth_mat
        if vis_depth:
            # save visualized depth
            depth_vis = get_depth_vis(depth_mat, depth_range=depth_range, ratio=ratio)
            depth_vis_all.append(depth_vis)
        pointer += 1  # move pointer
    depth_vis_all = np.stack(depth_vis_all, axis=0) if vis_depth else None
    return depth_clip, depth_vis_all, pointer


def compute_3d_trajectory(result_file, record_name):

    # read hand landmarks
    landmark_file = os.path.join(record_path, 'hand_frames', 'hand_landmarks.txt')
    if not os.path.exists(landmark_file):
        return False
    landmarks = read_landmarks(landmark_file, flip=True)
    # get the landmark text with flipped x value
    landmarks_text = read_landmarks_text(landmark_file, flip=True)

    # mkv video
    playback = PyK4APlayback(mkvraw_file)
    playback.open()
    pointer = 0

    # read clip ranges
    cliprng_file = os.path.join(record_path, 'hand_frames', 'clip_ranges.txt')
    clip_ranges = read_clip_ranges(cliprng_file)

    # process each clip
    results = []
    for cid, (start, end) in tqdm(clip_ranges.items(), desc=f'Record {record_name}', total=len(clip_ranges)):
        assert start < end, "invalid clip!"
        if end - start > 5400: # more than ~3min (invalid clip):
            continue
        # get 2D landmarks
        landmarks_2d = get_2d_coordinates(landmarks_text, start, end, resolution)

        # get depth frames
        depth_clip, depth_vis, pointer = get_depth(playback, start, end, pointer, ratio=vis_ratio, vis_depth=(cid==list(clip_ranges.keys())[0]))
        if depth_vis is not None:
            write_video(depth_vis, depth_visfile, fps=5)
        
        # get 3D landmarks
        trajectory_3d = get_3d_coordinates(landmarks, landmarks_2d, start, depth_clip=depth_clip)
        results.append(trajectory_3d)

    if len(results) > 0:
        with open(result_file.format('local'), 'wb') as f:
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    playback.close()
    return True


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
    return videos


def write_video(mat, video_file, fps=5):
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mat.shape[2], mat.shape[1]))
    for frame in mat:
        video_writer.write(frame)


def read_clip_ranges(rng_file):
    ranges = dict()
    with open(rng_file, 'r') as f:
        for line in f.readlines():
            cid = int(line.strip().split('. ')[0])
            start = int(line.strip().split('. ')[1].split(', ')[0])
            end = int(line.strip().split('. ')[1].split(', ')[1])
            ranges[cid] = (start, end)
    return ranges


def draw_on_frames(frame, idx_to_coords3d, ratio=0.25):
    # drawing config
    connection_drawing_spec = mp_style.get_default_hand_connections_style()
    landmark_drawing_spec = mp_style.get_default_hand_landmarks_style()
    
    def _XYZ_to_uv(X, Y, Z, ratio=1.0):
        u = X * 1.80820276e+03 / Z + 1.94228662e+03
        v = Y * 1.80794556e+03 / Z + 1.12382178e+03
        u, v = int(u * ratio), int(v * ratio)  # scaling
        return u, v

    # Draws the connections if the start and end landmarks are both visible.
    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]
        if start_idx in idx_to_coords3d and end_idx in idx_to_coords3d:
            drawing_spec = connection_drawing_spec[connection] if isinstance(
                connection_drawing_spec, Mapping) else connection_drawing_spec
            # start point (XYZ to uv)
            X, Y, Z = idx_to_coords3d[start_idx]  # (X, Y, Z) in meters
            if Z == 0: continue
            px_start, py_start = _XYZ_to_uv(X, Y, Z, ratio=ratio)
            # end point (XYZ to uv)
            X, Y, Z = idx_to_coords3d[end_idx]  # (X, Y, Z) in meters
            if Z == 0: continue
            px_end, py_end = _XYZ_to_uv(X, Y, Z, ratio=ratio)
            # draw line
            cv2.line(frame, (px_start, py_start), (px_end, py_end), drawing_spec.color, drawing_spec.thickness)
    
    # draw keypoints
    for idx, landmark_pxyz in idx_to_coords3d.items():
        drawing_spec = landmark_drawing_spec[idx] if isinstance(
        landmark_drawing_spec, Mapping) else landmark_drawing_spec
        # White circle border
        circle_border_radius = max(drawing_spec.circle_radius + 1,
                                    int(drawing_spec.circle_radius * 1.2))
        # (XYZ to uv)
        X, Y, Z = landmark_pxyz  # (X, Y, Z) in meters
        if Z == 0: continue
        px, py = _XYZ_to_uv(X, Y, Z, ratio=ratio)
        cv2.circle(frame, (px, py), circle_border_radius, WHITE_COLOR, drawing_spec.thickness)
        # Fill color into the circle
        cv2.circle(frame, (px, py), drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)


def draw_trajectory_3d(landmarks_3d, start, end, fps=5, size=1000, fig_size=5, views=[30, -120], minmax=[(0, 0, -1), (1, 1, 2)], savefile='temp.mp4'):

    xmin, ymin, zmin = minmax[0]
    xmax, ymax, zmax = minmax[1]

    # init a 3D figure
    fig = plt.figure(figsize=(fig_size, fig_size))
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

    # init video writer
    traj3d_writer = FFMpegWriter(fps=fps, metadata=dict(title='3D Trajectory', artist='Matplotlib'))
    # process
    with traj3d_writer.saving(fig, savefile, dpi=int(size / fig_size)):
        traj3d = []
        pts3d_last = np.array(list(list(landmarks_3d.values())[0].values()))
        for i in range(start, end):
            if i in landmarks_3d:
                pts3d = np.array(list(landmarks_3d[i].values()))
                pts3d_last = np.copy(pts3d)
            else:
                pts3d = np.copy(pts3d_last)
            # compute the 3d center of the hand as the trajectory point
            pt_hand = np.mean(pts3d[pts3d[:, 2] > 0], axis=0)
            traj3d.append(pt_hand)
            xs, ys, zs = np.array(traj3d)[:, 0], np.array(traj3d)[:, 1], np.array(traj3d)[:, 2]
            ax.plot(xs, zs, ys, '-o')
            ax.set_title('Hand trajectory [Frame: {}, View: ({}, {})]'.format(i-start, views[0], views[1]))
            # grab a frame
            traj3d_writer.grab_frame()  # dpi * (fig_size, fig_size)
    plt.close()


def vis(results, ratio=0.25):
    # video file path
    print('Read video...')
    video_file = os.path.join(record_path, 'rgb_video.mp4')
    video_all = read_video(video_file, ratio=ratio)  # (N, 540, 960, 3)

    # read clip ranges
    cliprng_file = os.path.join(record_path, 'hand_frames', 'clip_ranges.txt')
    clip_ranges = read_clip_ranges(cliprng_file)

    # visualize for each clip
    for cid, (start, end) in tqdm(clip_ranges.items(), desc='Visualizing clips', total=len(clip_ranges)):
        landmarks_3d = results[cid - 1]  # clip id starts from 1
        vis_mat = np.copy(video_all[start: end])

        for frame_id, idx_to_coords3d in landmarks_3d.items():
            # draw on video frame
            draw_on_frames(vis_mat[frame_id - start], idx_to_coords3d, ratio=ratio)

        # draw 3D trajectory video
        draw_trajectory_3d(landmarks_3d, start, end, fps=5, size=int(resolution[0]*ratio), views=[10, -75], minmax=[(-0.5, -0.5, 0), (0.5, 0.5, 1.0)], savefile='temp.mp4')
        video_traj3d = read_video('temp.mp4', ratio=1.0)  # (M, 540, 540, 3)

        # visualize RGB and 3D trajectory side-by-side
        vis_mat = np.concatenate([vis_mat, video_traj3d], axis=2)  # (M, 540, 1500, 3)

        vis_file = os.path.join(vis_dir, 'vis_{}_clip{}.mp4'.format(record_name, cid))
        write_video(vis_mat, vis_file, fps=5)
    
    print(len(results))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', action='store_true', help='vis the results')
    parser.add_argument('--scene', default=1, help='the scene id')
    args = parser.parse_args()
    scene_id = str(args.scene)
    
    root_path = os.path.join(os.path.dirname(__file__), '../../data')
    resolution = [2160, 3840]  # height, width
    vis_ratio = 0.25
    depth_range = [0, 1200]  # for depth visualization
    visited = False
    
    for record_name in sorted(os.listdir(os.path.join(root_path, 'EgoPAT3D-complete', scene_id))):
        # scene_name = 'bathroomCabinet'
        # record_id = '2'
        # record_name = '{}_{}'.format(scene_name, record_id)
        record_path = os.path.join(root_path, 'EgoPAT3D-complete', scene_id, record_name)  # './1/bathroomCabinet_2'
        mkvraw_file = os.path.join(root_path, 'EgoPAT3D-mkv', scene_id, record_name + '.mkv')  # './1/bathroomCabinet_2.mkv'
        if not os.path.isdir(record_path):
            continue
        
        # result trajectory path
        traj_path = os.path.join(root_path, 'trajectory', scene_id)
        os.makedirs(traj_path, exist_ok=True)
        traj_file = os.path.join(traj_path, record_name + '_{}.pkl')  # local, global
        
        # result depth visualization path
        depth_vispath = os.path.join(root_path, 'depth_vis{}'.format(vis_ratio), scene_id)
        os.makedirs(depth_vispath, exist_ok=True)
        depth_visfile = os.path.join(depth_vispath, record_name + '_clip0.mp4')

        if not os.path.exists(traj_file.format("local")):
            success = compute_3d_trajectory(traj_file, record_name)
        
        if args.vis and not visited:
            vis_dir = os.path.join(traj_path, record_name + '_vis')
            os.makedirs(vis_dir, exist_ok=True)
            
            # saved landmarks
            with open(traj_file.format("local"), 'rb') as f:
                results = pickle.load(f)
            vis(results, ratio=vis_ratio)
            visited = True  # only visualize one recording for each scene

    if os.path.exists('temp.mp4'):
        os.remove('temp.mp4')