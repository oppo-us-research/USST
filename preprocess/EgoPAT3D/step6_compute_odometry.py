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
import argparse
from tqdm import tqdm
import pickle
import numpy as np
try:
    # the pyk4a is only installed on monitor-connected Windows/Linux machine
    import pyk4a
    from pyk4a import PyK4APlayback, ImageFormat
    import open3d as o3d
except:
    pass
import cv2
from functools import reduce
import imageio


def read_video(video_file):
    assert os.path.exists(video_file), "File does not exist! {}".format(video_file)
    cap = cv2.VideoCapture(video_file)
    success, frame = cap.read()
    video = []
    while success:
        video.append(frame)
        success, frame = cap.read()
    video = np.array(video)
    return video


def read_traj_file(filepath, target='3d'):
    assert os.path.exists(filepath), "File does not exist! {}".format(filepath)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    traj2d = data['traj2d']
    traj3d = None
    if target == '3d':
        assert 'traj3d' in data, "3D trajectories do not exist in file: {}".format(filepath)
        traj3d = data['traj3d']
    num_preserve = data['num_preserve'] if 'num_preserve' in data else len(traj2d)
    return traj2d, traj3d, num_preserve


def read_odometry(odom_file):
    assert os.path.exists(odom_file), "File does not exist! {}".format(odom_file)
    all_transforms = np.load(odom_file)  # (T, 4, 4)
    return all_transforms


def convert_to_bgra_if_required(color_format, color_image):
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def get_rgb_depth(playback, start, end, pointer, ratio=1.0):
    """get depth video clip"""
    height = int(intrinsics['h'] * ratio)
    width = int(intrinsics['w'] * ratio)
    rgb_clip = np.zeros([end - start + 1, height, width, 3])  # (N, H, W, C)
    depth_clip = np.zeros([end - start + 1, height, width])  # (N, H, W)
    validities = np.zeros([end - start + 1], dtype=np.bool_)
    # move pointer to the clip start frame
    while pointer < start:
        capture = playback.get_next_capture()
        pointer += 1
    # get the depth of the clip
    while pointer < end + 1:
        capture = playback.get_next_capture()
        if (capture.color is None) or (capture.depth is None):
            pointer += 1  # move pointer
            has_invalid = True
            continue
        # get rgb frame
        rgb_mat = convert_to_bgra_if_required(playback.configuration["color_format"], capture.color)
        rgb_clip[pointer - start] = cv2.resize(rgb_mat, dsize=(width, height))
        # get depth frame
        depth_mat = pyk4a.depth_image_to_color_camera(capture.depth, playback.calibration, playback.thread_safe)
        depth_clip[pointer - start] = cv2.resize(depth_mat, dsize=(width, height))
        validities[pointer - start] = True
        pointer += 1  # move pointer
    return rgb_clip, depth_clip, pointer, validities


def compute_video_odometry(rgb_clip, depth_clip, validities):
    
    assert rgb_clip.shape[0] == depth_clip.shape[0]
    num_frames = rgb_clip.shape[0]
    
    cam_model = o3d.camera.PinholeCameraIntrinsic()
    cam_model.set_intrinsics(int(intrinsics['w']), int(intrinsics['h']), intrinsics['fx'], intrinsics['fy'], intrinsics['ox'], intrinsics['oy'])
    option = o3d.pipelines.odometry.OdometryOption()
    odo_method = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
    odo_init = np.identity(4)
    
    # construct the rgb-depth pair
    rgbd_list = []
    for t in range(num_frames):
        rgb = o3d.geometry.Image(rgb_clip[t].astype(np.uint8))
        depth = o3d.geometry.Image(depth_clip[t].astype(np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        rgbd_list.append(rgbd)
    
    # calculate odometry for each pair of adjacent RGBD frames
    all_transforms = [odo_init]
    for source, target, valid in zip(rgbd_list[1:], rgbd_list[:-1], validities[1:]):
        success, transform, info = o3d.pipelines.odometry.compute_rgbd_odometry(source, target, cam_model, odo_init, odo_method, option)
        if (not success) or (not valid):
            transform = np.identity(4)
        all_transforms.append(transform)
    
    return np.array(all_transforms)


def main():
    
    for scene_id in sorted(args.scenes):
        for record_name in sorted(os.listdir(os.path.join(traj_dir, scene_id))):
            # odometry results dir 
            odometry_result_path = os.path.join(odometry_dir, scene_id, record_name)
            os.makedirs(odometry_result_path, exist_ok=True)

            # trajectory file list must be sorted according to the clip order
            traj_folder = os.path.join(traj_dir, scene_id, record_name)
            all_trajfiles = list(filter(lambda x: x.endswith('.pkl'), os.listdir(traj_folder)))
            clip_ids = np.argsort([int(filename.split('_')[0][1:]) for filename in all_trajfiles])
            all_trajfiles = [all_trajfiles[i] for i in clip_ids]
            
            if len(os.listdir(odometry_result_path)) == len(all_trajfiles):
                continue
            
            # get depth from raw MKV videos
            mkv_filepath = os.path.join(mkv_dir, scene_id, record_name + '.mkv')
            playback = PyK4APlayback(mkv_filepath)
            playback.open()
            pointer = 0
            
            for traj_filename in tqdm(all_trajfiles, desc=f'Record {record_name}', total=len(all_trajfiles)):
                # result file
                odometry_file = os.path.join(odometry_result_path, traj_filename[:-4] + '.npy')

                # get depth clip
                start_frame = int(traj_filename.split(".pkl")[0].split('_')[1][1:])
                end_frame = int(traj_filename.split(".pkl")[0].split('_')[2][1:])
                rgb_clip, depth_clip, pointer, validities = get_rgb_depth(playback, start_frame, end_frame, pointer)
                
                # compute the odometry matrix for each frame
                all_transforms = compute_video_odometry(rgb_clip, depth_clip, validities)
                
                # save
                # write results to the pickle file
                with open(odometry_file, 'wb') as f:
                    np.save(f, all_transforms, allow_pickle=True)
            
            playback.close()


def XYZ_to_uv(traj3d, max_depth=3):
    """ traj3d: (T, 3), a list of 3D (X, Y,Z) points 
    """
    # transform the (X, Y, Z) into the (u,v) by PINHOLE camera model
    traj2d = np.zeros((traj3d.shape[0], 2), dtype=np.float32)
    traj2d[:, 0] = (traj3d[:, 0] * intrinsics['fx'] / traj3d[:, 2] + intrinsics['ox'])
    traj2d[:, 1] = (traj3d[:, 1] * intrinsics['fy'] / traj3d[:, 2] + intrinsics['oy'])
    # clip the coordinates 
    traj2d[:, 0] = np.clip(traj2d[:, 0], 0, intrinsics['w'])
    traj2d[:, 1] = np.clip(traj2d[:, 1], 0, intrinsics['h'])
    traj2d = np.floor(traj2d).astype(np.int32)
    
    return traj2d


def vis_frame_traj(frame, traj2d):
    """ frame: (H, W, 3)
        traj2d: (T, 2)
    """
    cv2.polylines(frame, [traj2d], False, (0, 255, 255))
    for t, uv in enumerate(traj2d):
        cv2.circle(frame, uv, radius=5, color=(0, 0, 255), thickness=-1)


def video_to_gif(video, giffile):
    assert giffile.endswith('.gif')
    with imageio.get_writer(giffile, mode='I', duration=0.2) as writer:
        for frame in video:
            writer.append_data(frame)
    
            
def visualize_transformed_traj(video, traj3d, odometry):
    """ video: (T, H, W, 3)
        traj3d: (T, 3)
        odometry: (T, 4, 4)
    """
    length = len(odometry)
    traj3d_homo = np.hstack((traj3d, np.ones((length, 1))))  # (T, 4)
    for i in range(length):
        # compute the transformed points of all future points
        traj_list = [traj3d[i]]  # initial 3D point
        for j in range(i + 1, length):
            odom = reduce(np.dot, odometry[(i+1):(j+1)])  # (4, 4)
            future_point = odom.dot(traj3d_homo[j].T)  # (4,)
            traj_list.append(future_point[:3])
        # visualize current frame
        traj2d = XYZ_to_uv(np.array(traj_list))
        vis_frame_traj(video[i], traj2d)


def vis_odometry():
    
    # examples to visualize
    examples = [{'scene': '1', 'record': 'bathroomCabinet_1', 'clip': 'c9_s439_e480_sx438_sy189_ex605_ey309'},
                {'scene': '1', 'record': 'bathroomCabinet_9', 'clip': 'c1_s64_e103_sx718_sy420_ex631_ey247'},
                {'scene': '2', 'record': 'bathroomCounter_10', 'clip': 'c1_s44_e152_sx600_sy381_ex510_ey415'},
                {'scene': '3', 'record': 'bin_7', 'clip': 'c3_s168_e204_sx550_sy315_ex494_ey354'}
                ]
    for data in examples:
        # input files
        video_file = os.path.join(video_dir, data['scene'], data['record'], data['clip'] + '.mp4')
        traj_file = os.path.join(traj_dir, data['scene'], data['record'], data['clip'] + '.pkl')
        odom_file = os.path.join(odometry_dir, data['scene'], data['record'], data['clip'] + '.npy')
        # read trajectory
        traj2d, traj3d, num_preserve = read_traj_file(traj_file)
        # read video 
        video = read_video(video_file)
        video = video[:num_preserve]
        # read odometry
        odometry = read_odometry(odom_file)
        odometry = odometry[:num_preserve]
        
        visualize_transformed_traj(video, traj3d, odometry)

        # write to GIF
        giffile = os.path.join(vis_path, 'odom_{}_{}_{}.gif'.format(data['scene'], data['record'], data['clip'].split("_")[0]))
        video_to_gif(video, giffile)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../../data/EgoPAT3D')
    parser.add_argument('--scenes', nargs='+', default=[])
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()
    
    # root path
    root_path = os.path.join(os.path.dirname(__file__), args.data_root)
    vis_ratio = 0.25
    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}
    for k, v in intrinsics.items():
        v *= vis_ratio  # scaling the camera intrinsics
        intrinsics[k] = v
    
    if not args.vis:  # not needed for visualization
        # mkv video path
        mkv_dir = os.path.join(root_path, 'EgoPAT3D-mkv')
        assert os.path.exists(mkv_dir), 'Path does not exist!'
    
    # (relative) trajectory path
    traj_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'trajectory_repair')
    assert os.path.exists(traj_dir), 'Path does not exist!'
    
    # result odometry path
    odometry_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'odometry')
    os.makedirs(odometry_dir, exist_ok=True)
    
    if len(args.scenes) == 0:
        args.scenes = os.listdir(traj_dir)
    print("Start processing scenes: {}".format(args.scenes))
    
    if args.vis:
        video_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'video_clips_hand')
        vis_path = os.path.join(os.path.dirname(__file__), '../../output/temp_odom_vis')
        os.makedirs(vis_path, exist_ok=True)
        # vis
        vis_odometry()
    else:
        main()
    
    print("Finished scenes: {}".format(args.scenes))
