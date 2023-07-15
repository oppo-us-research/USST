""" In this step, we use the provided manually annotated clip ranges and the hand landmarks
    to find the exact temporal region of each hand trajectory. Then, video clip of each trajectory is stored as mp4 file.
    At the same time, we resized the videos into 1/16 (1/4 x 1/4) of original 4K size.
"""
import os
from tqdm import tqdm
import cv2
import numpy as np


def read_clip_ranges(rng_file):
    ranges = dict()
    with open(rng_file, 'r') as f:
        for line in f.readlines():
            cid = int(line.strip().split('. ')[0])
            start = int(line.strip().split('. ')[1].split(', ')[0])
            end = int(line.strip().split('. ')[1].split(', ')[1])
            ranges[cid] = (start, end)
    return ranges



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


def read_video(video_file, ratio=0.25):
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


def get_hand_clip(start, end, landmarks):
    # get the first frame which has landmarks of hands
    first_frame = start
    while first_frame < end:
        if first_frame in landmarks:
            break
        first_frame += 1
    # get the last frame which has landmakrs of hands
    last_frame = end
    while last_frame > first_frame:
        if last_frame in landmarks:
            break
        last_frame -= 1
    return first_frame, last_frame


def compute_hand_center(landmarks):
    """ landmarks: (21.3) """
    pxyz = np.copy(landmarks)  # (21, 3)
    pxyz[1:, 2] += pxyz[0, 2]  # z values of top 20 are relative offsets to the 21-th z value
    center = np.mean(pxyz, axis=0)  # (3,)
    return center


def write_video(mat, video_file, fps=30):
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mat.shape[2], mat.shape[1]))
    for frame in mat:
        video_writer.write(frame)



if __name__ == '__main__':
    
    """ get all hand clips with landmarks in first frame """
    
    # input: hand landmarks directory
    root_path = os.path.join(os.path.dirname(__file__), '../../data/EgoPAT3D')
    resolution = [2160, 3840]  # height, width
    vis_ratio = 0.25
    CLIP_MIN, CLIP_MAX = 10, 5400  # more than ~3min or less than 10 frames (invalid clip):
    
    # output: first frame with hand center warpped by RAFT
    output_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'video_clips_hand')
    os.makedirs(output_dir, exist_ok=True)
    
    input_dir = os.path.join(root_path, 'EgoPAT3D-complete')
    for scene_id in sorted(os.listdir(input_dir)):
        # output path
        result_scene = os.path.join(output_dir, scene_id)
        os.makedirs(result_scene, exist_ok=True)
        
        for record_name in sorted(os.listdir(os.path.join(input_dir, scene_id))):
            record_path = os.path.join(input_dir, scene_id, record_name)
            if not os.path.isdir(record_path):
                continue
            
            # read hand landmarks
            landmark_file = os.path.join(record_path, 'hand_frames', 'hand_landmarks.txt')
            if not os.path.exists(landmark_file):
                continue  # discard record video without landmarks
            landmarks = read_landmarks(landmark_file, flip=True)
            
            # clip range file
            cliprng_file = os.path.join(record_path, 'hand_frames', 'clip_ranges.txt')
            clip_ranges = read_clip_ranges(cliprng_file)
            
            # output path
            result_record = os.path.join(result_scene, record_name)
            os.makedirs(result_record, exist_ok=True)
            
            # video file
            video_file = os.path.join(record_path, 'rgb_video.mp4')
            rgb_video, fps = read_video(video_file, vis_ratio)
            
            for cid, (start, end) in tqdm(clip_ranges.items(), desc=f'Record {record_name}', total=len(clip_ranges)):
                # get the first and last frames
                first_frame, last_frame = get_hand_clip(start, end, landmarks)
                
                # ignore too long or too short clips
                num_frames = last_frame - first_frame + 1
                if num_frames > CLIP_MAX or num_frames < CLIP_MIN:
                    continue
                
                # hand center (2D) of the start frame
                hand_center_1 = compute_hand_center(landmarks[first_frame])  # (u,v,z)
                px1 = int(resolution[1] * vis_ratio * hand_center_1[0])
                py1 = int(resolution[0] * vis_ratio * hand_center_1[1])
                
                # hand center (2D) of the end frame
                hand_center_2 = compute_hand_center(landmarks[last_frame])  # (u,v,z)
                px2 = int(resolution[1] * vis_ratio * hand_center_2[0])
                py2 = int(resolution[0] * vis_ratio * hand_center_2[1])
                
                # save frame
                result_clip_file = os.path.join(result_record, 'c{}_s{}_e{}_sx{}_sy{}_ex{}_ey{}.mp4'.format(cid, first_frame, last_frame, px1, py1, px2, py2))
                if not os.path.exists(result_clip_file):
                    write_video(rgb_video[first_frame: last_frame + 1], result_clip_file, fps=fps)
            