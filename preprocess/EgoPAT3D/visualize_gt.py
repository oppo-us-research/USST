import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from google.protobuf import text_format
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions import drawing_utils
import mediapipe.python.solutions.drawing_styles as mp_style
import mediapipe.python.solutions.hands as mp_hands


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


def read_landmarks(landmark_file):
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
                landmarks[frame_id][pid, 0] = 1.0 - float(line.split(': ')[1])
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


def generate_landmarks(start, end, origin=[0, 0, 0]):
    landmarks, landmarks_text = dict(), dict()
    # define a unit 3D cude, with pre-defined coordinate system
    # rightward: x, downward: y, innerward: z, cube center: origin
    cube = np.array([[1, -1, -1], [1, -1, 1], [-1, -1, 1], [-1, -1, -1],
                     [1, 1, -1], [1, 1, 1], [-1, 1, 1], [-1, 1, -1]], dtype=np.float32) * 0.5  # (8, 3)
    cube += origin
    step = int((end - start + 1) / 8)  # the number of frames for each of the 8 corners
    for s in range(8):
        t_start, t_end = start + s * step, start + (s+1) * step
        lm_points = np.repeat(np.expand_dims(cube[s], axis=0), 21, axis=0)  # (21, 3)
        lm_points[1:, 2] -= lm_points[0, 2]  # relative z
        for frame_id in range(t_start, t_end):
            landmarks[frame_id] = np.copy(lm_points)
            landmarks_text[frame_id] = ''.join(['landmark {' + 'x: {} y: {} z: {}'.format(pt[0], pt[1], pt[2]) + '} ' for pt in lm_points])
    return landmarks, landmarks_text



def read_clip_ranges(rng_file):
    ranges = dict()
    with open(rng_file, 'r') as f:
        for line in f.readlines():
            cid = int(line.strip().split('. ')[0])
            start = int(line.strip().split('. ')[1].split(', ')[0])
            end = int(line.strip().split('. ')[1].split(', ')[1])
            ranges[cid] = (start, end)
    return ranges


def write_video(mat, video_file, fps=5):
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mat.shape[2], mat.shape[1]))
    for frame in mat:
        video_writer.write(frame)


def draw_trajectory_3d(landmarks, start, end, pts3d_last, size=1000, fig_size=5, views=[30, -120], minmax=[(0, 0, -1), (1, 1, 1)], savefile='temp.mp4'):
    # find the range
    # pts_all = np.concatenate([landmarks[i] for i in range(start, end) if i in landmarks], axis=0)
    # pts_all = np.concatenate([pts_all, np.array([[0,0,0]])], axis=0)
    # xmin, ymin, zmin = np.min(pts_all, axis=0)
    # xmax, ymax, zmax = np.max(pts_all, axis=0)
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
    traj3d_writer = FFMpegWriter(fps=5, metadata=dict(title='3D Trajectory', artist='Matplotlib'))
    # process
    with traj3d_writer.saving(fig, savefile, dpi=int(size / fig_size)):
        traj3d = []
        for i in range(start, end):
            if i in landmarks:
                pts3d = np.copy(landmarks[i])
                pts3d_last = np.copy(pts3d)
            else:
                pts3d = np.copy(pts3d_last)
            pts3d[1:, 2] += pts3d[0, 2]
            pt_hand = np.mean(pts3d, axis=0)
            traj3d.append(pt_hand)
            xs, ys, zs = np.array(traj3d)[:, 0], np.array(traj3d)[:, 1], np.array(traj3d)[:, 2]
            ax.plot(xs, zs, ys, '-o')
            ax.set_title('Hand trajectory [Frame: {}, View: ({}, {})]'.format(i-start, views[0], views[1]))
            # grab a frame
            traj3d_writer.grab_frame()  # dpi * (fig_size, fig_size)
    plt.close()


def draw_landmarks_2d(video, landmarks_text, start, end):
    video_anno = video[start: end]
    for i in range(start, end):
        if i not in landmarks_text:
            continue
        # draw landmarks
        landmark_list = text_format.Parse(landmarks_text[i], landmark_pb2.NormalizedLandmarkList())
        drawing_utils.draw_landmarks(
            video_anno[i - start],
            landmark_list,
            mp_hands.HAND_CONNECTIONS,
            mp_style.get_default_hand_landmarks_style(),
            mp_style.get_default_hand_connections_style())
    return video_anno


def main():
    # vis path
    vis_dir = os.path.join(root_path, './vis')
    os.makedirs(vis_dir, exist_ok=True)

    # read video
    print('Read video...')
    video_file = os.path.join(record_path, 'rgb_video.mp4')
    video_all = read_video(video_file, ratio=0.25)  # (N, 540, 960, 3)

    # read hand landmarks
    landmark_file = os.path.join(record_path, 'hand_frames', 'hand_landmarks.txt')
    landmarks = read_landmarks(landmark_file)
    pts3d_last = list(landmarks.items())[0][1]  # (21, 3)

    landmarks_text = read_landmarks_text(landmark_file, flip=True)

    # read clip ranges
    cliprng_file = os.path.join(record_path, 'hand_frames', 'clip_ranges.txt')
    clip_ranges = read_clip_ranges(cliprng_file)

    # visualize for each clip
    for cid, (start, end) in tqdm(clip_ranges.items(), desc='Visualizing clips', total=len(clip_ranges)):
        # draw trajectory and transform into video
        draw_trajectory_3d(landmarks, start, end, pts3d_last, size=video_all.shape[1], views=[10, -75], savefile='temp.mp4')
        video_traj3d = read_video('temp.mp4', ratio=1.0)  # (M, 540, 540, 3)

        # draw landmarks on RGB video
        # video_rgb = video_all[start: end]
        video_rgb = draw_landmarks_2d(video_all, landmarks_text, start, end)

        # visualize RGB and 3D trajectory side-by-side
        mat_vis = np.concatenate([video_rgb, video_traj3d], axis=2)  # (M, 540, 1500, 3)

        # write result video
        vis_file = os.path.join(vis_dir, 'vis_{}_{}_clip{}.mp4'.format(scene_name, record_id, cid))
        write_video(mat_vis, vis_file, fps=5)


def coord_check():

    # vis path
    vis_dir = os.path.join(root_path, './vis_check')
    os.makedirs(vis_dir, exist_ok=True)

    # read video
    print('Read video...')
    video_file = os.path.join(record_path, 'rgb_video.mp4')
    video_all = read_video(video_file, ratio=0.25)  # (N, 540, 960, 3)

    # read clip ranges
    cliprng_file = os.path.join(record_path, 'hand_frames', 'clip_ranges.txt')
    clip_ranges = read_clip_ranges(cliprng_file)
    cid = 1
    start, end = clip_ranges[cid]

    # generate fake landmarks
    landmarks, landmarks_text = generate_landmarks(start, end, origin=[0.5, 0.5, 0])
    pts3d_last = list(landmarks.items())[0][1]  # (21, 3)

    # draw trajectory and transform into video
    draw_trajectory_3d(landmarks, start, end, pts3d_last, size=video_all.shape[1], views=[10, -75], minmax=[(-1,-1,-1),(1,1,1)], savefile='temp.mp4')
    video_traj3d = read_video('temp.mp4', ratio=1.0)  # (M, 540, 540, 3)

    # # draw landmarks on RGB video
    # video_rgb = video_all[start: end]
    video_rgb = draw_landmarks_2d(video_all, landmarks_text, start, end)

    # visualize RGB and 3D trajectory side-by-side
    mat_vis = np.concatenate([video_rgb, video_traj3d], axis=2)  # (M, 540, 1500, 3)

    # write result video
    vis_file = os.path.join(vis_dir, 'vis_{}_{}_clip{}.mp4'.format(scene_name, record_id, cid))
    write_video(mat_vis, vis_file, fps=5)




if __name__ == '__main__':

    scene_id = '1'
    scene_name = 'bathroomCabinet'
    record_id = '2'

    root_path = os.path.dirname(__file__, "../../data/EgoPAT3D")
    record_path = os.path.join(root_path, 'complete', scene_id, '{}_{}'.format(scene_name, record_id))  # 'complete/1/bathroomCabinet_2'

    # # coordinate check
    # coord_check()

    main()

    if os.path.exists('temp.mp4'):
        os.remove('temp.mp4')