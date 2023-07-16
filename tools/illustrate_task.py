import os
from demo import read_traj_file, read_video, read_odometry
from src.utils import video_to_gif
import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import reduce
from matplotlib.animation import FFMpegWriter

TRAJ_COLOR = (255, 255, 0)  # yellow
PAST_COLOR = (255, 0, 0)    # red
FUTURE_COLOR = (0, 0, 255)  # blue



def XYZ_to_uv(traj3d, intrinsics):
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


def vis_frame_traj(frame, traj2d, traj_start=0, future_start=15):
    cv2.polylines(frame, [traj2d], False, TRAJ_COLOR, thickness=2)
    for t, uv in enumerate(traj2d):
        color = PAST_COLOR if traj_start + t <= future_start else FUTURE_COLOR
        cv2.circle(frame, uv, radius=5, color=color, thickness=-1)


def visualize_transformed_traj(video, traj3d, odometry, intrinsics, ratio=0.5):
    """ video: (T, H, W, 3)
        traj3d: (T, 3)
        odometry: (T, 4, 4)
    """
    length = len(odometry)
    traj3d_homo = np.hstack((traj3d, np.ones((length, 1))))  # (T, 4)
    future_start = int(length * ratio)
    for i in range(length):
        # compute the transformed points of all future points
        traj_list = [traj3d[i]]  # initial 3D point
        for j in range(i + 1, length):
            odom = reduce(np.dot, odometry[(i+1):(j+1)])  # (4, 4)
            future_point = odom.dot(traj3d_homo[j].T)  # (4,)
            traj_list.append(future_point[:3])
        # visualize current frame
        traj2d = XYZ_to_uv(np.array(traj_list), intrinsics)
        vis_frame_traj(video[i], traj2d, traj_start=i, future_start=future_start)


def local_to_global(traj3d, odometry):
    """ Transform local 3D coordinates into the Global 3D coordinates
        traj3d: (T, 3)
        odometry: (T, 4, 4)
    """
    length = len(odometry)
    traj3d_homo = np.hstack((traj3d, np.ones((length, 1))))  # (T, 4)
    traj3d_global = []
    for i in range(length):
        # compute the transformed points of all future points
        traj_list = [traj3d[i]]  # initial 3D point
        for j in range(i + 1, length):
            odom = reduce(np.dot, odometry[(i+1):(j+1)])  # (4, 4)
            future_point = odom.dot(traj3d_homo[j].T)  # (4,)
            traj_list.append(future_point[:3])
        # record
        traj3d_global.append(np.array(traj_list))
    return traj3d_global



def draw_trajectory_3d(traj3d, video_size, ratio=0.5, fps=5, dpi=30, views=[30, -120], minmax=[(0, 0, -1), (1, 1, 2)], savefile='temp.mp4'):

    xmin, ymin, zmin = minmax[0]
    xmax, ymax, zmax = minmax[1]
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

    # init video writer
    traj3d_writer = FFMpegWriter(fps=fps, metadata=dict(title='3D Trajectory', artist='Matplotlib'))
    future_start = int(len(traj3d) * ratio)
    # process
    with traj3d_writer.saving(fig, savefile, dpi=dpi):
        traj3d_past, traj_future = [], []
        for t, pt in enumerate(traj3d):
            if t <= future_start:
                traj3d_past.append(pt)
            else:
                traj_future.append(pt)
            # draw past trajectory
            xs, ys, zs = np.array(traj3d_past)[:, 0], np.array(traj3d_past)[:, 1], np.array(traj3d_past)[:, 2]
            ax.plot(xs, zs, ys, '-o', color='red')
            if t > future_start:
                # draw future trajectory
                xs, ys, zs = np.array(traj_future)[:, 0], np.array(traj_future)[:, 1], np.array(traj_future)[:, 2]
                ax.plot(xs, zs, ys, '-o', color='blue')
            # grab a frame
            traj3d_writer.grab_frame()
    plt.close()


if __name__ == '__main__':

    data_root = os.path.join(os.path.dirname(__file__), '../data', 'EgoPAT3D')
    video_dir = os.path.join(data_root, 'EgoPAT3D-postproc', 'video_clips_hand')
    traj_dir = os.path.join(data_root, 'EgoPAT3D-postproc', 'trajectory_repair')
    odometry_dir = os.path.join(data_root, 'EgoPAT3D-postproc', 'odometry')
    
    data = {'scene': '1', 'record': 'bathroomCabinet_1', 'clip': 'c9_s439_e480_sx438_sy189_ex605_ey309'}
    # data = {'scene': '1', 'record': 'bathroomCabinet_2', 'clip': 'c61_s3163_e3201_sx567_sy385_ex487_ey207'}
    # data = {'scene': '1', 'record': 'bathroomCabinet_3', 'clip': 'c53_s2632_e2662_sx603_sy389_ex558_ey301'}
    
    folder = 'scene{}_record{}_{}'.format(data['scene'], data['record'].split('_')[-1], data['clip'].split('_')[0])
    result_dir = 'output/demo_task/{}'.format(folder)
    os.makedirs(result_dir, exist_ok=True)

    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}  # from EgoPAT3D preprocessing code
    vis_ratio = 0.25
    max_depth = 3
    for k, v in intrinsics.items():
        v *= vis_ratio  # scaling the camera intrinsics
        intrinsics[k] = v
    obs_ratio = 0.5

    # input files
    video_file = os.path.join(video_dir, data['scene'], data['record'], data['clip'] + '.mp4')
    traj_file = os.path.join(traj_dir, data['scene'], data['record'], data['clip'] + '.pkl')
    odom_file = os.path.join(odometry_dir, data['scene'], data['record'], data['clip'] + '.npy')
    # read trajectory
    traj2d, traj3d, num_preserve = read_traj_file(traj_file)
    # read video 
    video = read_video(video_file, toRGB=True)  # RGB video
    video = video[:num_preserve]
    # read odometry
    odometry = read_odometry(odom_file)
    odometry = odometry[:num_preserve]
    num_obs = int(num_preserve * obs_ratio)
    
    # visualize 2D trajectory on video
    visualize_transformed_traj(video, traj3d, odometry, intrinsics, ratio=obs_ratio)

    # write to GIF
    filename = 'vis_scene{}_record{}_clip.gif'.format(data['scene'], data['record'].split('_')[-1], data['clip'].split('_')[0])
    giffile = os.path.join(result_dir, filename)
    video_to_gif(video, giffile, toBGR=True)
    
    traj3d_global = local_to_global(traj3d, odometry)
    traj3d = traj3d_global[0]  # only refer to the camera of the first frame 

    print('x_min={:.6f}, x_max={:.6f} \ny_min={:.6f}, y_max={:.6f} \nz_min={:.6f}, z_max={:.6f}'.format(
        np.min(traj3d[:, 0]), np.max(traj3d[:, 0]),
        np.min(traj3d[:, 1]), np.max(traj3d[:, 1]),
        np.min(traj3d[:, 2]), np.max(traj3d[:, 2])))

    temp_file = os.path.join(result_dir, 'temp.mp4')

    # visualize 3D trajectory
    canvas_size = min(intrinsics['h'], intrinsics['w']) * 0.82
    draw_trajectory_3d(traj3d, (canvas_size, canvas_size), ratio=obs_ratio,
        fps=5, dpi=90, views=[30, -120], minmax=[(-0.1, -0.2, 0.1), (0.3, 0.1, 0.7)], savefile=temp_file)
    video_traj3d = read_video(temp_file, toRGB=True)
    video_to_gif(video_traj3d, os.path.join(result_dir, 'traj3d.gif'), toBGR=True)
    
    # overlap two frames
    for t, name in zip([0, num_obs, -1], ['start', 'current', 'end']):
        frame_vis = np.copy(video[t])
        frame3d = np.copy(video_traj3d[t])
        bottom_left = frame_vis[frame_vis.shape[0] - frame3d.shape[0]:, :frame3d.shape[1], :]
        frame_vis[frame_vis.shape[0] - frame3d.shape[0]:, :frame3d.shape[1], :] = cv2.addWeighted(bottom_left, 0.4, frame3d, 0.6, 0)  # bottom left
        cv2.imwrite(os.path.join(result_dir, 'task_{}.png'.format(name)), frame_vis)

    # overlap the two videos
    video_vis = np.copy(video)
    for i, (traj, rgb) in enumerate(zip(video_traj3d, video)):
        bottom_left = rgb[rgb.shape[0] - traj.shape[0]:, :traj.shape[1], :]
        rgb[rgb.shape[0] - traj.shape[0]:, :traj.shape[1], :] = cv2.addWeighted(bottom_left, 0.4, traj, 0.6, 0)  # bottom left
        video_vis[i] = rgb
    video_to_gif(video_vis, os.path.join(result_dir, 'demo.gif'), toBGR=True)

    # save frames
    frame_dir = os.path.join(result_dir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    for t, frame in enumerate(video_vis):
        # add frame number
        cv2.putText(frame, "#{}".format(t+1).zfill(2), (frame.shape[1] - 250, 150), cv2.FONT_HERSHEY_SIMPLEX, 4, color=(0,255,255), thickness=5)
        cv2.imwrite(os.path.join(frame_dir, '{}.png'.format(t + 1)), frame)
    
    if os.path.exists(temp_file):
        os.remove(temp_file)

