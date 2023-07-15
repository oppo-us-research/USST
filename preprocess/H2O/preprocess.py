import os
import numpy as np
import cv2
import pickle
import sys
import copy



def frames_to_video(frames_path):
    filenames = os.listdir(frames_path)
    num_frames = len(filenames)
    video = []
    for i in range(num_frames):
        img_file = os.path.join(frames_path, '{:06d}.jpg'.format(i))
        assert os.path.exists(img_file), 'Frame {} does not exist!'.format(img_file)
        im = cv2.imread(img_file)
        video.append(im)  # H x W x 3
    video = np.array(video)
    return video


def write_video(mat, video_file, fps=30):
    video_writer = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (mat.shape[2], mat.shape[1]))
    for frame in mat:
        video_writer.write(frame)


def read_depth_map(depth_file):
    assert os.path.exists(depth_file), "Depth file does not exist! {}".format(depth_file)
    depth_img = cv2.imread(depth_file, cv2.IMREAD_ANYDEPTH)  # (720, 1280), uint16, min=0, max=3168
    depth_img = depth_img.astype(np.float32) / 1000.0  # in meters
    return depth_img


def XYZ_to_uv(traj3d, intrinsics):
    """ traj3d: (T, 3), a list of 3D (X, Y,Z) points in local coordinate system
    """
    # transform the (X, Y, Z) into the (u,v) by PINHOLE camera model
    traj2d = np.zeros((traj3d.shape[0], 2), dtype=np.float32)
    traj2d[:, 0] = (traj3d[:, 0] * intrinsics['fx'] / traj3d[:, 2] + intrinsics['cx'])
    traj2d[:, 1] = (traj3d[:, 1] * intrinsics['fy'] / traj3d[:, 2] + intrinsics['cy'])
    # clip the coordinates 
    traj2d[:, 0] = np.clip(traj2d[:, 0], 0, intrinsics['width']-1)
    traj2d[:, 1] = np.clip(traj2d[:, 1], 0, intrinsics['height']-1)
    traj2d = np.floor(traj2d).astype(np.int32)
    return traj2d

def world_to_uv(point, cam2world, intrinsics):
    # world to camera
    extrinsics = np.linalg.inv(cam2world)
    pt_homo = np.concatenate([point[:, np.newaxis], np.ones((1, 1))], axis=0)  # (4, 1)
    pt_local = extrinsics.dot(pt_homo)[:3, :]  # (3, 1)
    # project 3D coordinates onto pixel 2D coordinates
    pt_uv = XYZ_to_uv(pt_local.T, intrinsics)  # (1, 2)
    return pt_uv[0]


def uvd_to_world(uvd, cam2world, intrinsics):
    # uv_to_XYZ
    pt_local = np.zeros_like(uvd)
    pt_local[0] = (uvd[0] - intrinsics['cx']) * uvd[2] / intrinsics['fx']
    pt_local[1] = (uvd[1] - intrinsics['cy']) * uvd[2] / intrinsics['fy']
    pt_local[2] = uvd[2]
    # cam to world
    pt_homo = np.concatenate((pt_local[:, np.newaxis], np.array([[1]])), axis=0)  # (4,1)
    
    # E = np.linalg.inv(cam2world)
    # R = np.linalg.inv(E.T.dot(E)).dot(E.T)
    # pt_world = R.dot(pt_homo)

    # Note that this results is almost the same as the commented lines, 
    # which is an exact form of cam_to_world transformation.
    pt_world = cam2world.dot(pt_homo)
    return pt_world[:3, 0]

def replace_with_depth(point, depth_map, cam2world, intrinsics):
    """ point: 
    """ 
    uv = world_to_uv(point, cam2world, intrinsics)
    d = np.array([depth_map[uv[1], uv[0]]])
    uvd = np.concatenate([uv.astype(np.float32), d])
    pt_world = uvd_to_world(uvd, cam2world, intrinsics)
    return pt_world


def split_traj(traj, cam_poses, valids):
    """ traj: ndarray, (L, 3)
        cam_poses: (L, 4, 4)
        valids: list, (L,), binary
    """
    valids_extend = np.array([0] + valids + [0]).astype(int) # L+2
    starts = np.where(np.diff(valids_extend) == 1)[0]
    ends = np.where(np.diff(valids_extend) == -1)[0] - 1
    
    traj_data = []
    for s, e in zip(starts, ends):
        data = {
            'traj3d': traj[s: e+1],
            'cam2world': cam_poses[s: e+1],
            'start': s,
            'end': e
        }
        traj_data.append(data)
    return traj_data


def get_3d_traj(cam_dir, campose_folder, handpose_folder, depth_folder='depth', from_depth=False, intrinsics=None):
    # parse poses
    campose_dir = os.path.join(cam_dir, campose_folder)
    handpose_dir = os.path.join(cam_dir, handpose_folder)
    num_frames = len(os.listdir(handpose_dir))
    if from_depth:
        depth_dir = os.path.join(cam_dir, depth_folder)

    left_traj, right_traj, cam_poses = [], [], []
    left_valids, right_valids = [], []
    for i in range(num_frames):
        hpose_file = os.path.join(handpose_dir, '{:06d}.txt'.format(i))
        assert os.path.exists(hpose_file), 'Hand pose file {} does not exist!'.format(hpose_file)
        # parse hand pose file
        data = np.loadtxt(hpose_file)
        left_hasanno, left_hpose = data[0] == 1, data[1:64].reshape(-1, 3)
        right_hasanno, right_hpose = data[64] == 1, data[65:128].reshape(-1, 3)

        # compute the center of each hand by the 4 joints of fingers root (5, 9, 13, 7)
        left_ctr = np.mean(left_hpose[np.array([5, 9, 13, 17]), :], axis=0)
        right_ctr = np.mean(right_hpose[np.array([5, 9, 13, 17]), :], axis=0)

        # parse camera pose file
        cpose_file = os.path.join(campose_dir, '{:06d}.txt'.format(i))
        assert os.path.exists(cpose_file), 'Camera pose file {} does not exist!'.format(cpose_file)
        cam2world = np.loadtxt(cpose_file).reshape(4, 4)
        cam_poses.append(cam2world)

        if from_depth:
            # read depth data
            depth_map = read_depth_map(os.path.join(depth_dir, '{:06d}.png'.format(i)))
            # update the 3D coordinates by using depth from sensors
            left_ctr = replace_with_depth(left_ctr, depth_map, cam2world, intrinsics)
            right_ctr = replace_with_depth(right_ctr, depth_map, cam2world, intrinsics)

        # save
        left_traj.append(left_ctr)
        right_traj.append(right_ctr)
        left_valids.append(left_hasanno)
        right_valids.append(right_hasanno)
    
    # figure out which segment is a valid trajectory (hand exist)
    trajdata_left = split_traj(np.stack(left_traj, axis=0), np.stack(cam_poses, axis=0), left_valids)
    trajdata_right = split_traj(np.stack(right_traj, axis=0), np.stack(cam_poses, axis=0), right_valids)
    
    return trajdata_left, trajdata_right


def process(from_depth=False):
    for sub in subject_ids:
        subject_dir = os.path.join(data_root, 'subject{}_ego'.format(sub))
        if not os.path.isdir(subject_dir):
            continue

        for scene in sorted(os.listdir(subject_dir)):
            scene_dir = os.path.join(subject_dir, scene)
            if not os.path.isdir(scene_dir):
                continue 

            for obj in sorted(os.listdir(scene_dir)):
                obj_dir = os.path.join(scene_dir, obj)
                if not os.path.isdir(obj_dir):
                    continue

                cam_dir = os.path.join(obj_dir, cam)
                # dest name
                sample_name = 'sub{}_{}_{}'.format(sub, scene, obj)
                print("process sample: {}".format(sample_name))

                # read video frames
                vid_file = os.path.join(video_dir, sample_name + '.mp4')
                if not os.path.exists(vid_file):
                    # RGB video frames
                    video = frames_to_video(os.path.join(cam_dir, 'rgb256'))
                    # save as mp4 video file
                    write_video(video, vid_file)
                
                traj_file = os.path.join(traj_dir, sample_name + '.pkl')
                if not os.path.exists(traj_file):
                    # read camera intrinsics
                    data = np.loadtxt(os.path.join(cam_dir, 'cam_intrinsics.txt'))
                    intrinsics = {'fx': data[0], 'fy': data[1], 'cx': data[2], 
                                  'cy': data[3], 'width': data[4], 'height': data[5]}

                    # get 2D and 3D hand trajectory
                    trajdata_left, trajdata_right = get_3d_traj(cam_dir, 'cam_pose', 'hand_pose', 
                                                                depth_folder='depth', from_depth=from_depth, intrinsics=intrinsics)

                    # save results
                    with open(traj_file, 'wb') as f:
                        pickle.dump({'left_hand': trajdata_left, 
                                     'right_hand': trajdata_right,
                                     'intrinsics': intrinsics}, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_traj(traj_file):
    with open(traj_file, 'rb') as f:
        traj_data = pickle.load(f)
        trajdata_left = traj_data['left_hand']
        trajdata_right = traj_data['right_hand']
        intrinsics = traj_data['intrinsics']
    return trajdata_left, trajdata_right, intrinsics


def get_stats(trajdata_ref, trajdata_target):
    # compute ADE in 3D world space
    displace_errors = np.sqrt(np.sum((trajdata_ref - trajdata_target)**2, axis=-1))
    de_mean = np.mean(displace_errors)
    de_max = np.max(displace_errors)
    # compute absolute depth error
    depth_errors = np.fabs(trajdata_ref[:, 2] - trajdata_target[:, 2])
    dz_mean = np.mean(depth_errors)
    dz_max = np.max(depth_errors)
    # ratio
    ratio = len(np.where(displace_errors > 0.2)[0]) / len(displace_errors)

    print("DE(mean): {:.3f} mm, DE(max): {:.3f} mm, DZ(mean): {:.3f} mm, DZ(max): {:.3f} mm, P(DE > 0.20) = {:.2f}%".format(
        de_mean*1000, de_max*1000, dz_mean*1000, dz_max*1000, ratio * 100))
    
    all_stats = {
        'displace_errors': displace_errors,
        'ratio': ratio,
        'de_mean': de_mean,
        'dz_mean': dz_mean
    }
    return all_stats


def plot_frequency(errors_before, errors_after, ratio_before, ratio_after):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 4))
    fontsize = 18
    sns.kdeplot(errors_before, color='red', fill=True, label='raw depth, P(DE>0.2)={:.2f}%'.format(ratio_before * 100))
    sns.kdeplot(errors_after, color='blue', fill=True, label='depth repair, P(DE>0.2)={:.2f}%'.format(ratio_after * 100))
    # bins = np.linspace(0, 1, 1000)
    # ax.hist(errors_before, bins, lw=1, ec="red", fc="red", alpha=0.5, label='before repair')
    # ax.hist(errors_after, bins, lw=1, ec="blue", fc="blue", alpha=0.5, label='after repair')
    plt.xlim(0,0.8)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.xlabel('Displacement Error (m)', fontsize=fontsize)
    plt.ylabel('Frequency', fontsize=fontsize)
    plt.legend(loc='upper right', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(os.path.join(dest_dir, 'compare_hist.png'))
    plt.savefig(os.path.join(dest_dir, 'compare_hist.pdf'))


def plot_stat_bars(stats_before, stats_after, fontsize=18):
    import matplotlib.pyplot as plt
    # plt.rcParams['font.family'] = 'DeJavu Serif'
    # plt.rcParams['font.serif'] = ['Times New Roman']
    measures = ('mDE', 'mDZ')
    values = {
        'H2O-DT w/o repair': (stats_before['de_mean'] * 100, stats_before['dz_mean'] * 100),
        'H2O-DT': (stats_after['de_mean'] * 100, stats_after['dz_mean'] * 100)
    }
    x = np.arange(len(measures))  # the label locations
    width = 0.3  # the width of the bars
    multiplier = 0.5
    fig, ax = plt.subplots(figsize=(5,4))

    for k, v in values.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, v, width, label=k)
        ax.bar_label(rects, padding=4, fontsize=fontsize, fmt='%.1f')
        multiplier += 1
    
    plt.yticks(np.arange(0, 25, 4), fontsize=fontsize)
    ax.set_ylabel('distance error (cm)', fontsize=fontsize)
    ax.set_ylim(0, 25)
    ax.set_xticks(x + width, measures, fontsize=fontsize)
    ax.set_xlim(-0.45, 2)
    ax.legend(loc='upper right', fontsize=fontsize-3)
    plt.tight_layout()
    plt.savefig(os.path.join(dest_dir, 'compare_bars.png'))
    plt.savefig(os.path.join(dest_dir, 'compare_bars.pdf'))


def compare_trajectories():
    """ Compare the trajecotries obtained by methods with & without depth images
    """
    trajdata_all, trajdata_all_withdepth, trajdata_all_repair = [], [], []
    for sample in sorted(os.listdir(traj_dir)):
        # read trajectory data from 3D pose
        trajdata_left, trajdata_right, intrinsics = read_traj(os.path.join(traj_dir, sample))

        trajdata = np.concatenate([np.concatenate([traj['traj3d'] for traj in trajdata_left], axis=0),
                                   np.concatenate([traj['traj3d'] for traj in trajdata_right], axis=0)], axis=0)
        trajdata_all.append(trajdata)

        # read trajectory data from 3D pose + repair
        trajdata_left, trajdata_right, intrinsics = read_traj(os.path.join(traj_dir + '_depth', sample))
        trajdata_withdepth = np.concatenate([np.concatenate([traj['traj3d'] for traj in trajdata_left], axis=0),
                                             np.concatenate([traj['traj3d'] for traj in trajdata_right], axis=0)], axis=0)
        trajdata_all_withdepth.append(trajdata_withdepth)

        # read trajectory data from 3D pose + repair
        trajdata_left, trajdata_right, intrinsics = read_traj(os.path.join(traj_dir + '_repair', sample))
        trajdata_repair = np.concatenate([np.concatenate([traj['traj3d'] for traj in trajdata_left], axis=0),
                                          np.concatenate([traj['traj3d'] for traj in trajdata_right], axis=0)], axis=0)
        trajdata_all_repair.append(trajdata_repair)
    
    trajdata_all = np.concatenate(trajdata_all, axis=0)  # 682659
    trajdata_all_withdepth = np.concatenate(trajdata_all_withdepth, axis=0)
    trajdata_all_repair = np.concatenate(trajdata_all_repair, axis=0)

    print("==> Before repair:")
    stats_before = get_stats(trajdata_all, trajdata_all_withdepth)
    # DE(mean): 170.892 mm, DE(max): 1343.636 mm, DZ(mean): 160.702 mm, DZ(max): 1173.206 mm (before repair)

    print("==> After repair:")
    stats_after = get_stats(trajdata_all, trajdata_all_repair)
    # DE(mean): 133.106 mm, DE(max): 888.224 mm, DZ(mean): 115.336 mm, DZ(max): 700.663 mm. (after repair)
    
    plot_frequency(stats_before['displace_errors'], stats_after['displace_errors'], stats_before['ratio'], stats_after['ratio'])

    plot_stat_bars(stats_before, stats_after)



def repair_by_lsq(traj_raw, traj_ref, test_data, num_frames, threshold):
    from scipy.optimize import leastsq
    def model_func(p, x):
        f = np.poly1d(p[:-2])
        y = f(x) + p[-2] * np.sin(p[-1]*x)
        return y

    def error_func(p, x, y):
        error = y - model_func(p, x)
        return error
    
    fit_data = [i for i in range(num_frames) if i not in test_data]
    fit_labels = [traj_raw[i, 2] for i in fit_data]
    
    # least square fitting by third-order multinomial + sin model
    param_init = [0, 1, 1, 0, 1, 0.05]
    ret = leastsq(error_func, param_init, 
                    args=(np.array(fit_data), np.array(fit_labels)), epsfcn=0.0001)  # start with straight line: y=0

    traj_repaired = np.copy(traj_raw)
    for i in test_data:
        depth_pred = model_func(ret[0], i)  # repair by LSQ modelz
        if np.fabs(depth_pred - traj_ref[i, 2]) < 2 * threshold:
            traj_repaired[i, 2] = depth_pred
    
    return traj_repaired


def repair_hand_traj(trajdata_hand_raw, trajdata_hand_ref, threshold, MIN_VALID_POINTS):

    repaired = False
    trajdata_repaired = copy.deepcopy(trajdata_hand_raw)
    for i, (trajdata_raw, trajdata_ref) in enumerate(zip(trajdata_hand_raw, trajdata_hand_ref)):
        traj_raw = np.clip(trajdata_raw['traj3d'], 0.1, 1.0) if CLIP_VAL else trajdata_raw['traj3d']
        traj_ref = trajdata_ref['traj3d']
        depth_errors = np.fabs(traj_raw[:, 2] - traj_ref[:, 2])
        num_frames = len(traj_raw)

        # find invalid trajectory points
        invalids = np.where(depth_errors > threshold)[0].tolist() # input frame indices whose depths need to be repaired
        # repair
        if len(invalids) > 0 and len(invalids) <= num_frames - MIN_VALID_POINTS:
            traj_repaired = repair_by_lsq(traj_raw, traj_ref, invalids, num_frames, threshold)
            trajdata_repaired[i]['traj3d'] = traj_repaired
            repaired = True
        
        if CLIP_VAL:
            # clamp the depth into [0.1, 1]
            zs = trajdata_repaired[i]['traj3d'][:, 2]
            if (zs < 0.1).any() or (zs > 1.0).any():
                trajdata_repaired[i]['traj3d'][:, 2] = np.clip(zs, 0.1, 1.0)
    
    return trajdata_repaired, repaired



def repair_depth(threshold=0.20):
    """ Repair the depth values for those trajectories that have large depth error
    """
    for sample in sorted(os.listdir(traj_dir)):
        # read trajectory data from 3D pose
        trajdata_left_ref, trajdata_right_ref, _ = read_traj(os.path.join(traj_dir, sample))
        # read trajectory data from 3D pose + depth
        trajdata_left_raw, trajdata_right_raw, intrinsics = read_traj(os.path.join(traj_dir + '_depth', sample))

        # repairing
        trajdata_left_repaired, repaired1 = repair_hand_traj(trajdata_left_raw, trajdata_left_ref, threshold, MIN_VALID_POINTS)
        trajdata_right_repaired, repaired2 = repair_hand_traj(trajdata_right_raw, trajdata_right_ref, threshold, MIN_VALID_POINTS)
        
        print("repair sample: {}({})".format(sample[:-4], 'repaired' if repaired1 or repaired2 else ''))

        # save
        with open(os.path.join(traj_repair_dir, sample), 'wb') as f:
            pickle.dump({'left_hand': trajdata_left_repaired, 
                         'right_hand': trajdata_right_repaired,
                         'intrinsics': intrinsics}, f, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    data_root = '../../data/H2O/ego'
    cam = 'cam4'  # egocentric camera
    subject_ids = ['1', '2', '3', '4']

    from_depth = True if len(sys.argv) > 1 and sys.argv[1] == 'from_depth' else False

    # destination dirs
    dest_dir = '../../data/H2O/Ego3DTraj'
    os.makedirs(dest_dir, exist_ok=True)

    video_dir = os.path.join(dest_dir, 'video')
    os.makedirs(video_dir, exist_ok=True)

    traj_dir = os.path.join(dest_dir, 'traj')
    if from_depth: traj_dir += '_depth'
    os.makedirs(traj_dir, exist_ok=True)

    if len(sys.argv) > 1 and sys.argv[1] == 'compare': 
        compare_trajectories()
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'repair':
        traj_repair_dir = os.path.join(dest_dir, 'traj_repair')
        os.makedirs(traj_repair_dir, exist_ok=True)
        # MIN_VALID_POINTS=10
        MIN_VALID_POINTS=6
        CLIP_VAL=False
        repair_depth(threshold=0.20)
    
    else:
        process(from_depth=from_depth)
