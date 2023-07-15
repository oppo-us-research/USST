""" 
    In this step, we repair the 3D trajectories whose depths are partially (or entirely) missing/invalid, and those contains outliers in tail part.
    For the missing depths, we use the valid hand depth and corresponding frame indices to fit a multinomial with sine model by least-square fitting. 
    And the missing depths are estimated. For move the tail outliers, we use first-order different to identify the cut-off location.
"""
import os
from tqdm import tqdm
import pickle
import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import shutil
    


def read_traj_file(filepath, target='3d'):
    assert os.path.exists(filepath), "File does not exist! {}".format(filepath)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    traj2d = data['traj2d']
    traj3d = None
    if target == '3d':
        assert 'traj3d' in data, "3D trajectories do not exist in file: {}".format(filepath)
        traj3d = data['traj3d']
    return traj2d, traj3d


def repair_depth(traj3d, traj2d, invalids, vis_ratio, vis=False):
    """ repair the depth using least square f"""
    def model_func(p, x):
        f = np.poly1d(p[:-2])
        y = f(x) + p[-2] * np.sin(p[-1]*x)
        return y

    def error_func(p, x, y):
        error = y - model_func(p, x)
        return error
    
    # get the valid true depth
    fit_data, fit_labels = [], []
    for ids, coord in enumerate(traj2d):
        if ids not in invalids:  # valid segments
            u, v = int(coord[0]), int(coord[1])
            fit_data.append(ids)
            fit_labels.append(traj3d[ids, 2])

    # least square fitting by third-order multinomial + sin model
    ret = leastsq(error_func, [0, 0, 1, 0, 1, 0.05], args=(np.array(fit_data), np.array(fit_labels)), epsfcn=0.0001)  # start with straight line: y=0
    
    # repaire
    traj3d_repaire = np.copy(traj3d)
    test_data, test_labels = [], []
    for ids in invalids:
        # repaire 3D coordinates
        u, v = int(traj2d[ids, 0]), int(traj2d[ids, 1])  # (u,v) in (540, 960) frame
        Z = model_func(ret[0], ids)
        test_data.append(ids)
        test_labels.append(Z)
        # (u,v) + Z -> (X, Y)
        X = (u / vis_ratio - 1.94228662e+03) * Z / 1.80820276e+03
        Y = (v / vis_ratio - 1.12382178e+03) * Z / 1.80794556e+03
        # substitute (X, Y, Z)
        traj3d_repaire[ids] = np.array([X, Y, Z])
    
    if vis:
        # plot the fitting 
        plt.figure()
        plt.scatter(fit_data, fit_labels, label='train data')
        x_draw = np.arange(1, len(traj2d) + 1, 0.0001)
        plt.plot(x_draw, model_func(ret[0], x_draw), 'r-', label='model')
        plt.scatter(test_data, test_labels, marker='^', label='test data')
        plt.xlabel('Frame index')
        plt.ylabel('Hand depth (m)')
        plt.legend()
        plt.tight_layout()
    
    return traj3d_repaire


def repair_depth_by_model(traj2d, depth_maps, vis_ratio=0.25):
    """ Replace with the estimated depth """
    traj3d = np.zeros((traj2d.shape[0], 3))
    for ids, (coord, depth) in enumerate(zip(traj2d, depth_maps)):
        u, v = int(coord[0]), int(coord[1])
        Z = depth[v, u] * 0.5
        X = (u / vis_ratio - 1.94228662e+03) * Z / 1.80820276e+03
        Y = (v / vis_ratio - 1.12382178e+03) * Z / 1.80794556e+03
        traj3d[ids] = np.array([X, Y, Z])
    return traj3d



def count_invalid():
    
    total_invalid, total = 0, 0
    for scene_id in sorted(os.listdir(traj_dir)):
        for record_name in sorted(os.listdir(os.path.join(traj_dir, scene_id))):
            
            num_clips_invalid = 0
            all_trajfiles = os.listdir(os.path.join(traj_dir, scene_id, record_name))
            all_trajfiles = list(filter(lambda x: x.endswith('.pkl'), all_trajfiles))
            total += len(all_trajfiles)
            
            for traj_filename in sorted(all_trajfiles):
                # read trajectory file
                traj_file = os.path.join(traj_dir, scene_id, record_name, traj_filename)
                traj2d, traj3d = read_traj_file(traj_file)  # (N,2), (N,3)
                
                invalids = np.where(traj3d[:, 2] <= MIN_DEPTH)[0]
                if len(invalids) >= len(traj3d) - MIN_VALID_POINTS:  # all points are valid
                    num_clips_invalid += 1

            total_invalid += num_clips_invalid
            print("Record:{}, total clips: {}, invalid clips: {}".format(record_name, len(all_trajfiles), num_clips_invalid))
    
    print('Total invalid: {} / {}'.format(total_invalid, total))


def remove_tail_outliers(Zs, num_tails=6, thresh=0.15):
    """ Remove the outlier depth in the tail of a trajectory"""
    num_preserve = len(Zs)
    if len(Zs) >= num_tails:
        tails = Zs[-num_tails:]
        diffs = np.abs(tails[1:] - tails[:-1])
        if np.max(diffs) > thresh:
            num_preserve = len(Zs) - num_tails + np.argmax(diffs) + 1
    return num_preserve


def main():
    
    num_vis = 0
    for scene_id in sorted(os.listdir(traj_dir)):
        for record_name in sorted(os.listdir(os.path.join(traj_dir, scene_id))):
            # repaire results dir 
            traj_repair_path = os.path.join(traj_repair_dir, scene_id, record_name)
            os.makedirs(traj_repair_path, exist_ok=True)

            all_trajfiles = os.listdir(os.path.join(traj_dir, scene_id, record_name))
            all_trajfiles = list(filter(lambda x: x.endswith('.pkl'), all_trajfiles))
            for traj_filename in tqdm(all_trajfiles, desc=f'Record {record_name}', total=len(all_trajfiles)):
                # result file
                traj_repair_file = os.path.join(traj_repair_path, traj_filename)
                
                # read trajectory file
                traj_file = os.path.join(traj_dir, scene_id, record_name, traj_filename)
                traj2d, traj3d = read_traj_file(traj_file)  # (N,2), (N,3)
                
                # trim the trajectory rail part
                num_preserve = remove_tail_outliers(traj3d[:, 2])
                traj2d, traj3d = traj2d[:num_preserve], traj3d[:num_preserve]
                
                invalids = np.where(traj3d[:, 2] <= MIN_DEPTH)[0]
                if len(invalids) == 0:  # all points are valid
                    pass # no further action needed
                
                elif len(invalids) > 0 and len(invalids) <= len(traj3d) - MIN_VALID_POINTS: # there are invalid depths, and at least 5 valid depths
                    # repair invalid depth
                    traj3d = repair_depth(traj3d, traj2d, invalids, vis=VIS)
                    num_vis += 1
                    if VIS and num_vis <= 20:
                        plt.savefig('../output/temp_trim_fit/fit_{}.png'.format(num_vis))
                        plt.close()

                else:  # most points are invalid, we directly discard the trajectory
                    # print("Discard the trajectory: {}/{}/{}".format(scene_id, record_name, traj_filename))
                    continue

                # write results to the pickle file
                with open(traj_repair_file, 'wb') as f:
                    pickle.dump({'traj2d': traj2d, 'traj3d': traj3d, 'num_preserve': num_preserve}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # delete empty trajectory dir
    for scene_id in sorted(os.listdir(traj_repair_dir)):
        for record_name in sorted(os.listdir(os.path.join(traj_repair_dir, scene_id))):
            traj_folder = os.path.join(traj_repair_dir, scene_id, record_name)
            all_trajfiles = list(filter(lambda x: x.endswith('.pkl'), os.listdir(traj_folder)))
            if len(all_trajfiles) == 0:
                shutil.rmtree(traj_folder)
                print("Removed the empty folder: {}".format(traj_folder))


if __name__ == '__main__':
    
    # root path
    root_path = os.path.join(os.path.dirname(__file__), '../../data/EgoPAT3D')
    vis_ratio = 0.25
    MIN_DEPTH = 0.001
    MIN_VALID_POINTS = 10  # the least number of valid 3D points for a valid trajectory
    NUM_TAIL_FRAMES = 6
    VIS=True
    
    # trajectory path
    traj_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'trajectory_warpped')
    assert os.path.exists(traj_dir), 'Path does not exist!'
    
    # video path
    rgb_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'video_clips_hand')
    assert os.path.exists(rgb_dir), 'Path does not exist!'
    
    # repaired trajectory path
    traj_repair_dir = os.path.join(root_path, 'EgoPAT3D-postproc', 'trajectory_repair')
    os.makedirs(traj_repair_dir, exist_ok=True)
    
    # count_invalid()
    
    main()
