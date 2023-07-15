import numpy as np
import os
import torch
import importlib
from src.config import parse_configs
from src.EgoPAT3DLoader import build_dataloaders
from src.utils import set_deterministic, output_transform, read_video, get_test_ratios, video_to_gif, denormalize
from src.utils_vis import vis_traj3d
from src.utils_io import load_checkpoint
from tqdm import tqdm
import cv2
import pickle
from functools import reduce
from copy import deepcopy



def overlay_video(vis2d, vis3d=None):
    if vis3d is None:
        return vis2d
    video_vis = np.copy(vis2d)
    for i, (frame2d, frame3d) in enumerate(zip(vis2d, vis3d)):
        # overlay
        bottom_left = frame2d[frame2d.shape[0] - frame3d.shape[0]:, :frame3d.shape[1], :]
        frame2d[frame2d.shape[0] - frame3d.shape[0]:, :frame3d.shape[1], :] = cv2.addWeighted(bottom_left, 0.4, frame3d, 0.6, 0)  # bottom left
        video_vis[i] = frame2d
    return video_vis


def overlay_frame(frame2d, frame3d):
    frame = np.copy(frame2d)
    bottom_left = frame2d[frame2d.shape[0] - frame3d.shape[0]:, :frame3d.shape[1], :]
    frame[frame2d.shape[0] - frame3d.shape[0]:, :frame3d.shape[1], :] = cv2.addWeighted(bottom_left, 0.4, frame3d, 0.6, 0)  # bottom left
    return frame


def compute_mse(outputs, traj_gt, num_full, future=False):
    mse = 0
    for i, (r, preds) in enumerate(outputs.items()):
        num_obs = torch.floor(num_full * float(r)).to(torch.long).numpy()
        gts = traj_gt[num_obs: num_full] if future else traj_gt[:num_obs]
        mse += np.mean(np.sqrt(np.sum((preds['traj'] - gts)**2, axis=-1)))
    mse /= (i+1)
    return float(mse)




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


def vis_frame_traj(frame, traj2d, traj_start=0, future_start=15, 
                   TRAJ_COLOR=(255, 255, 0), PAST_COLOR=(255, 0, 0), FUTURE_COLOR=(0, 0, 255)):
    cv2.polylines(frame, [traj2d], False, TRAJ_COLOR, thickness=2)
    for t, uv in enumerate(traj2d):
        color = PAST_COLOR if traj_start + t <= future_start else FUTURE_COLOR
        cv2.circle(frame, uv, radius=5, color=color, thickness=-1)


def visualize_transformed_traj(video, traj3d, odometry, intrinsics, ratio=0.5,
                               TRAJ_COLOR=(255, 255, 0), PAST_COLOR=(255, 0, 0), FUTURE_COLOR=(0, 0, 255)):
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
        vis_frame_traj(video[i], traj2d, traj_start=i, future_start=future_start, 
                       TRAJ_COLOR=TRAJ_COLOR, PAST_COLOR=PAST_COLOR, FUTURE_COLOR=FUTURE_COLOR)


def run_demo_test(cfg, model, data_loader, candidates=[]):

    all_files, all_nframes, all_odoms = [], [], []
    local_results = {'outputs': [], 'gt_traj': [], 'mse': []}
    global_results = {'outputs': [], 'gt_traj': [], 'mse': []}
    for batch_id, batch_data in tqdm(enumerate(data_loader), total=len(data_loader), desc='Run testing'):
        
        filename, clip, odometry, nframes, traj_gt = batch_data
        filename, input_data, odometry, nframes, gt_traj = filename[0], clip[0:1], odometry[0:1], nframes[0:1], traj_gt[0:1]
        
        # filtering
        scene_record = filename[:-4].split('/')[-2]
        clip_name = filename[:-4].split('/')[-1].split('_')[0]
        if '{}_{}'.format(scene_record, clip_name) not in candidates:
            continue
        
        # inference
        with torch.no_grad():
            ratios = get_test_ratios(cfg.TEST.ratios, nframes)
            outputs = model.inference(input_data.cuda(), nframes, ratios, traj=gt_traj.cuda())
        
        # transform from global to local coordinate frame (for 2D visualize)
        outputs_local, gt_traj_local = output_transform(outputs, gt_traj.numpy(), nframes.numpy(), odometry.numpy(), 
                                            intrinsics=intrinsics, vis_ratio=vis_ratio, max_depth=max_depth, 
                                            centralize=model.centralize, use_odom=True, ignore_depth=model.ignore_depth)

        outputs_global, gt_traj_global = output_transform(outputs, gt_traj.numpy(), nframes.numpy(), odometry.numpy(), 
                                        intrinsics=intrinsics, vis_ratio=vis_ratio, max_depth=max_depth, 
                                        centralize=model.centralize, use_odom=False, ignore_depth=model.ignore_depth)
        
        mse_global = compute_mse(outputs_global[0][0], gt_traj_global[0], nframes[0], future=True)
        mse_local = compute_mse(outputs_local[0][0], gt_traj_local[0], nframes[0], future=True)
        
        all_files.append(filename)
        all_nframes.append(nframes.numpy())
        all_odoms.append(odometry.numpy())
        local_results['outputs'].append(outputs_local)
        local_results['gt_traj'].append(gt_traj_local)
        local_results['mse'].append(mse_local)
        global_results['outputs'].append(outputs_global)
        global_results['gt_traj'].append(gt_traj_global)
        global_results['mse'].append(mse_global)
        
    return all_files, all_nframes, all_odoms, local_results, global_results


def scale_intrinsics():
    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}  # from EgoPAT3D preprocessing code
    vis_ratio = 0.25
    max_depth = 3
    for k, v in intrinsics.items():
        v *= vis_ratio  # scaling the camera intrinsics
        intrinsics[k] = v
    return 

def vis_2d(video, outputs, trajectories, nframes, odoms, obs_ratios=[0.6]):
    outputs_unobs, outputs_obs = outputs
    vis_video = {}
    for r in obs_ratios:
        num_full = int(nframes[0])  # only one video
        num_obs = int(num_full * float(r))
        
        traj_gt_norm = trajectories[0, :num_full]  # (Tm,3)
        vis_video[r] = np.copy(video[:num_full])
        odometry = odoms[0, :num_full]
        
        # visualize ground truth 2D trajectory
        traj_gt = denormalize(traj_gt_norm, target='3d', intrinsics=intrinsics, max_depth=max_depth)
        visualize_transformed_traj(vis_video[r], traj_gt, odometry, scaled_intrinsics, ratio=r,
                                   TRAJ_COLOR=(255, 255, 0), PAST_COLOR=(0, 255, 0), FUTURE_COLOR=(0, 255, 0))  # green -> green
        
        # visualize predicted 2D trajectory
        # preds_obs = outputs_obs[0][str(r)]['traj'] if outputs_obs is not None else np.copy(traj_gt_norm[:num_obs])
        preds_obs = np.copy(traj_gt_norm[:num_obs])
        preds_norm = np.concatenate((preds_obs, outputs_unobs[0][str(r)]['traj']))
        preds = denormalize(preds_norm, target='3d', intrinsics=intrinsics, max_depth=max_depth)
        visualize_transformed_traj(vis_video[r], preds, odometry, scaled_intrinsics, ratio=r, 
                                   TRAJ_COLOR=(0, 255, 255), PAST_COLOR=(0, 0, 255), FUTURE_COLOR=(255, 0, 0))  # blue -> red
    return vis_video


def visualize(all_files, all_nframes, all_odoms, local_results, global_results, result_path, target='3d', saveframe=False, saveGIF=False):
    
    for filename, nframes, odoms, outputs_local, gt_traj_local, outputs_global, gt_traj_global in zip(all_files, all_nframes, all_odoms,
                                                                                             local_results['outputs'], local_results['gt_traj'],
                                                                                             global_results['outputs'], global_results['gt_traj'],):
        scene_record = filename[:-4].split('/')[-2]
        clip_name = filename[:-4].split('/')[-1].split('_')[0]
        # directly read the original video
        video = read_video(filename, toRGB=False)
        
        # visualize 2D
        vis_video = vis_2d(video, outputs_local, gt_traj_local, nframes, odoms, obs_ratios=cfg.TEST.ratios)
        # visualize 3D
        vis3d_video = {r: None for r in cfg.TEST.ratios}
        if target == '3d':
            canvas_size = (intrinsics['h'] * vis_ratio * 0.5, intrinsics['w'] * vis_ratio * 0.5)
            vis3d_video = vis_traj3d(outputs_global, gt_traj_global, cfg.TEST.ratios, nframes, canvas_size, result_prefix=None, return_video=True)
        
        # save gif/png
        for r in cfg.TEST.ratios:

            if saveGIF:
                gif_dir = os.path.join(result_path, 'gif_{}'.format(r))
                os.makedirs(gif_dir, exist_ok=True)
                result_prefix = os.path.join(gif_dir, '{}_{}'.format(scene_record, clip_name))
                # vis
                vis = overlay_video(vis_video[r], vis3d=vis3d_video[r])
                video_to_gif(vis, result_prefix + '.gif')

            if saveframe:
                img_dir = os.path.join(result_path, 'img_{}'.format(r))
                os.makedirs(img_dir, exist_ok=True)
                result_prefix = os.path.join(img_dir, '{}_{}'.format(scene_record, clip_name))
                # vis
                vis = overlay_frame(vis_video[r][0], vis3d_video[r][-1])
                vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
                cv2.imwrite(result_prefix + '.png', vis)


def run_visualization(model, test_loader, result_path, candidates):
    os.makedirs(result_path, exist_ok=True)
    
    cache_file = os.path.join(result_path, 'results.pkl')
    if not os.path.exists(cache_file):
        # run test inference
        all_files, all_nframes, all_odoms, local_results, global_results = run_demo_test(cfg, model, test_loader, candidates)
        with open(cache_file, 'wb') as f:
            pickle.dump([all_files, all_nframes, all_odoms, local_results, global_results], f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(cache_file, 'rb') as f:
            print('Loading the cached results: {}'.format(cache_file))
            all_files, all_nframes, all_odoms, local_results, global_results = pickle.load(f)
        # filtering
        inds = [i for i, filename in enumerate(all_files) if '{}_{}'.format(filename[:-4].split('/')[-2], 
                                                                            filename[:-4].split('/')[-1].split('_')[0]) in candidates]
        all_files = [all_files[i] for i in inds]
        all_nframes = [all_nframes[i] for i in inds]
        all_odoms = [all_odoms[i] for i in inds]
        local_results = {k: [v[i] for i in inds] for k, v in local_results.items()}
        global_results = {k: [v[i] for i in inds] for k, v in global_results.items()}

    # visualization
    visualize(all_files, all_nframes, all_odoms, local_results, global_results, result_path, model.target, saveframe=False, saveGIF=True)
    


def demo():
    
    # build test dataloaders
    test_loader, testnovel_loader = build_dataloaders(cfg, phase='test')

    # build the model
    model_module = importlib.import_module('src.models.{}'.format(cfg.MODEL.arch))
    model = getattr(model_module, cfg.MODEL.arch)(cfg.MODEL, seq_len=cfg.DATA.max_frames, input_size=cfg.DATA.transform.input_size[0])
    model = model.to(device=cfg.device)
    # load checkpoints
    model, test_epoch = load_checkpoint(cfg, model)
    model = model.eval()
    
    # run on see test data
    run_visualization(model, test_loader, 
                      os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'demo_paper', 'seen'),
                      candidates=['microwave_9_c100', 'pantryShelf_9_c36'])
    
    # run on unsee test data
    run_visualization(model, testnovel_loader, 
                      os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'demo_paper', 'unseen'),
                      candidates=['stoveTop_10_c100', 'windowsillAC_8_c29'])


if __name__ == '__main__':
    # parse input arguments
    cfg = parse_configs(phase='test')
    cfg.TEST.batch_size = 1
    cfg.DATA.load_all = False

    # fix random seed 
    set_deterministic(cfg.seed)
    
    # constant camera parameters
    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}  # from EgoPAT3D preprocessing code
    max_depth = 3
    vis_ratio = 0.25
    
    scaled_intrinsics = deepcopy(intrinsics)
    for k, v in intrinsics.items():
        v *= vis_ratio  # scaling the camera intrinsics
        scaled_intrinsics[k] = v
    
    # test
    demo()