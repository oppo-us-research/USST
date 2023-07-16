import os
import argparse
import yaml
from easydict import EasyDict
import numpy as np
import cv2
import pickle
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.video_transforms.video_transforms import Compose, Resize, Normalize
from src.video_transforms.volume_transforms import ClipToTensor
from src.utils import set_deterministic, output_transform, read_video
from src.utils_vis import vis_demo, vis_traj3d, video_to_gif
import importlib
import random
from functools import reduce
from src.utils_io import load_checkpoint



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


def read_odometry(filepath):
    assert os.path.exists(filepath), "File does not exist! {}".format(filepath)
    all_transforms = np.load(filepath)  # (T, 4, 4)
    return all_transforms


def get_projected_traj3d(traj3d, odometry):
    """ traj3d: (T, 3)
        odometry: (T, 4, 4)
    """
    length = len(odometry)
    traj3d_homo = np.hstack((traj3d, np.ones((length, 1))))  # (T, 4)
    all_traj3d_proj = []
    for i in range(length):
        # compute the transformed points of all future points
        traj3d_proj = [traj3d[i]]  # initial 3D point
        for j in range(i + 1, length):
            odom = reduce(np.dot, odometry[(i+1):(j+1)])  # (4, 4)
            future_point = odom.dot(traj3d_homo[j].T)  # (4,)
            traj3d_proj.append(future_point[:3])
        all_traj3d_proj.append(np.array(traj3d_proj))
    all_traj3d_proj = np.concatenate(all_traj3d_proj, axis=0)  # (T*(T+1)/2, 3)
    return all_traj3d_proj


def _XYZ_to_uv(traj3d, intrinsics=None, vis_ratio=0.25):
    width = intrinsics['w'] * vis_ratio
    height = intrinsics['h'] * vis_ratio
    # transform the (X, Y, Z) into the (u,v) by PINHOLE camera model
    u = (traj3d[:, 0] * intrinsics['fx'] / traj3d[:, 2] + intrinsics['ox']) * vis_ratio
    v = (traj3d[:, 1] * intrinsics['fy'] / traj3d[:, 2] + intrinsics['oy']) * vis_ratio
    u = np.clip(u, a_min=0, a_max=width-1)
    v = np.clip(v, a_min=0, a_max=height-1)
    traj2d = torch.stack((u, v), dim=-1)
    return traj2d


def _normalize(traj, intrinsics=None, vis_ratio=0.25, max_depth=3.0, target='3d', centralize=False):
    traj_new = traj.clone()
    width = intrinsics['w'] * vis_ratio
    height = intrinsics['h'] * vis_ratio
    if target == '2d':
        traj_new[:, 0] = traj[:, 0] / width
        traj_new[:, 1] = traj[:, 1] / height
    elif target == '3d':
        # transform the (X, Y, Z) into the normalized (u,v,z) by PINHOLE camera model
        traj2d = _XYZ_to_uv(traj, intrinsics, vis_ratio)
        traj_new[:, 0] = traj2d[:, 0] / width
        traj_new[:, 1] = traj2d[:, 1] / height
        traj_new[:, 2] = traj[:, 2] / max_depth
    if centralize:
        traj_new -= 0.5
    return traj_new


def random_select_data(cfg):
    ### randomly select one clip as an example
    data_root = os.path.join(cfg.DATA.data_path, cfg.DATA.dataset)
    scene_test_splits = {'1': ['9', '10'], '2': ['9', '10'], '3': ['9', '10'], '4': ['9', '10'], '5': ['8', '9'], '6': ['8', '9'], 
                            '7': ['9', '10'], '9': ['9', '10'], '10': ['9', '10'], '11': ['9', '10'], '12': ['9', '10']}
    if hasattr(cfg.DATA, 'scenes'):
        scene_test_splits = dict(filter(lambda elem: elem[0] in cfg.DATA.scenes, scene_test_splits.items()))
    # random select a scene
    scene_id = random.choice(list(scene_test_splits))
    # random select a record
    record_id = random.choice(scene_test_splits[scene_id])
    record_path = os.path.join(data_root, 'EgoPAT3D-postproc', 'trajectory_repair', scene_id)
    record_name = list(filter(lambda x: x.split('_')[-1] == record_id, os.listdir(record_path)))[0]
    # random select a clip
    clip_path = os.path.join(record_path, record_name)
    all_clips = list(filter(lambda x: x.endswith('.pkl'), os.listdir(clip_path)))
    clip_name = random.choice(all_clips)[:-4]
    return {'scene': scene_id, 'record': record_name, 'clip': clip_name}


def prepare_demo_data(example, cfg, intrinsics=None, vis_ratio=0.25, max_depth=3.0):
    
    data_root = os.path.join(cfg.DATA.data_path, cfg.DATA.dataset)
    path_template = os.path.join(data_root, 'EgoPAT3D-postproc', '{}', example['scene'], example['record'], example['clip'] + '.{}')

    # prepare ground truth
    traj_file = path_template.format('trajectory_repair', 'pkl')
    traj2d, traj3d, num_preserve = read_traj_file(traj_file, target=cfg.MODEL.target)
    traj2d = torch.from_numpy(traj2d).to(torch.float32)
    traj3d = torch.from_numpy(traj3d).to(torch.float32)
    traj = traj3d if cfg.MODEL.target == '3d' else traj2d
    
    # read video
    video_file = path_template.format('video_clips_hand', 'mp4')
    video = read_video(video_file)  # (T, H, W, C), RGB data
    video = video[:num_preserve]

    # read odometry
    odom_data = np.eye(4, dtype=np.float32)[None, :, :].repeat(num_preserve, axis=0)  # identity matrix
    use_odom = getattr(cfg.MODEL, 'use_odom', False)
    if use_odom:
        odom_file = path_template.format('odometry', 'npy')
        odom_data = read_odometry(odom_file)
        odom_data = odom_data[:num_preserve]  # (T, 4, 4)
    
    # preprocessing
    transform_test = Compose([Resize(cfg.DATA.transform.input_size), 
                              ClipToTensor(), 
                              Normalize(mean=cfg.DATA.transform.means, std=cfg.DATA.transform.stds)])
    input_data = transform_test(video)   # (C, T, h, w)
    max_frames = traj.size(0) if cfg.DATA.max_frames < 0 else cfg.DATA.max_frames
    len_valid = min(input_data.size(1), max_frames)

    input_data = torch.zeros([input_data.size(0), max_frames, input_data.size(2), input_data.size(3)], dtype=input_data.dtype)
    input_data[:, :len_valid] = input_data[:, :len_valid]
    input_data = input_data.unsqueeze(0)
    
    nframes = torch.tensor(len_valid).unsqueeze(0)
    
    odometry = torch.eye(4)[None, :, :].repeat(max_frames, 1, 1).to(torch.float32)
    gt_traj = torch.zeros([max_frames, traj.size(1)], dtype=traj.dtype)
    traj_valid = traj[:len_valid]  # (T, 3)
    if use_odom:
        odom_valid = odom_data[:len_valid]
        odometry[:len_valid] = torch.from_numpy(odom_valid)
        # project all future hand locations to each of current frames
        traj3d_valid_all = get_projected_traj3d(traj3d[:len_valid].numpy(), odom_valid)  # (T*(T+1)/2, 3)
        traj3d_valid = torch.from_numpy(traj3d_valid_all[:len_valid])  # only use the trajectory in the first frame
        traj_valid = _XYZ_to_uv(traj3d_valid, intrinsics, vis_ratio) if cfg.MODEL.target == '2d' else traj3d_valid

    traj_valid = _normalize(traj_valid, intrinsics=intrinsics, vis_ratio=vis_ratio, max_depth=max_depth,
                target=cfg.MODEL.target, centralize=getattr(cfg.MODEL, 'centralize', False))
    gt_traj[:len_valid] = traj_valid
    gt_traj = gt_traj.unsqueeze(0)
    odometry = odometry.unsqueeze(0)
    
    return video, input_data, odometry, nframes, gt_traj


def run_demo(cfg, sample=None, ratios=None):

    # result path
    result_path = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'demo')
    os.makedirs(result_path, exist_ok=True)

    # constant camera parameters
    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}  # from EgoPAT3D preprocessing code
    vis_ratio = 0.25
    max_depth = 3

    # input data
    if sample is None:
        sample = random_select_data(cfg=cfg)
    video, input_data, odometry, nframes, gt_traj = prepare_demo_data(sample, cfg, intrinsics=intrinsics, vis_ratio=vis_ratio, max_depth=max_depth)
    
    # input observation ratios
    if ratios is None:
        ratios = cfg.TEST.ratios
    test_ratios = torch.tensor([[float(r) for r in ratios]]).repeat(nframes.size(0), 1)  # (B, M)
    
    # load model
    model_module = importlib.import_module('src.models.{}'.format(cfg.MODEL.arch))
    model = getattr(model_module, cfg.MODEL.arch)(cfg.MODEL, seq_len=cfg.DATA.max_frames, input_size=cfg.DATA.transform.input_size[0])
    model = model.to(device=cfg.device)
    model, epoch = load_checkpoint(cfg, model)
    model = model.eval()

    # inference
    with torch.no_grad():
        outputs = model.inference(input_data.cuda(), nframes, test_ratios, traj=gt_traj.cuda())
    
    # transform from global to local coordinate frame
    outputs, gt_traj = output_transform(outputs, gt_traj.numpy(), nframes.numpy(), odometry.numpy(), 
                                        intrinsics=intrinsics, vis_ratio=vis_ratio, max_depth=max_depth, 
                                        centralize=model.centralize, use_odom=model.use_odom, ignore_depth=model.ignore_depth)
    # visualize
    prefix_name = '{}_{}_{}_epoch{}_'.format(sample['scene'], sample['record'], sample['clip'].split('_')[0], epoch)
    result_prefix = os.path.join(result_path, prefix_name)
    vis_demo(video, outputs, nframes, gt_traj, result_prefix, obs_ratios=ratios)

    # visualize 3D trajectory
    result_3d_path = os.path.join(result_path, 'traj3d')
    os.makedirs(result_3d_path, exist_ok=True)

    canvas_size = (intrinsics['h'] * vis_ratio * 0.5, intrinsics['w'] * vis_ratio * 0.5)
    result_prefix = os.path.join(result_3d_path, prefix_name)
    vis_traj3d(outputs, gt_traj, ratios, nframes, canvas_size, result_prefix)



def recover_video(input_data, cfg, video_size=(540, 960)):
    """ input_data: (C, T, H, W)
    """
    means = np.array([[cfg.DATA.transform.means]])  # (1, 1, 3)
    stds = np.array([[cfg.DATA.transform.stds]])  # (1, 1, 3)
    num_frames = input_data.size(1)
    video = []
    for t in range(num_frames):
        frame = input_data[:, t].permute(1, 2, 0).contiguous().numpy()  # (H, W, 3)
        frame = (frame * stds + means) * 255  # broadcast add & mul 
        frame = cv2.resize(frame, dsize=(video_size[1], video_size[0]))
        frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
        video.append(frame)
    video = np.array(video, dtype=np.uint8)  # (T, 224, 224, 3)
    return video
    

def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/baseline.yml',
                        help='The relative path of dataset.')
    parser.add_argument('--gpus', type=str, default="0", 
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='The number of workers to load dataset. Default: 4')
    parser.add_argument('--tag', type=str, default='default',
                        help='The tag to save model results')
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        cfg = EasyDict(yaml.safe_load(f))
    cfg.update(vars(args))
    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg.update(device=device)
    cfg.MODEL.update(device=device)
    # add root path
    root_path = os.path.dirname(__file__)
    cfg.update(root_path=root_path)

    cfg.DATA.load_all = False

    return cfg
    

if __name__ == '__main__':
    
    # parse input arguments
    cfg = parse_configs()
    
    # fix random seed 
    set_deterministic(cfg.seed)
    
    # demo_sample = {'scene': '1', 'record': 'bathroomCabinet_9', 'clip': 'c1_s64_e103_sx718_sy420_ex631_ey247'}
    demo_sample = {'scene': '1', 'record': 'bathroomCabinet_8', 'clip': 'c100_s5336_e5369_sx586_sy367_ex518_ey346'}
    run_demo(cfg, sample=demo_sample)