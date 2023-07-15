from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import os
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from functools import reduce
from src.utils import denormalize, global_to_local, normalize, vis_frame_traj, write_video,video_to_gif, read_video



class EgoPAT3D(Dataset):
    def __init__(self, root_dir, phase='train', transform=None, data_cfg=None, model_cfg=None):
        self.root_dir = root_dir
        self.phase = phase
        self.transform = transform
        self._MAX_FRAMES = getattr(data_cfg, 'max_frames', -1) # the maximum number of input frames
        self._MAX_DEPTH = getattr(data_cfg, 'max_depth', 3)  # the max depth of hand trajectory
        self.target = getattr(model_cfg, 'target', '3d')
        self.scenes = getattr(data_cfg, 'scenes', None)
        self.modalities = getattr(model_cfg, 'modalities', ['rgb', 'loc'])
        self.tinyset = getattr(data_cfg, 'tinyset', False)
        self.load_all = getattr(data_cfg, 'load_all', False)
        self.use_odom = getattr(model_cfg, 'use_odom', False)
        if self.use_odom: assert self.target == '3d', "Do not support 2D target when using odometry!"
        self.centralize = getattr(model_cfg, 'centralize', False)
        # Our own dataset split
        self.scene_splits = {'train': {'1': ['1', '2', '3', '4', '5', '6', '7'], 
                                       '2': ['1', '2', '3', '4', '5', '6', '7'], 
                                       '3': ['1', '2', '3', '4', '5', '6'],
                                       '4': ['1', '2', '3', '4', '5', '6', '7'],
                                       '5': ['1', '2', '3', '4', '5', '6'], 
                                       '6': ['1', '2', '3', '4', '5', '6'], 
                                       '7': ['1', '2', '3', '4', '5', '6', '7'],
                                       '9': ['1', '2', '3', '4', '5', '6', '7'],
                                       '10': ['1', '2', '3', '4', '5', '6', '7'],
                                       '11': ['1', '2', '3', '4', '5', '6', '7'],
                                       '12': ['1', '2', '3', '4', '5', '6', '7']
                                        },
                             'val': {'1': ['8'], 
                                     '2': ['8'], '3': ['7'], '4': ['8'], '5': ['7'], '6': ['7'], 
                                     '7': ['8'], '9': ['8'], '10': ['8'], '11': ['8'], '12': ['8']
                                     },
                             'test': {'1': ['9', '10'], 
                                      '2': ['9', '10'], '3': ['9', '10'], '4': ['9', '10'], '5': ['8', '9'], '6': ['8', '9'], 
                                      '7': ['9', '10'], '9': ['9', '10'], '10': ['9', '10'], '11': ['9', '10'], '12': ['9', '10']
                                     },
                             'test_novel': {'13': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], 
                                            '14': ['2', '3', '4', '5', '6', '7', '8', '9', '10'], 
                                            '15': ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']}}
        assert self.phase in self.scene_splits, "Invalid dataset split: {}".format(self.phase)
        if self.scenes is not None:
            data = dict(filter(lambda elem: elem[0] in self.scenes, self.scene_splits[self.phase].items()))
            self.scene_splits[self.phase] = data
        # camera intrinsics
        self.intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                           'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                           'w': 3840, 'h': 2160}  # from EgoPAT3D preprocessing code
        self.vis_ratio = 0.25  # the ratio of rgb video size to the 4K size
        
        # the data path needed
        self.rgb_dir, self.traj_dir, self.odom_dir = self._init_data_path()
        if self.tinyset:
            selects = self._read_selected_list()  # 100 selected samples
            samples = selects[:64] if self.phase == 'train' else selects[64:]
            self.rgb_paths, self.traj_data, self.odom_data, self.preserves = self._read_selected_data(samples)
        else:    
            # read dataset lists
            self.rgb_paths, self.traj_data, self.odom_data, self.preserves = self._read_list()

        if self.load_all and 'rgb' in self.modalities:
            self.rgb_data = self._read_all_videos()
        
        
    def _read_traj_file(self, filepath):
        assert os.path.exists(filepath), "File does not exist! {}".format(filepath)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        traj2d = data['traj2d']
        traj3d = data['traj3d']
        num_preserve = data['num_preserve'] if 'num_preserve' in data else len(traj2d)
        return traj2d, traj3d, num_preserve


    def _read_odom_file(self, filepath):
        assert os.path.exists(filepath), "File does not exist! {}".format(filepath)
        all_transforms = np.load(filepath)  # (T, 4, 4)
        return all_transforms

    
    def _init_data_path(self):
        # rgb video root
        rgb_dir = os.path.join(self.root_dir, 'EgoPAT3D-postproc', 'video_clips_hand')
        assert os.path.exists(rgb_dir), 'Path does not exist! {}'.format(rgb_dir)
        # trajectory root
        traj_dir = os.path.join(self.root_dir, 'EgoPAT3D-postproc', 'trajectory_repair')
        assert os.path.exists(traj_dir), 'Path does not exist! {}'.format(traj_dir)
        # visual odometry path (used only if self.use_odom=True)
        odom_dir = os.path.join(self.root_dir, 'EgoPAT3D-postproc', 'odometry')
        assert os.path.exists(odom_dir), 'Path does not exist! {}'.format(traj_dir)
        return rgb_dir, traj_dir, odom_dir


    def _read_selected_list(self):
        select_listfile = os.path.join(self.root_dir, 'EgoPAT3D-postproc', 'selected.txt')
        assert os.path.exists(select_listfile), 'Path does not exist! {}'.format(select_listfile)
        selects = []
        with open(select_listfile, 'r') as f:
            for line in f.readlines():
                selects.append(line.strip())
        return selects
    
    
    def _read_list(self):
        rgb_paths, traj_data, odom_data, preserves = [], [], [], []
        # read the paths
        for scene_id, record_splits in tqdm(self.scene_splits[self.phase].items(), ncols=0, desc='Read trajectories'):
            scene_rgb_path = os.path.join(self.rgb_dir, scene_id)
            scene_traj_path = os.path.join(self.traj_dir, scene_id)
            scene_odom_path = os.path.join(self.odom_dir, scene_id)
            
            record_names = list(filter(lambda x: x.split('_')[-1] in record_splits, os.listdir(scene_traj_path)))  # get the splits of records
            for record in record_names:
                record_rgb_path = os.path.join(scene_rgb_path, record)
                record_traj_path = os.path.join(scene_traj_path, record)
                record_odom_path = os.path.join(scene_odom_path, record)
                
                traj_files = list(filter(lambda x: x.endswith('.pkl'), os.listdir(record_traj_path)))  # a list of '*.pkl'
                for traj_name in traj_files:
                    # read trajectory
                    traj2d, traj3d, num_preserve = self._read_traj_file(os.path.join(record_traj_path, traj_name))
                    traj_data.append({'traj2d': traj2d, 'traj3d': traj3d})
                    # save rgb video path
                    rgb_paths.append(os.path.join(record_rgb_path, traj_name[:-4] + '.mp4'))
                    # read odometry data
                    if self.use_odom:
                        odom = self._read_odom_file(os.path.join(record_odom_path, traj_name[:-4] + '.npy'))  # (T, 4, 4)
                    else:
                        odom = np.eye(4, dtype=np.float32)[None, :, :].repeat(num_preserve, axis=0)  # identity matrix
                    odom_data.append(odom)
                    preserves.append(num_preserve)

        return rgb_paths, traj_data, odom_data, preserves


    def _read_selected_data(self, sample_list):
        rgb_paths, traj_data, odom_data, preserves = [], [], [], []
        # read the paths
        for sample in sample_list:
            scene_id, record, clip_name = sample.split('/')
            # read trajectory
            traj2d, traj3d, num_preserve = self._read_traj_file(os.path.join(self.traj_dir, scene_id, record, clip_name + '.pkl'))
            traj_data.append({'traj2d': traj2d, 'traj3d': traj3d})
            # save rgb video path
            rgb_paths.append(os.path.join(self.rgb_dir, scene_id, record, clip_name + '.mp4'))
            # read odometry data
            if self.use_odom:
                odom = self._read_odom_file(os.path.join(self.odom_dir, scene_id, record, clip_name + '.npy'))  # (T, 4, 4)
            else:
                odom = np.eye(4, dtype=np.float32)[None, :, :].repeat(num_preserve, axis=0)  # identity matrix
            odom_data.append(odom)
            preserves.append(num_preserve)

        return rgb_paths, traj_data, odom_data, preserves
    
    
    def _read_all_videos(self):
        """ Read all video data at once into CPU memory
        """
        video_data = []
        for filename, preserve in zip(self.rgb_paths, self.preserves):
            rgb = self._read_rgb(filename)
            rgb = rgb[:preserve]
            if self.transform:
                rgb = self.transform(rgb)  # (3, T, H=540, W=960)
            else:
                rgb = torch.from_numpy(np.transpose(rgb, (3, 0, 1, 2)))  # (3, T, H=540, W=960)
            video_data.append(rgb)
        return video_data


    def _read_rgb(self, video_file):
        cap = cv2.VideoCapture(video_file)
        success, frame = cap.read()
        videos = []
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            videos.append(frame)
            success, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        videos = np.array(videos)
        return videos
    

    def _XYZ_to_uv(self, traj3d):
        width = self.intrinsics['w'] * self.vis_ratio
        height = self.intrinsics['h'] * self.vis_ratio
        # transform the (X, Y, Z) into the (u,v) by PINHOLE camera model
        u = (traj3d[:, 0] * self.intrinsics['fx'] / traj3d[:, 2] + self.intrinsics['ox']) * self.vis_ratio
        v = (traj3d[:, 1] * self.intrinsics['fy'] / traj3d[:, 2] + self.intrinsics['oy']) * self.vis_ratio
        u = torch.clamp(u, min=0, max=width-1)
        v = torch.clamp(v, min=0, max=height-1)
        traj2d = torch.stack((u, v), dim=-1)
        return traj2d
    
    
    def _normalize(self, traj):
        traj_new = traj.clone()
        width = self.intrinsics['w'] * self.vis_ratio
        height = self.intrinsics['h'] * self.vis_ratio
        if self.target == '2d':
            # normalize the (u,v) into the range 0-1
            traj_new[:, 0] /= width
            traj_new[:, 1] /= height
        elif self.target == '3d':
            # transform the (X, Y, Z) into the normalized (u,v,z) by PINHOLE camera model
            traj2d = self._XYZ_to_uv(traj)
            traj_new[:, 0] = traj2d[:, 0] / width
            traj_new[:, 1] = traj2d[:, 1] / height
            traj_new[:, 2] = traj[:, 2] / self._MAX_DEPTH
        if self.centralize:
            traj_new -= 0.5
        return traj_new

    
    def _get_projected_traj3d(self, traj3d, odometry):
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
    
    
    def __len__(self):
        return len(self.rgb_paths)
    
    
    def __getitem__(self, index):
        
        traj2d = torch.from_numpy(self.traj_data[index]['traj2d']).to(torch.float32)  # (T, 2)
        traj3d = torch.from_numpy(self.traj_data[index]['traj3d']).to(torch.float32)  # (T, 3)
        traj = traj2d if self.target == '2d' else traj3d
        num_preserve = self.preserves[index]
        if 'rgb' in self.modalities:
            if self.load_all:
                rgb = self.rgb_data[index]  # (3, T, H=540, W=960)
            else:
                # read mp4 file
                rgb = self._read_rgb(self.rgb_paths[index])  # (T, H=540, W=960, 3)
                rgb = rgb[:num_preserve]
                # ------------  process video input  ------------
                # data transformation
                if self.transform:
                    rgb = self.transform(rgb)  # (3, T, H=540, W=960)
                else:
                    rgb = torch.from_numpy(np.transpose(rgb, (3, 0, 1, 2)))  # (3, T, H=540, W=960)
            assert rgb.shape[1] == traj.shape[0], "Trajectory is inconsistent with RGB video!"
            
        # clipping or padding with zero
        max_frames = traj.size(0) if self._MAX_FRAMES < 0 else self._MAX_FRAMES
        len_valid = min(traj.size(0), max_frames)
        
        input_data = torch.tensor([])  # a placeholder (not used)
        if 'rgb' in self.modalities:
            input_data = torch.zeros([rgb.size(0), max_frames, rgb.size(2), rgb.size(3)], dtype=rgb.dtype)
            input_data[:, :len_valid] = rgb[:, :len_valid]
        
        # ------------  process target trajectory output  ------------ 
        len_pad, len_real = max_frames, len_valid
        odometry = torch.eye(4)[None, :, :].repeat(len_pad, 1, 1).to(torch.float32)
        output_traj = torch.zeros([len_pad, traj.size(1)], dtype=traj.dtype)
        traj_valid = traj[:len_valid]  # (T, 3) or (T, 2): [3D or 2D local trajectory]
        if self.use_odom:
            odom = self.odom_data[index][:num_preserve]  # (T, 4, 4)
            odom_valid = odom[:len_valid]
            odometry[:len_valid] = torch.from_numpy(odom_valid)
            # project all future hand locations to each of current frames
            traj3d_valid_all = self._get_projected_traj3d(traj3d[:len_valid].numpy(), odom_valid)  # (T*(T+1)/2, 3)
            traj3d_valid = torch.from_numpy(traj3d_valid_all[:len_real])  # only use the trajectory in the first frame
            traj_valid = self._XYZ_to_uv(traj3d_valid) if self.target == '2d' else traj3d_valid  # (T, 3) or (T, 2): [3D or 2D global trajectory]
        
        # transformed to normalized 2d representation
        traj_norm_valid = self._normalize(traj_valid)
        output_traj[:len_real] = traj_norm_valid

        return self.rgb_paths[index], input_data, odometry, len_valid, output_traj



def build_dataloaders(cfg, phase='trainval'):
    """Loading the dataset"""
    from .video_transforms.video_transforms import Compose, Resize, Normalize
    from .video_transforms.volume_transforms import ClipToTensor
    
    data_root = os.path.join(cfg.DATA.data_path, cfg.DATA.dataset)
    transform_train, transform_test = None, None
    if cfg.DATA.transform is not None:
        input_size = cfg.DATA.transform.input_size
        transform_train = Compose([Resize(input_size), ClipToTensor(), Normalize(mean=cfg.DATA.transform.means, std=cfg.DATA.transform.stds)])
        transform_test = Compose([Resize(input_size), ClipToTensor(), Normalize(mean=cfg.DATA.transform.means, std=cfg.DATA.transform.stds)])
    
    if phase == 'trainval':
        # train set
        trainset = EgoPAT3D(data_root, phase='train', transform=transform_train, data_cfg=cfg.DATA, model_cfg=cfg.MODEL)
        train_loader = DataLoader(trainset, batch_size=cfg.TRAIN.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        # validation set
        valset = EgoPAT3D(data_root, phase='val', transform=transform_test, data_cfg=cfg.DATA, model_cfg=cfg.MODEL)
        val_loader = DataLoader(valset, batch_size=cfg.TEST.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        # print
        print("Number of train/val: {}/{}".format(len(trainset), len(valset)))
        return train_loader, val_loader
    else:
        # test set
        testset = EgoPAT3D(data_root, phase='test', transform=transform_test, data_cfg=cfg.DATA, model_cfg=cfg.MODEL)
        test_loader = DataLoader(testset, batch_size=cfg.TEST.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        # test_novel set
        testnovel_set = EgoPAT3D(data_root, phase='test_novel', transform=transform_test, data_cfg=cfg.DATA, model_cfg=cfg.MODEL)
        testnovel_loader = DataLoader(testnovel_set, batch_size=cfg.TEST.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        # print
        print("Number of test/test_novel samples: {}/{}".format(len(testset), len(testnovel_set)))
        return test_loader, testnovel_loader


    
def vis_input_output(train_loader, trainset, target='3d', use_odom=False, save_gif=True):

    result_dir = '../output/selected_samples/'
    os.makedirs(result_dir, exist_ok=True)
    vis_dir = os.path.join(result_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    filelist_fid = open(os.path.join(result_dir, 'filelist.txt'), 'w')
    
    for i, (filename, clip, odometry, nf, traj) in tqdm(enumerate(train_loader), total=len(trainset), desc='Train'):
        # get file info
        sample_info = filename[0][:-4].split('/')
        vis_filename = 'vis_{}_{}'.format(sample_info[-2], sample_info[-1].split('_')[0])

        # read video by filename
        video = read_video(filename[0])
        video_size = video.shape[1:3]  # (H, W)

        if trainset.centralize:
            traj += 0.5
        u = torch.floor(traj[0, :, 0] * video_size[1]).to(torch.long)  # (T,) or (T*(T+1)/2,)
        v = torch.floor(traj[0, :, 1] * video_size[0]).to(torch.long)  # (T,) or (T*(T+1)/2,)

        if use_odom:
            frame = cv2.cvtColor(video[0], cv2.COLOR_RGB2BGR)  # use the first frame
            traj2d = torch.hstack((u[:, None], v[:, None])).numpy()
            vis_frame_traj(frame, traj2d)
            vis_file = os.path.join(vis_dir, vis_filename + '_global.png')
            cv2.imwrite(vis_file, frame)
        
        # visualize
        num_frames = int(nf[0])
        vis_mat = []
        for t in range(num_frames):
            frame = video[t]
            if use_odom:
                # back-project future hand locations of the first frame to current frame
                if t > 0:
                    # normalized 3D to actual 3D in camera_0
                    traj3d = denormalize(traj[0, t:], target='3d', intrinsics=trainset.intrinsics, max_depth=trainset._MAX_DEPTH)
                    # camera_0 to camera_t (Note: all T-t points are in the same camera_0 ,thus their odometry are ignored by setting to identity matrix)
                    odom_new = np.copy(odometry[0].numpy())
                    odom_new[(t+1):] = np.eye(4)[None, :, :]
                    traj3d = global_to_local(traj3d, odom_new, start_idx=t+1)
                    # in camera_t: 3D to normalized 2D
                    traj2d = normalize(traj3d, target='3d', intrinsics=trainset.intrinsics, vis_ratio=trainset.vis_ratio, max_depth=trainset._MAX_DEPTH)
                    # normalized 2D to (u,v)
                    u = np.floor(traj2d[:, 0] * video_size[1]).astype(np.int32)
                    v = np.floor(traj2d[:, 1] * video_size[0]).astype(np.int32)
                    traj2d = np.stack((u, v), axis=1)
                else:
                    traj2d = torch.hstack((u[:, None], v[:, None])).numpy()
                vis_frame_traj(frame, traj2d)
            else:
                cv2.circle(frame, (int(u[t]), int(v[t])), radius=5, color=(0, 255, 0), thickness=-1)
            vis_mat.append(frame)
        vis_mat = np.array(vis_mat)  # (T, 224, 224, 3)
        if save_gif:
            video_to_gif(vis_mat, os.path.join(vis_dir, vis_filename) + '.gif', fps=5)
        else:
            write_video(vis_mat, os.path.join(vis_dir, vis_filename) + '.mp4', fps=5)
        
        filelist_fid.writelines('{}/{}/{}\n'.format(sample_info[-3], sample_info[-2], sample_info[-1]))
    filelist_fid.close()


def get_frame_stats():
    
    num_frames = []
    for i, (filename, clip, odometry, nf, traj) in tqdm(enumerate(train_loader), total=len(trainset), desc='Train'):
        num_frames.append(int(nf[0]))
    
    for i, (filename, clip, odometry, nf, traj) in tqdm(enumerate(val_loader), total=len(valset), desc='Val'):
        num_frames.append(int(nf[0]))
    
    print("# frames: max={}, min={}, mean={}".format(max(num_frames), min(num_frames), np.mean(num_frames)))
    plt.hist(num_frames, bins=100)
    plt.savefig('../data/EgoPAT3D/num_frames_tinyset.png')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from video_transforms.video_transforms import Compose, Resize, Normalize
    from video_transforms.volume_transforms import ClipToTensor
    import sys

    rgb_transform = Compose([Resize([64, 64]), ClipToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    class data_cfg: pass
    data_cfg.scenes = ['1']
    data_cfg.tinyset = True
    data_cfg.load_all = True

    class model_cfg: pass
    model_cfg.target = '3d'
    model_cfg.use_odom = True
    
    trainset = EgoPAT3D('../data/EgoPAT3D', phase='train', transform=rgb_transform, data_cfg=data_cfg, model_cfg=model_cfg)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    valset = EgoPAT3D('../data/EgoPAT3D', phase='val', transform=rgb_transform, data_cfg=data_cfg, model_cfg=model_cfg)
    val_loader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)

    print("Number of train/val samples: {}/{}".format(len(trainset), len(valset)))
    
    if sys.argv[1] == 'vis':
        # visualize data from DataLoader
        vis_input_output(train_loader, trainset, target=model_cfg.target, use_odom=model_cfg.use_odom)
    
    elif sys.argv[1] == 'stat':
        # get the stats of the number frames for all clips
        get_frame_stats()

    elif sys.argv[1] == 'check_val':
        for i, (clip, nframes, traj_gt) in tqdm(enumerate(val_loader),
                total=len(val_loader), ncols=0, desc='check valset'):
            if i == 0:
                print(clip.size(), traj_gt.size())  #  (B, 3, T, H, W),  (B, T, 3)
    else:
        raise NotImplementedError

    print("Done!")