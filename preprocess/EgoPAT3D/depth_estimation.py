import os, sys, cv2
import numpy as np
import torch
from torch.autograd import Variable



def init_depth_model():
    NeWCRFs_ROOT = os.path.join(os.path.dirname(__file__), '../../3rdparty/NeWCRFs')
    sys.path.append(os.path.join(NeWCRFs_ROOT, 'newcrfs'))
    from networks.NewCRFDepth import NewCRFDepth

    # model config
    class cfg: pass
    cfg.version = 'large07'
    cfg.checkpoint_path = os.path.join(NeWCRFs_ROOT, 'model_zoo', 'model_nyu.ckpt')
    cfg.inv_depth = False
    cfg.max_depth = 10  # meters
    cfg.input_size = (480, 640)  # (height, width)
    cfg.input_mean = (123.68, 116.78, 103.94)
    cfg.input_std = (0.017, 0.017, 0.017)
    
    model = NewCRFDepth(version=cfg.version, inv_depth=False, max_depth=cfg.max_depth)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(cfg.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()
    
    return model, cfg


def preprocess_rgb(frame, cfg):
    frame = cv2.resize(frame, (cfg.input_size[1], cfg.input_size[0]), interpolation=cv2.INTER_LINEAR)
    input_image = frame.astype(np.float32)
    # normalize
    for i in range(3):
        input_image[:, :, i] = (input_image[:, :, i] - cfg.input_mean[i]) * cfg.input_std[i]
    # transpose
    input_images = np.expand_dims(input_image, axis=0)  # (1, H, W, C)
    input_images = np.transpose(input_images, (0, 3, 1, 2))  # (1, C, H, W)
    image = Variable(torch.from_numpy(input_images)).cuda()
    return image


def postprocess_depth(depth, frame_size):
    """ depth: (h, w)
        frame_size: (H, W)
    """
    # resize depth map to the rgb frame size
    depth_resize = cv2.resize(depth, (frame_size[1], frame_size[0]))
    return depth_resize


def monocular_depth_estimation(video, model, cfg):
    """ depth estimation for each frame """
    depth_maps = []
    for frame in video:
        # preprocess video frame as the model input
        frame_input = preprocess_rgb(frame, cfg)
        with torch.no_grad(): # inference
            depth_est = model(frame_input)
            depth_est = depth_est.cpu().numpy().squeeze()  # (h, w)
        # post process
        depth_est = postprocess_depth(depth_est, frame.shape[:2])  # (H, W)
        depth_maps.append(depth_est)
    depth_maps = np.stack(depth_maps, axis=0)  # (T, H, W)
    return depth_maps
