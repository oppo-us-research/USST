import argparse
import yaml
from easydict import EasyDict
from pprint import pformat
import os
import torch



def parse_configs(phase='train'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/baseline.yml',
                        help='The relative path of dataset.')
    parser.add_argument('--gpus', type=str, default="0", 
                        help="The delimited list of GPU IDs separated with comma. Default: '0'.")
    parser.add_argument('--num_workers', type=int, default=4, metavar='N',
                        help='The number of workers to load dataset. Default: 4')
    parser.add_argument('--eval', action='store_true', 
                        help='If specified, run the evaluation only.')
    parser.add_argument('--tag', type=str, default='debug',
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

    if cfg.tag == 'debug':
        cfg.TRAIN.epoch = 1
        if cfg.TRAIN.scheduler.type == 'cosine_warmup':
            cfg.TRAIN.epoch += cfg.TRAIN.scheduler.num_restarts # the minimum training epochs
        cfg.TRAIN.eval_interval = 1
        cfg.TRAIN.snapshot_interval = 1
        cfg.DATA.load_all = False
    
    if phase == 'test' and hasattr(cfg.DATA, 'load_all'):
        cfg.DATA.load_all = False
    
    # save configs to file
    exp_dir = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config_{}.yaml'.format(phase)), 'w') as f:
        f.writelines(pformat(vars(cfg)))

    return cfg