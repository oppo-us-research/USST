"""
* Copyright (c) 2023 OPPO. All rights reserved.
*
*
* Licensed under the Apache License, Version 2.0 (the "License"):
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and 
* limitations under the License.
"""

import numpy as np
import os
import torch
import importlib

from src.config import parse_configs
from src.EgoPAT3DLoader import build_dataloaders
from src.utils import set_deterministic, compute_displacement_errors, compute_block_distances 
from src.runtime import run_test
from src.utils_io import load_checkpoint, print_de_table, print_delta_table


def get_test_results(model, test_loader, result_file):
    if not os.path.exists(result_file):
        # run test inference
        with torch.no_grad():
            all_preds, all_gt = run_test(cfg, model, test_loader)
        # save predictions
        np.savez(result_file[:-4], pred=all_preds, gt=all_gt)
    else:
        print("Result file exists. Loaded from file: %s."%(result_file))
        all_results = np.load(result_file, allow_pickle=True)
        all_preds, all_gt = all_results['pred'][()], all_results['gt'][()]
    return all_preds, all_gt


def test(cfg, intrinsics):

    # build test dataloaders
    print("Loading dataset...")
    test_loader, testnovel_loader = build_dataloaders(cfg, phase='test')

    # build the model
    model_module = importlib.import_module('src.models.{}'.format(cfg.MODEL.arch))
    model = getattr(model_module, cfg.MODEL.arch)(cfg.MODEL, seq_len=cfg.DATA.max_frames, input_size=cfg.DATA.transform.input_size[0])
    model = model.to(device=cfg.device)
    # load checkpoints
    model, test_epoch = load_checkpoint(cfg, model)
    model = model.eval()
    
    # result folder
    result_path = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'test-e{}'.format(test_epoch))
    os.makedirs(result_path, exist_ok=True)
    eval_space = getattr(cfg.TEST, 'eval_space', '3d')

    ### test on the seen scenes
    print("Seen secenes...")
    all_preds, all_gt = get_test_results(model, test_loader, os.path.join(result_path, 'test_seen.npz'))
    # evaluation
    all_ades_seen, all_fdes_seen = compute_displacement_errors(all_preds, all_gt, 
                                                               target=model.target, eval_space=eval_space, intrinsics=intrinsics)
    all_dxs_seen, all_dys_seen, all_dzs_seen = compute_block_distances(all_preds, all_gt, 
                                                               target=model.target, eval_space=eval_space, intrinsics=intrinsics)

    # print tables
    print_de_table(all_ades_seen, all_fdes_seen, subset='Seen')
    print_delta_table(all_dxs_seen, all_dys_seen, all_dzs_seen, subset='Seen')


    ### test on the unseen scenes
    print("\nUnseen secenes...")
    all_preds, all_gt = get_test_results(model, testnovel_loader, os.path.join(result_path, 'test_unseen.npz'))
    # evaluation
    all_ades_unseen, all_fdes_unseen = compute_displacement_errors(all_preds, all_gt, 
                                                               target=model.target, eval_space=eval_space, intrinsics=intrinsics)
    all_dxs_unseen, all_dys_unseen, all_dzs_unseen = compute_block_distances(all_preds, all_gt, 
                                                               target=model.target, eval_space=eval_space, intrinsics=intrinsics)
    
    print_de_table(all_ades_unseen, all_fdes_unseen, subset='Unseen')
    print_delta_table(all_dxs_unseen, all_dys_unseen, all_dzs_unseen, subset='Unseen')

    print("\nDone!")


if __name__ == '__main__':
    # parse input arguments
    cfg = parse_configs(phase='test')

    # fix random seed 
    set_deterministic(cfg.seed)
    
    # constant camera parameters
    intrinsics = {'fx': 1.80820276e+03, 'fy': 1.80794556e+03, 
                  'ox': 1.94228662e+03, 'oy': 1.12382178e+03,
                  'w': 3840, 'h': 2160}  # from EgoPAT3D preprocessing code

    # test
    test(cfg, intrinsics)