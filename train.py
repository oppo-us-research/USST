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

import os
import torch
from tensorboardX import SummaryWriter
import importlib

from src.config import parse_configs
from src.EgoPAT3DLoader import build_dataloaders
from src.utils import set_deterministic, compute_displacement_errors, save_the_latest
from src.runtime import train_epoch, eval_epoch, print_eval_results
import src.Losses as loss_module
from src.optimizers import get_optimizer, get_scheduler



def train(cfg):
    # model snapshots
    model_dir = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'snapshot')
    os.makedirs(model_dir, exist_ok=True)

    # tensorboard logging
    logs_dir = os.path.join(cfg.output_dir, cfg.DATA.dataset, cfg.tag, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    writer = SummaryWriter(logs_dir)

    # build data loaders
    print("Loading dataset...")
    traindata_loader, valdata_loader = build_dataloaders(cfg, phase='trainval')
    
    # build the model
    model_module = importlib.import_module('src.models.{}'.format(cfg.MODEL.arch))
    model = getattr(model_module, cfg.MODEL.arch)(cfg.MODEL, seq_len=cfg.DATA.max_frames, input_size=cfg.DATA.transform.input_size[0]).train()
    model = model.to(device=cfg.device)

    # build the loss criterion
    args_loss = {'cfg': cfg.TRAIN.loss}
    criterion = getattr(loss_module, cfg.TRAIN.loss.type)(**args_loss)
    criterion = criterion.to(device=cfg.device)
    
    # optimizer & lr scheduler
    optimizer = get_optimizer(cfg.TRAIN, model.parameters())
    scheduler = get_scheduler(cfg.TRAIN, optimizer)
    
    # training loop
    for epoch in range(cfg.TRAIN.epoch):
        # train one epoch
        loss_train = train_epoch(cfg, model, traindata_loader, criterion, optimizer, writer, epoch)

        if (epoch + 1) % cfg.TRAIN.eval_interval == 0 or epoch + 1 == cfg.TRAIN.epoch:
            # test a model snapshot
            with torch.no_grad():
                all_preds_val, all_gt_val = eval_epoch(cfg, model, valdata_loader, criterion, writer, epoch)
                all_ades_val, all_fdes_val = compute_displacement_errors(all_preds_val, all_gt_val)

            print_eval_results(writer, all_ades_val, all_fdes_val, epoch=epoch, loss_train=loss_train)

        if (epoch + 1) % cfg.TRAIN.snapshot_interval == 0 or epoch + 1 == cfg.TRAIN.epoch:
            # save model snapshot
            save_dict = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            model_file = os.path.join(model_dir, cfg.TRAIN.snapshot_prefix + '%02d.pth'%(epoch + 1))
            save_the_latest(save_dict, model_file, topK=1, ignores=getattr(cfg.TRAIN.scheduler, 'lr_decay_epoch', []))
            print('Model has been saved as: %s'%(model_file))

        # update learning rate
        scheduler.step(epoch=epoch)

    writer.close()


if __name__ == '__main__':
    # parse input arguments
    cfg = parse_configs()

    # fix random seed 
    set_deterministic(cfg.seed)

    # train
    train(cfg)