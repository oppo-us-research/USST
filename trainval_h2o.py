import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
import importlib

from src.config import parse_configs
from src.H2OLoader import build_dataloaders
from src.utils import set_deterministic, save_the_latest
from src.utils_h2o import compute_displacement_errors, compute_block_distances
from src.runtime import train_epoch, eval_h2o_epoch, run_h2o_test, print_eval_results
import src.Losses as loss_module
from src.optimizers import get_optimizer, get_scheduler
from src.utils_io import load_checkpoint, print_de_table, print_delta_table



def get_test_results(model, test_loader, result_file):
    if not os.path.exists(result_file):
        # run test inference
        with torch.no_grad():
            all_preds, all_gt, all_cam_poses = run_h2o_test(cfg, model, test_loader)
        # save predictions
        np.savez(result_file[:-4], pred=all_preds, gt=all_gt, campose=all_cam_poses)
    else:
        print("Result file exists. Loaded from file: %s."%(result_file))
        all_results = np.load(result_file, allow_pickle=True)
        all_preds, all_gt, all_cam_poses = all_results['pred'][()], all_results['gt'][()], all_results['campose'][()]
    return all_preds, all_gt, all_cam_poses


def test(cfg):

    # build test dataloaders
    print("Loading dataset...")
    test_loader = build_dataloaders(cfg, phase='test')

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
    all_preds, all_gt, all_cam_poses = get_test_results(model, test_loader, os.path.join(result_path, 'test_results.npz'))
    # evaluation
    all_ades, all_fdes = compute_displacement_errors(all_preds, all_gt, all_cam_poses,
                                                               target=model.target, eval_space=eval_space, use_global=cfg.MODEL.use_global)
    all_dxs, all_dys, all_dzs = compute_block_distances(all_preds, all_gt, all_cam_poses,
                                                               target=model.target, eval_space=eval_space, use_global=cfg.MODEL.use_global)
    
    # print tables
    print_de_table(all_ades, all_fdes, subset='test')
    print_delta_table(all_dxs, all_dys, all_dzs, subset='test')

    print("\nDone!")


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
    criterion = getattr(loss_module, cfg.TRAIN.loss.type)(cfg=cfg.TRAIN.loss)
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
                all_preds_val, all_gt_val, all_cam_poses = eval_h2o_epoch(cfg, model, valdata_loader, criterion, writer, epoch)
                all_ades_val, all_fdes_val = compute_displacement_errors(all_preds_val, all_gt_val, all_cam_poses, use_global=cfg.MODEL.use_global)

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

    if cfg.eval:
        test(cfg)
    else:
        train(cfg)