""" Code adapted from: https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/utils/schedulers.py
"""
import torch
import math


class CosineAnnealingWarmupRestarts_New(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):

    def __init__(self, warmup_epochs, lr_decay=1.0, lr_decay_epochs=None, *args, **kwargs):

        super(CosineAnnealingWarmupRestarts_New, self).__init__(*args, **kwargs)

        # Init optimizer with low learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.eta_min

        self.warmup_epochs = warmup_epochs
        self.lr_decay = lr_decay
        self.lr_decay_epochs = lr_decay_epochs

        self.warmup_lrs = []
        for base_lr in self.base_lrs:
            # Get target LR after warmup is complete
            target_lr = self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * warmup_epochs / self.T_i)) / 2

            # Linearly interpolate between minimum lr and target_lr
            linear_step = (target_lr - self.eta_min) / self.warmup_epochs
            warmup_lrs = [self.eta_min + linear_step * (n + 1) for n in range(warmup_epochs)]
            self.warmup_lrs.append(warmup_lrs)

    def step(self, epoch=None):

        # Called on super class init
        if epoch is None:
            super(CosineAnnealingWarmupRestarts_New, self).step(epoch=epoch)

        else:
            if epoch < self.warmup_epochs:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.warmup_lrs[i][epoch]

                # Fulfill misc super() funcs
                self.last_epoch = math.floor(epoch)
                self.T_cur = epoch
                self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

            else:
                # decay the base_lr which will be used in the step() call for cosine restart
                if self.lr_decay_epochs and epoch + 1 in self.lr_decay_epochs:
                    self.base_lrs = [lr * self.lr_decay for lr in self.base_lrs]
                super(CosineAnnealingWarmupRestarts_New, self).step(epoch=epoch)


def get_optimizer(cfg, params):
    # get optimizer
    if cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(params, lr=cfg.base_lr, momentum=cfg.momentum)
    elif cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'Adan':
        from adan_pytorch import Adan
        optimizer = Adan(params, lr=cfg.base_lr, weight_decay=cfg.weight_decay, betas=cfg.betas)
    else:
        raise NotImplementedError
    
    return optimizer


def get_scheduler(cfg, optimizer):
    # get learning rate scheduler
    if cfg.scheduler.type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    elif cfg.scheduler.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epoch, last_epoch=-1, eta_min=0)
    elif cfg.scheduler.type == 'cosine_warmup':
        scheduler = CosineAnnealingWarmupRestarts_New(warmup_epochs=cfg.scheduler.warmup_epoch,
                                                      lr_decay=cfg.scheduler.lr_decay,
                                                      lr_decay_epochs=cfg.scheduler.lr_decay_epoch, 
                                                      optimizer=optimizer, 
                                                      T_0=int(cfg.epoch / (cfg.scheduler.num_restarts + 1)),
                                                      eta_min=cfg.base_lr * cfg.scheduler.min_lr_factor)
    else:
        raise NotImplementedError
    return scheduler
    