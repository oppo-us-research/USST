import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers, math



class CoordLoss(nn.Module):
    def __init__(self, cfg, max_epoch=1500):
        """Simple Mean-squared Error as the loss function"""
        super(CoordLoss, self).__init__()
        if cfg.function == 'MSE':
            self.coord_loss = nn.MSELoss(reduction='none')
        elif cfg.function == 'Huber':
            self.coord_loss = nn.HuberLoss(reduction='none', delta=getattr(cfg, 'delta', 1.0))

        self.scale = getattr(cfg, 'scale', 1.0)
        self.robust_weighing = getattr(cfg, 'robust_weighing', False)
        self.tau = getattr(cfg, 'tau', 0.1)
        self.velo_coeff = getattr(cfg, 'velo_coeff', 0)
        self.velo_warp = getattr(cfg, 'velo_warp', 0)


    def get_robust_weights(self, z_values, tau=0.1):
        """ Compute the robust weights using temporal depth values
        """
        diffs = torch.cat([torch.tensor([0.0]).to(z_values.device), torch.abs(torch.diff(z_values))])
        horizon = z_values.size(0)
        weights = horizon * F.softmax(-diffs / tau, dim=-1)
        return weights
    

    def compute_loss(self, y_hat, y_true, logvar=torch.tensor([0.0]), weights=torch.tensor([1.0])):
        """ Loss function
        """
        # huber loss
        loss = self.coord_loss(y_hat, y_true)  # (Tu, 3)
        # uncertainty regularization
        beta_1 = torch.exp(-logvar).to(loss.device)
        beta_2 = logvar.to(loss.device)
        if logvar.size(-1) in [1, 3]:  # temporal or spacetime or None
            loss = beta_1 * loss + beta_2  # (Tu, 3)
        elif logvar.size(-1) == 2: # hybrid
            loss_xy = beta_1[:, 0:1] * loss[:, :-1] + beta_2[:, 0:1]
            loss_z = (beta_1[:, 1:2] * loss[:, -1:] + beta_2[:, 1:2]) * weights.unsqueeze(-1)
            loss = torch.cat([loss_xy, loss_z], dim=-1)
        # take sum over xyz, and mean over t
        loss = torch.mean(torch.sum(loss, dim=-1))
        return loss


    def forward(self, outputs, nframes, trajectories):
        """
            outputs: a list of dict() with obs_ratio (scalar) as key and unobs_pred (Tu, 3) as value
            nframes: (B,)
            trajectories: (B, T, 3)
        """
        outputs_unobs, outputs_obs = outputs
        unobs_avg_loss, obs_avg_loss, cnt = 0, 0, 0
        unobs_velo_loss, unobs_warp_loss, obs_velo_loss = 0, 0, 0
        unobs_kld_loss, obs_kld_loss = 0, 0
        
        for b, ratio_preds in enumerate(outputs_unobs):
            # get the number observed frames and full frames
            num_full = nframes[b]
            traj_gt = trajectories[b]  # (T,3)
            weights = self.get_robust_weights(traj_gt[:num_full, 2], tau=self.tau) if self.robust_weighing and traj_gt.size(-1) == 3 else torch.ones((num_full)).to(traj_gt.device)

            for i, (r, preds) in enumerate(ratio_preds.items()):
                num_obs = torch.floor(num_full * float(r)).to(torch.long)
                out_dim = preds['traj'].size(-1)

                # prepare the ground truth (from the last observed to the last valid frame)
                gts = traj_gt[num_obs: num_full, :]

                logvar = preds['unct'] if 'unct' in preds else torch.tensor([0.0])
                unobs_avg_loss += self.compute_loss(preds['traj'], gts[:, :out_dim], logvar=logvar, weights=weights[num_obs: num_full])
                
                if 'velo' in preds:
                    # velocity loss
                    gt_velo = torch.diff(torch.cat((traj_gt[num_obs-1: num_obs, :], gts), dim=0), dim=0)
                    unobs_velo_loss += self.velo_coeff * self.compute_loss(preds['velo'], gt_velo[:, :out_dim])
                    # accumulative warping loss
                    traj_warp = traj_gt[num_obs-1: num_obs, :out_dim] + torch.cumsum(preds['velo'], dim=0)
                    unobs_warp_loss += self.velo_warp * self.compute_loss(traj_warp, preds['traj'])
                
                # loss for the observed frames
                if outputs_obs is not None:
                    pred_obs = outputs_obs[b][r]
                    logvar = pred_obs['unct'] if 'unct' in pred_obs else torch.tensor([0.0])
                    obs_avg_loss += self.compute_loss(pred_obs['traj'], traj_gt[:num_obs, :out_dim], logvar=logvar, weights=weights[:num_obs])
                    
                    if 'velo' in pred_obs:
                        # velocity loss
                        gt_velo = torch.diff(torch.cat((torch.zeros((1, out_dim)).to(gts.device), traj_gt[:num_obs]), dim=0), dim=0)
                        obs_velo_loss += self.velo_coeff * self.compute_loss(pred_obs['velo'], gt_velo[:, :out_dim])     
                cnt += 1

        unobs_avg_loss /= cnt
        total_loss = unobs_avg_loss
        loss_out = {'total_loss': total_loss,
                    'unobs_loss': unobs_avg_loss}
        
        if self.velo_coeff > 0:
            unobs_velo_loss /= cnt
            total_loss += unobs_velo_loss
            loss_out.update({'total_loss': total_loss,
                             'unobs_velo_loss': unobs_velo_loss})
        
        if self.velo_warp > 0:
            unobs_warp_loss /= cnt
            total_loss += unobs_warp_loss
            loss_out.update({'total_loss': total_loss,
                             'unobs_warp_loss': unobs_warp_loss})
        
        if outputs_obs is not None:
            obs_avg_loss /= cnt
            total_loss += obs_avg_loss
            loss_out.update({'total_loss': total_loss,
                             'obs_loss': obs_avg_loss})
            
            if self.velo_coeff > 0:
                obs_velo_loss /= cnt
                total_loss += obs_velo_loss
                loss_out.update({'total_loss': total_loss,
                                'obs_velo_loss': obs_velo_loss})

        for k, v in loss_out.items():
            loss_out[k] = v * self.scale

        return loss_out