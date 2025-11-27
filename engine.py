from collections import defaultdict
import json
import math
from pathlib import Path
import random
import sys
from typing import Iterable
import numpy as np
import torch
from tqdm import tqdm
from body_models.skel_utils.transforms import params_q2rep
from body_models.skel_wrapper import SKELWrapper
from datasets.constants import SMPL_MODEL_DIR
from run_hsmr_test import eval_pipeline, get_data
from util.geometry import rotation_6d_to_matrix
import util.misc as utils
from util.pylogger import get_pylogger
from smplx import SMPL
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F


logger = get_pylogger(__name__)


def to_device(obj, device):
  
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).to(device)
    elif isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_device(v, device) for v in obj]
    else:
        return obj


def compute_aux_loss(per_layer_params, targets):

    assert per_layer_params is not None and targets is not None, "per_layer_params and targets are required"
    B = targets['poses'].shape[0]
    loss = 0.
    
    gt_poses = params_q2rep(targets['poses']).reshape(-1, 24, 6)
    gt_rotmat = rotation_6d_to_matrix(gt_poses)

    for i, layer_params in enumerate(per_layer_params):
        pd_poses_i = layer_params[f'pd_poses_{i}'].reshape(-1, 24, 6)
        pd_rotmat_i = rotation_6d_to_matrix(pd_poses_i)
        loss += F.l1_loss(pd_rotmat_i, gt_rotmat, reduction='mean')

    return loss


def train_one_epoch(cfg, model: torch.nn.Module, ema_model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, 
                    lr_scheduler: torch.optim.lr_scheduler._LRScheduler, device: torch.device, writter, epoch: int):
    model.train()
    print_freq = cfg.logger.logger_interval
    
    try:
        from torch.amp import GradScaler
        scaler = GradScaler(device_type='cuda')
    except Exception:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    
    for iter, batch in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Training {epoch}"):

        # get the image
        batch = {k: v.to(device) for k, v in batch.items() if k != 'ds_name'}
        img = batch['img']
        B = img.shape[0]

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda'):
            predict_enc, predict_dec, per_layer_params = model(batch)         
            outputs_enc = {
                'kp2d': predict_enc['pd_kp2d'],  # (B, K=44, 2)
                'kp3d': predict_enc['pd_kp3d'],  # (B, K=44, 3)
                'betas': predict_enc['pd_skel_params']['betas'], # (B, 10)
                'poses': predict_enc['pd_skel_params']['poses'], # (B, 46)
                'global_orient': predict_enc['pd_skel_params']['poses'][:, :3],  # (B, 3)
            }

            outputs_dec = {
                'kp2d': predict_dec['pd_kp2d'],  # (B, K=44, 2)
                'kp3d': predict_dec['pd_kp3d'],  # (B, K=44, 3)
                'betas': predict_dec['pd_skel_params']['betas'], # (B, 10)
                'poses': predict_dec['pd_skel_params']['poses'], # (B, 46)
                'global_orient': predict_dec['pd_skel_params']['poses'][:, :3],  # (B, 3)
            }

            targets = {
                'kp2d': batch['kp2d'],  # (B, K=44, 3)
                'kp3d': batch['kp3d'],  # (B, K=44, 4)
                'betas': batch['skel_betas'],  # (B, 10)
                'poses': batch['skel_poses'],  # (B, 46)
                'global_orient': batch['skel_poses'][:, :3],  # (B, 3)
            }
            # using half-mixed precision Loss calculation
       
            loss_enc_dict = criterion(outputs_enc, targets, batch['_trans'])
            loss_dec_dict = criterion(outputs_dec, targets, batch['_trans'])

           
            if cfg.misc.aux_loss_weight:
                loss_layer = compute_aux_loss(per_layer_params, targets)

            assert loss_enc_dict is not None and loss_dec_dict is not None, "Loss dict is None"
            # scale the losses
            weight_dict = cfg.loss_weights 
            losses_enc = sum(loss_enc_dict[k] * weight_dict[k] for k in loss_enc_dict.keys() if k in weight_dict)
            losses_dec = sum(loss_dec_dict[k] * weight_dict[k] for k in loss_dec_dict.keys() if k in weight_dict)
            
            if cfg.misc.aux_loss_weight:
                total_loss = cfg.trainer.lamda * losses_enc + cfg.misc.aux_loss_weight * loss_layer + losses_dec
            else:
                total_loss = cfg.trainer.lamda * losses_enc + losses_dec

        # reduce losses over all GPUs for logging purposes
        loss_enc_dict_reduced = utils.reduce_dict(loss_enc_dict)
        loss_enc_dict_reduced_scaled = {k: (v * weight_dict[k]).item()
                                    for k, v in loss_enc_dict_reduced.items() if k in weight_dict}
        
        loss_dec_dict_reduced = utils.reduce_dict(loss_dec_dict)
        loss_dec_dict_reduced_scaled = {k: (v * weight_dict[k]).item()
                                    for k, v in loss_dec_dict_reduced.items() if k in weight_dict}
       
        scaler.scale(total_loss).backward()
        
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.module.get_trainable_parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        lr_scheduler.step()
        ema_model.update(model.module)
        
        if (iter % print_freq == 0 or iter == len(data_loader) - 1) and utils.get_rank() == 0:
            
            checkpoint_path = Path(cfg.output_dir) / 'checkpoints/last_step.pth'
            utils.save_on_master({
                'model': model.module.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'step': iter,
                'global_step': epoch * len(data_loader) + iter,
                'random_state': random.getstate(),
                'numpy_state': np.random.get_state(),
                'torch_state': torch.get_rng_state(),
                'cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                'cfg': cfg,
            }, checkpoint_path)


            # tensorboard log
            if writter and math.isfinite(total_loss.item()):
                for k, v in loss_enc_dict_reduced_scaled.items():
                    writter.add_scalar(f'train_loss_enc/{k}', v, epoch * len(data_loader) + iter)

                for k, v in loss_dec_dict_reduced_scaled.items():
                    writter.add_scalar(f'train_loss_dec/{k}', v, epoch * len(data_loader) + iter)
                # total_loss
                writter.add_scalar(f'train_loss/total_loss', total_loss, epoch * len(data_loader) + iter)
                # LR_curve
                writter.add_scalar(f'train_loss/LR', optimizer.param_groups[0]['lr'], epoch * len(data_loader) + iter)

            if math.isfinite(total_loss.item()):
                if cfg.misc.aux_loss_weight:
                    logger.info(
                        f'Epoch: [{epoch}] Iter: [{iter}/{len(data_loader)}] '
                        f'aux_loss_weight: {cfg.misc.aux_loss_weight:.3f} '
                        f'Loss: {total_loss:.4f} '
                        f'layer Losses: {loss_layer * cfg.misc.aux_loss_weight:.4f} '
                        f'enc Losses: {json.dumps(loss_enc_dict_reduced_scaled, indent=2)} '
                        f'dec Losses: {json.dumps(loss_dec_dict_reduced_scaled, indent=2)} '
                        f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
                    )
                else:
                    logger.info(
                        f'Epoch: [{epoch}] Iter: [{iter}/{len(data_loader)}] '
                        f'Loss: {total_loss:.4f} '
                        f'enc Losses: {json.dumps(loss_enc_dict_reduced_scaled, indent=2)} '
                        f'dec Losses: {json.dumps(loss_dec_dict_reduced_scaled, indent=2)} '
                        f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
                    )


@torch.no_grad()
def evaluate_moyo(cfg, model: nn.Module, data_loader, device, writter, epoch, ema=False):
    model.eval()

    # evaluate emdb dataset
    res_mpjpe, res_pampjpe, res_pve = [], [], []
    smpl_model = SMPL(SMPL_MODEL_DIR, gender='neutral').to(device)
    
    data_inputs = Path(cfg.paths.data_inputs)
    skel_model = SKELWrapper(
        gender = 'male',
        model_path = data_inputs / 'body_models' / 'skel',
        joint_regressor_extra = data_inputs / 'body_models' / 'train-eval-utils' / 'SMPL_to_J19.pkl',
        joint_regressor_custom = data_inputs / 'body_models' / 'J_regressor_SMPL_MALE.pkl',
    ).to(device)
    
    res_mpjpe, res_pampjpe, res_pve, res_papve = [], [], [], []
    for _, targets in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Testing {epoch}"):

        targets = {k: to_device(v, device) for k, v in targets.items()} 
        
        gt_smpl_params = targets['smpl']
        smpl_output = smpl_model(**gt_smpl_params)
        gt_vertices = smpl_output.vertices
        gt_joints = smpl_output.joints[:, :24] # (B, 24, 3)

        gt_pelvis = gt_joints[:, [0], :].clone()
        gt_joints = gt_joints - gt_pelvis
        gt_vertices = gt_vertices - gt_pelvis
        
        _, total_predict, _ = model.forward(targets)  
        skel_output = skel_model(**total_predict['pd_skel_params'], skelmesh=False)

        pd_vertices = skel_output.skin_verts
        pd_joints = skel_output.joints_custom
        pred_pelvis = pd_joints[:, [0], :].clone()
        pd_joints = pd_joints - pred_pelvis
        pd_vertices = pd_vertices - pred_pelvis
        

         # evaluate metrics
        mpjpe = eval_mpjpe(pd_joints, gt_joints)  # (B,)
        pa_mpjpe = eval_pampjpe(pd_joints, gt_joints)  # (B,)
        pve = eval_pve(pd_vertices, gt_vertices)  # (B,)    
        papve = eval_papve(pd_vertices, gt_vertices)  # (B,)

        res_mpjpe.append(mpjpe)
        res_pampjpe.append(pa_mpjpe)
        res_pve.append(pve)
        res_papve.append(papve)

    res_mpjpe = torch.cat(res_mpjpe, dim=0)
    res_pampjpe = torch.cat(res_pampjpe, dim=0)
    res_pve = torch.cat(res_pve, dim=0)
    res_papve = torch.cat(res_papve, dim=0)
        
    # cal the metrics 
    res_mpjpe = cal_metric_dist(res_mpjpe)
    res_pampjpe = cal_metric_dist(res_pampjpe)
    res_pve = cal_metric_dist(res_pve)
    res_papve = cal_metric_dist(res_papve)
    
    if utils.get_rank() == 0:
        if writter:
            writter.add_scalar(f'val_metrics/mpjpe', res_mpjpe, epoch)
            writter.add_scalar(f'val_metrics/pampjpe', res_pampjpe, epoch)
            writter.add_scalar(f'val_metrics/pve', res_pve, epoch)
            writter.add_scalar(f'val_metrics/papve', res_papve, epoch)

        if ema:
            logger.info(f"Evaluating EMA model")
                    
        logger.info(
                 f"MPJPE: {res_mpjpe:.4f}, PAMPJPE: {res_pampjpe:.4f}, PVE: {res_pve:.4f}, PAPVE: {res_papve:.4f}"
        )

    return res_mpjpe, res_pampjpe, res_pve, res_papve 
    

@torch.no_grad()
def evaluate_hmr2(cfg, model: nn.Module, data_list):
    model.eval()
    # 3. Evaluation.
    eval_pipeline(model, data_list)


@torch.no_grad()
def evaluate(cfg, model: nn.Module, data_loader, device, writter, epoch, ema=False):
    model.eval()

    # evaluate emdb dataset
    res_mpjpe, res_pampjpe, res_pve = [], [], []
    smpl = SMPL(SMPL_MODEL_DIR, gender='neutral')
    
    for _, targets in tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Testing {epoch}"):

        targets = {k: to_device(v, device) for k, v in targets.items()} 
        _, predict_dec, _ = model.forward(targets)       
        smpl_J_regressor = smpl.J_regressor.to(device)

        gt_vertices = targets['vertices']
        gt_kp_3d = torch.matmul(smpl_J_regressor, gt_vertices)  # (B, 24, 3)
        gt_pelvis = (gt_kp_3d[:, [1], :] + gt_kp_3d[:, [2], :]) / 2.0
        gt_kp_3d = gt_kp_3d - gt_pelvis  # (B, 24, 3)
        gt_vertices = gt_vertices - gt_pelvis

        pd_vertices = predict_dec['pd_skin_verts'].to(device)  # (B, 6890, 3)
        pd_kp_3d = torch.matmul(smpl_J_regressor, pd_vertices)  # (B, 24, 3)    
        pd_pelvis = (pd_kp_3d[:, [1], :] + pd_kp_3d[:, [2], :]) / 2.0
        pd_kp_3d = pd_kp_3d - pd_pelvis
        pd_vertices = pd_vertices - pd_pelvis

        # evaluate metrics
        mpjpe = eval_mpjpe(pd_kp_3d, gt_kp_3d)  # (B,)
        pa_mpjpe = eval_pampjpe(pd_kp_3d, gt_kp_3d)  # (B,)
        pve = eval_pve(pd_vertices, gt_vertices)  # (B,)    

        res_mpjpe.append(mpjpe)
        res_pampjpe.append(pa_mpjpe)
        res_pve.append(pve)

    res_mpjpe = torch.cat(res_mpjpe, dim=0)
    res_pampjpe = torch.cat(res_pampjpe, dim=0)
    res_pve = torch.cat(res_pve, dim=0)
        
    # cal the metrics 
    res_mpjpe = cal_metric_dist(res_mpjpe)
    res_pampjpe = cal_metric_dist(res_pampjpe)
    res_pve = cal_metric_dist(res_pve)
    
    if utils.get_rank() == 0:
        if writter:
            writter.add_scalar(f'val_metrics/mpjpe', res_mpjpe, epoch)
            writter.add_scalar(f'val_metrics/pampjpe', res_pampjpe, epoch)
            writter.add_scalar(f'val_metrics/pve', res_pve, epoch)

        if ema:
            logger.info(f"Evaluating EMA model")
                    
        logger.info(
                 f"MPJPE: {res_mpjpe:.4f}, PAMPJPE: {res_pampjpe:.4f}, PVE: {res_pve:.4f}"
        )

    return res_mpjpe, res_pampjpe, res_pve 

def cal_metric_dist(data):
    
    tensor_list = []
    world_size = utils.get_world_size()
    for _ in range(world_size):
        tensor_list.append(torch.empty(data.shape, dtype=data.dtype, device=data.device))

    dist.all_gather(tensor_list, data)
    data = torch.cat(tensor_list, dim=0)
    data = data.mean().item()
    return data 


def eval_papve(pd_verts, gt_verts):
    pd_aligned = similarity_align_to(pd_verts, gt_verts)
    return eval_pve(pd_aligned, gt_verts)


def eval_pve(pd_verts, gt_verts):
    ## return B'
    scale = 1000
    result = torch.sqrt(torch.sum((gt_verts - pd_verts) ** 2, dim=-1)).mean(dim=-1)
    return result * scale

def eval_mpjpe(pd_j3d, gt_j3d):
    scale = 1000
    result = torch.sqrt(torch.sum((gt_j3d - pd_j3d) ** 2, dim=-1)).mean(dim=-1)
    return result * scale

def eval_pampjpe(pd_j3d, gt_j3d):
    pd_aligned = similarity_align_to(pd_j3d, gt_j3d)
    return eval_mpjpe(pd_aligned, gt_j3d)


def similarity_align_to(
        S1 : torch.Tensor,
        S2 : torch.Tensor,
    ):
       
    assert (S1.shape[-1] == 3 and S2.shape[-1] == 3), 'The last dimension of `S1` and `S2` must be 3.'
    assert (S1.shape[:-2] == S2.shape[:-2]), 'The batch size of `S1` and `S2` must be the same.'
    
    dtype = S1.dtype 
    device = S1.device
    batch_size = S1.shape[0]
    S1 = S1.to(torch.float32)
    S2 = S2.to(torch.float32)
            
    S1 = S1.transpose(-1, -2) # (B', 3, N) <- (B', N, 3)
    S2 = S2.transpose(-1, -2) # (B', 3, N) <- (B', N, 3)

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True) # (B', 3, 1)
    mu2 = S2.mean(axis=-1, keepdims=True) # (B', 3, 1)
    X1 = S1 - mu1 # (B', 3, N)
    X2 = S2 - mu2 # (B', 3, N)
    # 2. Compute variance of X1 used for scales.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0,2,1))
  
    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)
    
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=device)[None].repeat(batch_size, 1, 1) # (B', 3, 3)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1)) # (B', 3, 3)

    # 5. Recover scales.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1) # (B', 1, 1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1) # (B', 3, 1)

    # 7. Error:
    S1_aligned = scale * (R @ S1) + t # (B', 3, N)
    S1_aligned = S1_aligned.transpose(-1, -2) # (B', N, 3) <- (B', 3, N)
    # S1_aligned = S1_aligned.reshape(original_BN3) # (...B, N, 3)

    return S1_aligned.to(dtype)