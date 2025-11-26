from typing import Dict
import torch
import torch.nn as nn
from body_models.skel.skel_model import SKELOutput
from body_models.skel_utils.transforms import params_q2rep
from util.misc import trans_points2d_parallel
from util.geometry import axis_angle_to_matrix, pose_params_to_rot
import torch.nn as nn

from util.pylogger import get_pylogger


logger = get_pylogger(__name__)

class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        batch_size = conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()
    

class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 39):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        return loss.sum()


class ParameterLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        SKEL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')


    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor):
        """
        Compute SKEL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        loss_param = self.loss_fn(pred_param, gt_param)
        return loss_param.sum()
 


class HPE_Loss(nn.Module):
    """
    Human loss module that combines 2D keypoint loss, 3D keypoint loss, and SKEL parameter loss.
    """

    def __init__(self, cfg):
        super(HPE_Loss, self).__init__()
        self.cfg = cfg 
        # using 2DLossScaled
        loss_type = 'l1'
        self.kp2d_loss = Keypoint2DLoss(loss_type=loss_type)
        self.kp3d_loss = Keypoint3DLoss(loss_type=loss_type)
        self.param_loss = ParameterLoss(loss_type=loss_type)
        
        logger.info(f'using the loss type: {loss_type} for 2D and 3D keypoint loss, HPE Loss init!')

    def forward(self, output, target, _trans) -> Dict:
        """
        Compute the total HPE loss.
        Args:
            output (SKELOutput): Model outputs containing predicted keypoints and parameters.
            target (SKELOutput): Ground truth data containing target keypoints and parameters.
        Returns:
            torch.Tensor: Total human loss.
        """
        pd_poses_mat, _ = pose_params_to_rot(output['poses'])  # (B, J=24, 3, 3)
        pd_poses = pd_poses_mat.reshape(-1, 24*3*3)  # (B, 24*3*3)

        gt_poses_mat, _ = pose_params_to_rot(target['poses'])  # (B, J=24, 3, 3)
        gt_poses = gt_poses_mat.reshape(-1, 24*3*3)  # (B, 24*3*3)
        
        B = target['kp3d'].shape[0]

        # 2. Get Cropped gt keypoints2d
        gt_kp2d = target['kp2d']
        pd_kp2d = output['kp2d']
        pd_kp2d_cropped = trans_points2d_parallel(pd_kp2d, _trans)
        pd_kp2d_cropped = pd_kp2d_cropped / self.cfg.policy.img_patch_size - 0.5
        # 3. Calculate the kp2d Loss
        loss_kp2d = self.kp2d_loss(pd_kp2d_cropped, gt_kp2d) / B

        # 4. Calculate the kp3d Loss
        loss_kp3d = self.kp3d_loss(output['kp3d'], target['kp3d']) / B 
        # 5. use SMPL betas as supervision
        loss_betas = self.param_loss(output['betas'], target['betas']) / B 
        loss_poses = self.param_loss(pd_poses[:, 9:], gt_poses[:, 9:]) / B 
        loss_orient = self.param_loss(pd_poses[:, :9], gt_poses[:, :9]) / B
        
       
        loss_dict ={
            'kp3d': loss_kp3d,
            'kp2d': loss_kp2d,
            'betas': loss_betas,
            'poses_body': loss_poses,
            'poses_orient': loss_orient,
        }
        
        return loss_dict
    