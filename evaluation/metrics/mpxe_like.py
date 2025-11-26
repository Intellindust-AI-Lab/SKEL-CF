from typing import Optional
import torch

from .utils import *


'''
All MPxE-like metrics will be implements here.

- Local Metrics: the inputs motion's translation should be removed (or may be automatically removed).
    - MPxE: call `eval_MPxE()`
    - PA-MPxE: cal `eval_PA_MPxE()`
- Global Metrics: the inputs motion's translation should be kept.
    - G-MPxE: call `eval_MPxE()`
    - W2-MPxE: call `eval_Wk_MPxE()`, and set k = 2
    - WA-MPxE: call `eval_WA_MPxE()`
'''


def eval_MPxE(
    pred  : torch.Tensor,
    gt    : torch.Tensor,
    scale : float = m2mm,
):
    '''
    Calculate the Mean Per <X> Error. <X> might be joints position (MPJPE), or vertices (MPVE).

    The results will be the sequence of MPxE of each multi-dim batch.

    ### Args
    - `pred`: torch.Tensor
        - shape = (...B, N, 3), where B is the multi-dim batch size, N is points count in one batch
        - the predicted joints/vertices position data
    - `gt`: torch.Tensor
        - shape = (...B, N, 3), where B is the multi-dim batch size, N is points count in one batch
        - the ground truth joints/vertices position data
    - `scale`: float, default = `m2mm`

    ### Returns
    - torch.Tensor
        - shape = (...B)
        - shape = ()
    '''
    # Calculate the MPxE.
    ret = L2_error(pred, gt).mean(dim=-1) * scale # (...B,)
    return ret


def eval_PA_MPxE(
    pred  : torch.Tensor,
    gt    : torch.Tensor,
    scale : float = m2mm,
):
    '''
    Calculate the Procrustes-Aligned Mean Per <X> Error. <X> might be joints position (PA-MPJPE), or
    vertices (PA-MPVE). Targets will be Procrustes-aligned and then calculate the per frame MPxE.

    The results will be the sequence of MPxE of each batch.

    ### Args
    - `pred`: torch.Tensor
        - shape = (...B, N, 3), where B is the multi-dim batch size, N is points count in one batch
        - the predicted joints/vertices position data
    - `gt`: torch.Tensor
        - shape = (...B, N, 3), where B is the multi-dim batch size, N is points count in one batch
        - the ground truth joints/vertices position data
    - `scale`: float, default = `m2mm`

    ### Returns
    - torch.Tensor
        - shape = (...B)
        - shape = ()
    '''
    # Perform Procrustes alignment.
    pred_aligned = similarity_align_to(pred, gt) # (...B, N, 3)
    # Calculate the PA-MPxE
    return eval_MPxE(pred_aligned, gt, scale) # (...B,)


def eval_Wk_MPxE(
    pred  : torch.Tensor,
    gt    : torch.Tensor,
    scale : float = m2mm,
    k_f   : int   = 2,
):
    '''
    Calculate the first k frames aligned (World aligned) Mean Per <X> Error. <X> might be joints
    position (PA-MPJPE), or vertices (PA-MPVE). Targets will be aligned using the first k frames
    and then calculate the per frame MPxE.

    The results will be the sequence of MPxE of each batch.

    ### Args
    - `pred`: torch.Tensor
        - shape = (..., L, N, 3), where L is the length of the sequence, N is points count in one batch
        - the predicted joints/vertices position data
    - `gt`: torch.Tensor
        - shape = (..., L, N, 3), where L is the length of the sequence, N is points count in one batch
        - the ground truth joints/vertices position data
    - `scale`: float, default = `m2mm`
    - `k_f`: int, default = 2
        - the number of frames to use for alignment

    ### Returns
    - torch.Tensor
        - shape = (..., L)
        - shape = ()
    '''
    L = max(pred.shape[-3], gt.shape[-3])
    assert L >= 2, f'Length of the sequence should be at least 2, but got {L}.'
    # Perform first two alignment.
    pred_aligned = first_k_frames_align_to(pred, gt, k_f) # (..., L, N, 3)
    # Calculate the PA-MPxE
    return eval_MPxE(pred_aligned, gt, scale) # (..., L)


def eval_WA_MPxE(
    pred  : torch.Tensor,
    gt    : torch.Tensor,
    scale : float = m2mm,
):
    '''
    Calculate the all frames aligned (World All aligned) Mean Per <X> Error. <X> might be joints
    position (PA-MPJPE), or vertices (PA-MPVE). Targets will be aligned using the first k frames
    and then calculate the per frame MPxE.

    The results will be the sequence of MPxE of each batch.

    ### Args
    - `pred`: torch.Tensor
        - shape = (..., L, N, 3), where L is the length of the sequence, N is points count in one batch
        - the predicted joints/vertices position data
    - `gt`: torch.Tensor
        - shape = (..., L, N, 3), where L is the length of the sequence, N is points count in one batch
        - the ground truth joints/vertices position data
    - `scale`: float, default = `m2mm`

    ### Returns
    - torch.Tensor
        - shape = (..., L)
        - shape = ()
    '''
    L_pred = pred.shape[-3]
    L_gt = gt.shape[-3]
    assert (L_pred == L_gt), f'Length of the sequence should be the same, but got {L_pred} and {L_gt}.'
    # WA_MPxE is just Wk_MPxE when k = L.
    return eval_Wk_MPxE(pred, gt, scale, L_gt)