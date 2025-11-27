
from pathlib import Path
from typing import Optional, Tuple, Union
import torch 
import numpy as np
import cv2
from torch.utils import data
import os
from torchvision.transforms import Normalize
from util.misc import resize_image
from util.pylogger import get_pylogger 

logger = get_pylogger(__name__)

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}  # optional, for dicts
    else:
        return np.array(x)  # fallback for numbers, lists of numbers, etc.


IMG_MEAN_255 = to_numpy([0.485, 0.456, 0.406]) * 255.
IMG_STD_255  = to_numpy([0.229, 0.224, 0.225]) * 255.

def lurb_to_cwh(
    lurb: Union[list, np.ndarray, torch.Tensor]
):
    c = (lurb[..., :2] + lurb[..., 2:])/2 # (..., 2)
    wh = lurb[..., 2:] - lurb[..., :2]  # (..., 2)  

    cwh = np.concatenate([c, wh], axis=-1)  # (..., 4)
    return cwh 

def cwh_to_lurb(
    cwh: Union[list, np.ndarray, torch.Tensor],
):
    """
    convert the center-width-height format to left-upper-right-bottom
    """
    # center - w/2, center + w/2
    l = cwh[..., :2] - cwh[..., 2:]/2 # (..., 2)
    r = cwh[..., :2] + cwh[..., 2:]/2

    lurb = np.concatenate([l, r], axis=-1)  # (..., 4)  
    return lurb 


def fit_bbox_to_aspect_ratio(
    bbox: np.ndarray,
    tgt_ratio: Optional[Tuple[int, int]] = None,
    bbox_type: str = 'lurb'
):
    bbox = bbox.copy()
    if bbox_type == 'lurb':
        bbox_cwh = lurb_to_cwh(bbox)
        bbox_wh = bbox_cwh[2:]
    elif bbox_type == 'cwh':
        bbox_wh = bbox[2:]
    else:
        raise ValueError(f'Unsupported bbox type: {bbox_type}')
    
    new_bbox_wh = expand_wh_to_aspect_ratio(bbox_wh, tgt_ratio)

    if bbox_type == 'lurb':
        bbox_cwh[2:] = new_bbox_wh
        new_bbox = cwh_to_lurb(bbox_cwh)
    elif bbox_type == 'cwh':
        new_bbox = np.concatenate([bbox[:2], new_bbox_wh])
    else:
        raise ValueError(f"Unsupported bbox type: {bbox_type}")

    return new_bbox

def cwh_to_cs(
    cwh : Union[list, np.ndarray, torch.Tensor],
    reduce: Optional[str] = None, 
):
    '''
        reduce to the square bbox, the larger will be used 
    '''
    assert cwh.shape[-1] == 4, f"Invalid shape: {cwh.shape}, should be (..., 4)"
    c = cwh[..., :2] # (..., 2)
    s = cwh[..., 2:].max(axis=-1)

    cs = np.concatenate([c, s[..., None]], axis=-1)  # (..., 3)
    return cs 

def cs_to_cwh(
    cs: Union[list, np.ndarray, torch.Tensor],
):
    c = cs[..., :2] # (..., 2)
    s = cs[..., 2] # (..., 1)

    cwh = np.concatenate([c, s[..., None], s[..., None]], axis=-1)  # (..., 4)
    return cwh

def expand_wh_to_aspect_ratio(bbox_wh: np.ndarray, tgt_aspect_ratio: Optional[Tuple[int, int]]=None):
    if tgt_aspect_ratio is None:
        return bbox_wh
    try:
        bbox_w, bbox_h = bbox_wh
    except (ValueError, ValueError):
        raise ValueError(f"Invalid bbox_wh content: {bbox_wh}")
    
    tgt_w, tgt_h = tgt_aspect_ratio
    if bbox_h / bbox_w < tgt_h / tgt_w:
        new_h = max(bbox_w * tgt_h / tgt_w, bbox_h)
        new_w = bbox_w
    else:
        new_h = bbox_h
        new_w = max(bbox_h * tgt_w / tgt_h, bbox_w)
    assert new_h >= bbox_h and new_w >= bbox_w

    return to_numpy([new_w, new_h])


def cs_to_lurb(
    cs: Union[list, np.ndarray, torch.Tensor],
):
    return cwh_to_lurb(cs_to_cwh(cs))


def crop_with_lurb(data, lurb, padding=0):
    """
    Crop the img-like data according to the lurb bounding box.
    
    ### Args
    - data: Union[np.ndarray, torch.Tensor], shape (H, W, C)
        - Data like image.
    - lurb: Union[list, np.ndarray, torch.Tensor], shape (4,)
        - Bounding box with [left, upper, right, bottom] coordinates.
    - padding: int, default 0
        - Padding value for out-of-bound areas.
        
    ### Returns
    - Union[np.ndarray, torch.Tensor], shape (H', W', C)
        - Cropped image with padding if necessary.
    """
    lurb = np.array(lurb).astype(np.int32)
    l_, u_, r_, b_ = lurb
    H, W, C = data.shape

    # compute the cropped patch size 
    H_patch = b_ - u_
    W_patch = r_ - l_

    # create an output buffer of the crop size, initialized to padding 
    if isinstance(data, np.ndarray):
        output = np.full((H_patch, W_patch, C), padding, dtype=data.dtype)
    else:
        output = torch.full((H_patch, W_patch, C), padding, dtype=data.dtype)
    
    # Calculate the valid region in the original data
    valid_l_ = max(0, l_)
    valid_u_ = max(0, u_)
    valid_r_ = min(W, r_)
    valid_b_ = min(H, b_)

    target_l_ = valid_l_ - l_
    target_u_ = valid_u_ - u_
    target_r_ = target_l_ + (valid_r_ - valid_l_)
    target_b_ = target_u_ + (valid_b_ - valid_u_)

    # Calculate the corresponding valid region in the output
    target_l_ = valid_l_ - l_
    target_u_ = valid_u_ - u_
    target_r_ = target_l_ + (valid_r_ - valid_l_)
    target_b_ = target_u_ + (valid_b_ - valid_u_)

    output[target_u_:target_b_, target_l_:target_r_, :] = data[valid_u_:valid_b_, valid_l_:valid_r_, :]

    return output


class EvalMoyoDataset(data.Dataset):

    def __init__(self, cfg, npz_fn: Union[str, Path], ignore_img=False):
        super().__init__()
        self.data = None
        self.cfg = cfg 
        self._load_data(npz_fn)
        self.ds_root = self._get_ds_root()
        self.bbox_ratio = (192, 256) # the ViT backbone's input size is w=192, h=256
        self.ignore_img = ignore_img   
        self.normalize_img = Normalize(mean=cfg.policy.img_mean,
                                std=cfg.policy.img_std)
        self.ds_name = 'moyo_hard' if 'moyo_hard' in str(npz_fn) else 'moyo_v2'

    def _load_data(self, npz_fn: Union[str, Path]):
        raw_data = np.load(npz_fn, allow_pickle=True)

        # Load basic information 
        self.seq_names = raw_data['names'] # (L, )
        self.img_paths = raw_data['img_paths'] # (L, )
        self.bbox_centers = raw_data['centers'].astype(np.float32) # (L, 2)
        self.bbox_scales = raw_data['scales'].astype(np.float32) # (L, 2)   
        self.L = len(self.seq_names)
        # Load the g.t. SMPL parameters.
        self.genders = raw_data.get('genders', None) # (L, 2) or None 
        self.global_orient = raw_data['smpl'].item()['global_orient'].reshape(-1, 1 ,3).astype(np.float32)  # (L, 1, 3)
        self.body_pose     = raw_data['smpl'].item()['body_pose'].reshape(-1, 23 ,3).astype(np.float32)  # (L, 23, 3)
        self.betas         = raw_data['smpl'].item()['betas'].reshape(-1, 10).astype(np.float32)  # (L, 10)

        logger.info(f'Loaded {self.L} samples from {npz_fn}')
    
    def __len__(self):
        return self.L
    
    def _get_ds_root(self):
        return Path(self.cfg.paths.data_inputs) / "skel-evaluation-data" / "moyo"

    def _process_img_patch(self, idx):
        ''' Load and crop according to bbox'''
        if self.ignore_img:
            return np.zeros((1), dtype=np.float32)
        # Load images and metas
        img = cv2.imread(self.ds_root / self.img_paths[idx], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = img 
        # 1. get the image size 
        img_size = np.array([img.shape[0], img.shape[1]])
        # 2. resize the image to 256x256
        aspect_ratio, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                        (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        scale = self.bbox_scales[idx] # (2,)
        center = self.bbox_centers[idx] # (2,)

        # bbox_cwh = [1292, 845, 351, 778]
        bbox_cwh = np.concatenate([center, scale], axis=0) # (4,) lurb format
        bbox_cwh = fit_bbox_to_aspect_ratio(
            bbox = bbox_cwh,
            tgt_ratio = self.bbox_ratio,
            bbox_type = 'cwh'
        )
        # make it to square
        bbox_cs = cwh_to_cs(bbox_cwh, reduce='max') # (3,) make it to square 
        bbox_lurb = cs_to_lurb(bbox_cs)
        img_patch = crop_with_lurb(img, bbox_lurb) # (H', W', RGB)
        img_patch = cv2.resize(img_patch, (256, 256))
        img_patch_normalized = (img_patch - IMG_MEAN_255) / IMG_STD_255 # (H', W', RGB)
        img_patch_normalized = img_patch_normalized.astype(np.float32).transpose(2,0,1)
        return img_patch_normalized, cv_img, img_size, img_full_resized

    def __getitem__(self, idx):
        ret = {}
        ret['seq_name'] = self.seq_names[idx]
        ret['smpl'] = {
                'global_orient': self.global_orient[idx],
                'body_pose'    : self.body_pose[idx],
                'betas'        : self.betas[idx],
            }
        if self.genders is not None:
            ret['gender'] = self.genders[idx]
        # ret['img'], ret['cv_img'] = self._process_img_patch(idx)
        ret['img'], _, ret['img_size'], ret['img_full_resized'] = self._process_img_patch(idx)
        ret['ds_name'] = self.ds_name
        ret['box_center'] = self.bbox_centers[idx]
        ret['bbox_size'] = self.bbox_scales[idx].max()
        return ret 