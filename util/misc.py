from collections import defaultdict, deque
import datetime
import os
from pathlib import Path
from typing import Dict 
from omegaconf import OmegaConf
import torch 
import torch.distributed as dist
import cv2
import numpy as np 
import time


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True 

def init_distributed_mode(cfg):
    OmegaConf.set_struct(cfg, False)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.gpu = int(os.environ["LOCAL_RANK"])
        cfg.distributed = True
    elif 'SLURM_PROCID' in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
        cfg.world_size = int(os.environ['WORLD_SIZE'])
        cfg.distributed = True
        print('slurm !! ')
    else:
        print('Not using distribued mode')
        cfg.distributed = False
        return 

    cfg.distributed = True 
    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        cfg.rank, 'env://'), flush=True)

    dist.init_process_group(
        backend=cfg.dist_backend, 
        init_method='env://',
        world_size=cfg.world_size, 
        rank=cfg.rank
    )
    torch.distributed.barrier(device_ids=[cfg.gpu])
    setup_for_distributed(cfg.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def save_on_master(params: Dict, path: Path):
    if get_rank() == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(params, path)

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        # tensorize the values
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
        
    return reduced_dict


def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width/height

    # Calculate the new size while maintaining the aspect ratio
    if width > height:
        new_width = target_size
        new_height = int(target_size/aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)
    # Resize the image using OpenV
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it 
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x: start_x + new_width, :] = resized_image
    
    return aspect_ratio, final_img

def trans_points2d_parallel(keypoints_2d, trans):
    # Augment keypoints with ones to apply affine transformation
    ones = torch.ones((*keypoints_2d.shape[:2], 1), dtype=keypoints_2d.dtype, device=keypoints_2d.device)
    keypoints_augmented = torch.cat([keypoints_2d, ones], dim=-1)
    
    if trans.dtype != keypoints_2d.dtype:
        trans = trans.to(dtype=keypoints_2d.dtype) 
    # Apply transformation using batch matrix multiplication
    transformed_keypoints = torch.einsum('bij,bkj->bki', trans, keypoints_augmented)
    return transformed_keypoints[..., :2]