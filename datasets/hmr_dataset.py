

import copy
import os
from typing import Any, Dict
import cv2
import numpy as np
import torch
from torchvision.transforms import Normalize

from datasets.constants import FLIP_KEYPOINT_PERMUTATION
from datasets.utils import expand_to_aspect_ratio, get_example
from datasets.utils_hmr2 import get_example_hmr2
from util.misc import resize_image


body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

class HmrDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg,
                 dataset_file: str,
                 img_dir: str,
                 ds_name: str, 
                 train: bool = False,
                 prune: Dict[str, Any] = {},
                 **kwargs):
        """
        Dataset class used for loading images and corresponding annotations.
        Args:
            cfg (CfgNode): Model config file.
            dataset_file (str): Path to npz file containing dataset info.
            img_dir (str): Path to image folder.
            train (bool): Whether it is for training or not (enables data augmentation).
        """
        super(HmrDataset, self).__init__()
        self.is_train = train
        self.cfg = cfg

        self.img_size = cfg.policy.img_patch_size
        self.mean = 255. * np.array(cfg.policy.img_mean)
        self.std = 255. * np.array(cfg.policy.img_std)

        self.img_dir = img_dir
        self.data = np.load(dataset_file, allow_pickle=True)

        self.imgname = self.data['imgname']
        self.personid = np.zeros(len(self.imgname), dtype=np.int32)
        self.extra_info = self.data.get('extra_info', [{} for _ in range(len(self.imgname))])

        self.use_skimage_antialias = False
        self.border_mode = cv2.BORDER_CONSTANT
        self.ds_name = ds_name

        num_pose = 3 * 24

        # Bounding boxes are assumed to be in the center and scale format
        self.center = self.data['center']
        self.scale = self.data['scale'].reshape(len(self.center), -1) / 200.0
        if self.scale.shape[1] == 1:
            self.scale = np.tile(self.scale, (1, 2))
        assert self.scale.shape == (len(self.center), 2)

        # Get gt SMPLX parameters, if available
        try:
            self.body_pose = self.data['body_pose'].astype(np.float32)
            self.has_body_pose = self.data['has_body_pose'].astype(np.float32)
        except KeyError:
            self.body_pose = np.zeros((len(self.imgname), num_pose), dtype=np.float32)
            self.has_body_pose = np.zeros(len(self.imgname), dtype=np.float32)
        try:
            self.betas = self.data['betas'].astype(np.float32)
            self.has_betas = self.data['has_betas'].astype(np.float32)
        except KeyError:
            self.betas = np.zeros((len(self.imgname), 10), dtype=np.float32)
            self.has_betas = np.zeros(len(self.imgname), dtype=np.float32)

        # try:
        #     self.trans = self.data['trans'].astype(np.float32)
        # except KeyError:
        #     self.trans = np.zeros((len(self.imgname), 3), dtype=np.float32)

        # Try to get 2d keypoints, if available
        try:
            body_keypoints_2d = self.data['body_keypoints_2d']
        except KeyError:
            body_keypoints_2d = np.zeros((len(self.center), 25, 3))
        # Try to get extra 2d keypoints, if available
        try:
            extra_keypoints_2d = self.data['extra_keypoints_2d']
        except KeyError:
            extra_keypoints_2d = np.zeros((len(self.center), 19, 3))

        self.keypoints_2d = np.concatenate((body_keypoints_2d, extra_keypoints_2d), axis=1).astype(np.float32)

        # Try to get 3d keypoints, if available
        try:
            body_keypoints_3d = self.data['body_keypoints_3d'].astype(np.float32)
        except KeyError:
            body_keypoints_3d = np.zeros((len(self.center), 25, 4), dtype=np.float32)
        # Try to get extra 3d keypoints, if available
        try:
            extra_keypoints_3d = self.data['extra_keypoints_3d'].astype(np.float32)
        except KeyError:
            extra_keypoints_3d = np.zeros((len(self.center), 19, 4), dtype=np.float32)

        body_keypoints_3d[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], -1] = 0

        self.keypoints_3d = np.concatenate((body_keypoints_3d, extra_keypoints_3d), axis=1).astype(np.float32)
        self.normalize_img = Normalize(mean=cfg.policy.img_mean, std=cfg.policy.img_std)

    def __len__(self) -> int:
        return len(self.scale)

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns an example from the dataset.
        """
        try:
            image_file_rel = self.imgname[idx].decode('utf-8')
        except AttributeError:
            image_file_rel = self.imgname[idx]
        image_file = os.path.join(self.img_dir, image_file_rel)
        keypoints_2d = self.keypoints_2d[idx].copy()
        keypoints_3d = self.keypoints_3d[idx].copy()

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]
        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.policy.bbox_shape
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        bbox_expand_factor = bbox_size / ((scale*200).max())
        body_pose = self.body_pose[idx].copy().astype(np.float32)
        betas = self.betas[idx].copy().astype(np.float32)
        # trans = self.trans[idx].copy().astype(np.float32)

        has_body_pose = self.has_body_pose[idx].copy()
        has_betas = self.has_betas[idx].copy()

        smpl_params = {'global_orient': body_pose[:3],
                       'body_pose': body_pose[3:],
                       'betas': betas,
                    #    'trans': trans,
                      }

        has_smpl_params = {'global_orient': has_body_pose,
                           'body_pose': has_body_pose,
                           'betas': has_betas
                           }

        smpl_params_is_axis_angle = {'global_orient': True,
                                     'body_pose': True,
                                     'betas': False
                                    }

        augm_config = copy.deepcopy(self.cfg.datasets.config)
        # Crop image and (possibly) perform data augmentation

        cv_img, img_patch_cv, img_patch, keypoints_2d, keypoints_3d, smpl_params, has_smpl_params, img_size, trans, augm_record = get_example_hmr2(image_file,
                                                                                                    center_x, center_y,
                                                                                                    bbox_size, bbox_size,
                                                                                                    keypoints_2d, keypoints_3d,
                                                                                                    smpl_params, has_smpl_params,
                                                                                                    FLIP_KEYPOINT_PERMUTATION, # keypoint flip
                                                                                                    self.img_size, self.img_size,
                                                                                                    self.mean, self.std, self.is_train, augm_config,
                                                                                                    return_trans=True)
        # not cropped, just full resized image
        _, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                        (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())
        
       
        item = {}
        # These are the keypoints in the original image coordinates (before cropping)
        orig_keypoints_2d = self.keypoints_2d[idx].copy()

        item['img'] = img_patch
        item['img_vis'] = img_patch_cv
        item['keypoints_2d'] = keypoints_2d.astype(np.float32)
        item['keypoints_3d'] = keypoints_3d.astype(np.float32)
        item['orig_keypoints_2d'] = orig_keypoints_2d
        item['box_center'] = self.center[idx].copy()
        item['bbox_size'] = bbox_size
        item['bbox_expand_factor'] = bbox_expand_factor
        item['img_size'] = 1.0 * img_size.copy() # img_height, img_width
        item['smpl_params'] = smpl_params
        item['has_smpl_params'] = has_smpl_params
        item['smpl_params_is_axis_angle'] = smpl_params_is_axis_angle
        item['imgname'] = image_file
        item['imgname_rel'] = image_file_rel
        item['personid'] = int(self.personid[idx])
        item['extra_info'] = copy.deepcopy(self.extra_info[idx])
        item['idx'] = idx
        item['_scale'] = scale
        item['img_full_resized'] = img_full_resized 
        item['_trans'] = trans  
        # item['augm_record'] = augm_config  # Augmentation record for recovery in self-improvement process.
        return item


