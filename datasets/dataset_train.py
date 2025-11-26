import os
import warnings
import cv2
import torch
import copy
import numpy as np
from torch.utils.data import Dataset

from util.misc import resize_image
from util.pylogger import get_pylogger
from .configs import DATASET_FOLDERS, DATASET_FILES
from .constants import NUM_JOINTS, NUM_BETAS, NUM_PARAMS_SMPL
from .utils import expand_to_aspect_ratio, get_example
from torchvision.transforms import Normalize

logger = get_pylogger(__name__)

class DatasetTrainAllLabels(Dataset):
    def __init__(self, cfg, ds_name, is_train=True):
        super(DatasetTrainAllLabels, self).__init__()

        self.ds_name = ds_name
        self.is_train = is_train
        self.cfg = cfg
        self.img_size = cfg.policy.img_patch_size # 256
        self.mean = 255. * np.array(cfg.policy.img_mean) 
        self.std = 255. * np.array(cfg.policy.img_std)
        self.normalize_img = Normalize(mean=cfg.policy.img_mean, std=cfg.policy.img_std)
        self.border_mode = cv2.BORDER_CONSTANT
        self.img_dir = DATASET_FOLDERS[ds_name]
        
        self.labels = np.load(DATASET_FILES[self.is_train][self.ds_name], allow_pickle=True)
    
        self.imgname = self.labels['imgname']
        self.scale = self.labels['scale'] # (N,) or (N,2)
        self.center = self.labels['center']  # (N,) or (N,2)

        # # load smpl poses 
        # if 'poses' in self.labels:
        #     self.smpl_poses = self.labels['poses']
        # else:
        #     raise ValueError('Unknown smpl poses key')
        
        # # load smpl betas
        # if 'betas' in self.labels:
        #     self.smpl_betas = self.labels['betas']
        # else:
        #     raise ValueError('Unknown smpl betas key')

        # 1. load skel poses 
        if 'skel_poses' in self.labels:
            self.skel_poses = self.labels['skel_poses']
        elif 'orig_poses' in self.labels:
            self.skel_poses = self.labels['orig_poses']
        else:
            raise ValueError('Unknown skel poses key')
        
        # 2. load skel betas
        if 'skel_betas' in self.labels:
            self.skel_betas = self.labels['skel_betas'][:, :10]
        elif 'orig_betas' in self.labels:
            self.skel_betas = self.labels['orig_betas'][:, :10]
        else: 
            raise ValueError('Unknown skel betas key')
        
        # 2dkp
        if 'gtkps' in self.labels:
            self.kp2d = self.labels['gtkps'][:,:NUM_JOINTS]
        elif 'keypoints_2d' in self.labels:
            self.kp2d = self.labels['keypoints_2d'][:,:NUM_JOINTS]
        else:
            raise ValueError('Unknown kp2d key')
        
        # 3dkp
        if 'keypoints3d' in self.labels:
            self.kp3d = self.labels['keypoints3d']
        elif 'keypoints_3d' in self.labels:
            self.kp3d = self.labels['keypoints_3d']
        else:
            raise ValueError('Unknown kped key')

        # Add confience for bedlam dataset
        if self.kp3d is not None and self.kp3d.shape[2] < 4:
            self.kp3d = np.concatenate((self.kp3d, np.ones((self.kp3d.shape[0], self.kp3d.shape[1], 1))), axis=2)
            logger.info(f'Add confience, kp3d shape: {self.kp3d.shape}')

        # cam int
        if 'cam_int' in self.labels:
            self.cam_int = self.labels['cam_int']
        else:
            raise ValueError('the cam_int is None')
        
        self.length = self.labels['scale'].shape[0]
        logger.info(f'Loaded {self.ds_name} dataset, num samples {self.length}')
         

    def prepocess_path(self, img_dir, imgname):
      
        if 'insta' in self.ds_name:
            parts = imgname.split(os.sep)
            segments = parts[2].split('_')
            parts[2] = f"{segments[0]}_{segments[1][:2]}_{segments[2][:2]}_{segments[-1]}"
            name, ext = os.path.splitext(parts[-1])
            parts[-1] = f"{int(name):04d}{ext}"
            imgname = os.sep.join(parts)
            imgname = os.path.join(img_dir, imgname.replace('insta-train/',"", 1))
        elif 'aic' in self.ds_name:
            imgname = imgname.replace('aic-train/',"", 1)
            imgname = os.path.join(img_dir, imgname)
        elif 'mpii' in self.ds_name or 'coco' in self.ds_name:
            imgname = os.path.basename(imgname)
            imgname = os.path.join(img_dir, imgname)
        elif 'h36m' in self.ds_name or 'mpi' in self.ds_name:
            imgname = imgname + '.jpg'
            imgname = os.path.join(img_dir, imgname)
        else:
            # bedlam & agora dataset
            imgname = imgname
            imgname = os.path.join(img_dir, imgname)

        return imgname 

    def __getitem__(self, index):
        item = {}
        
        imgname = self.imgname[index] # (,)
        scale = self.scale[index]  # (N,) or (N,2)
        center = self.center[index] # (N,) or (N,2)
        kp2d = self.kp2d[index][:,:NUM_JOINTS]
        kp3d = self.kp3d[index]

        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=None).max()

        if np.any(bbox_size < 1):
            raise ValueError(f"Invalid bbox size: {bbox_size}. It must be >= 1")
            
        augm_config = copy.deepcopy(self.cfg.datasets.config)

        imgname = self.prepocess_path(self.img_dir, str(imgname))
        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv_img[:, :, ::-1]

        _, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                        (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        img_patch_rgba, \
        _, keypoints_2d, img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug = get_example(
                                      img_path=imgname,
                                      center_x=center[0], center_y=center[1],
                                      # from scale 
                                      width=bbox_size, height=bbox_size,
                                      keypoints_2d=kp2d,
                                      # patch size
                                      patch_width=self.img_size, patch_height=self.img_size,
                                      mean=self.mean, std=self.std, 
                                      do_augment=self.is_train, augm_config=augm_config,
                                      is_bgr=True, return_trans=True,
                                      use_skimage_antialias=False,
                                      border_mode = self.border_mode,
                                      ds_name = self.ds_name
                            )
        
        item.update({
            'img': img_patch_rgba[:3],  # remove alpha
            'kp2d': keypoints_2d.astype(np.float32),
            'orig_kp2d': kp2d.astype(np.float32),
            'kp3d': kp3d.astype(np.float32),
            'box_center': np.array([cx, cy]),
            'bbox_size' : bbox_w * scale_aug,
            'img_size': np.array([img_size[0], img_size[1]]), # img_full_size, img_height, img_width
            'skel_poses': self.skel_poses[index],
            'skel_betas': self.skel_betas[index],
            # 'smpl_poses': self.smpl_poses[index],
            # 'smpl_betas': self.smpl_betas[index],
            'ds_name': self.ds_name,
            'cam_int' : np.array(self.cam_int[index]).astype(np.float32),
            'img_full_resized' : img_full_resized,
            '_trans' : trans
        })

        return item

    def __len__(self):
        return self.length
