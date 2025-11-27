import numpy as np 
import cv2
import smplx 
import os 
import pickle
import copy 
import torch 
from torchvision.transforms import Normalize
from torch.utils.data import Dataset

from datasets.constants import NUM_PARAMS_SMPL, NUM_BETAS, NUM_JOINTS, SMPL_MODEL_DIR
from datasets.utils import expand_to_aspect_ratio, get_example
from util.misc import resize_image
from util.pylogger import get_pylogger
from .configs import DATASET_FOLDERS, DATASET_FILES

logger = get_pylogger(__name__)

class DatasetVal(Dataset):
    def __init__(self, cfg, ds_name, is_train=False):
        super(DatasetVal, self).__init__()

        self.ds_name = ds_name
        self.is_train = False
        self.cfg = cfg
        self.img_size = cfg.policy.img_patch_size
        self.bbox_shape = cfg.policy.get('bbox_shape', None)
        
        self.mean = 255. * np.array(cfg.policy.img_mean)
        self.std = 255. * np.array(cfg.policy.img_std)
        self.normalize_img = Normalize(mean=cfg.policy.img_mean, std=cfg.policy.img_std)
        self.use_skimage_antialias = False
        self.border_mode = cv2.BORDER_CONSTANT

        self.img_dir = DATASET_FOLDERS[ds_name] 
        # Load label
        self.labels = np.load(DATASET_FILES[is_train][ds_name], allow_pickle=True)
        self.length = self.labels['scale'].shape[0]               
        # === Basic attributes ===
        self.imgname = self.labels['img_paths'] if 'moyo' in ds_name else self.labels['imgname']
        self.scale = self.labels['scales'] if 'moyo' in ds_name else self.labels['scale']
        self.center = self.labels['centers'] if 'moyo' in ds_name else self.labels['center']
       
        # === Pose ===
        if 'pose_cam' in self.labels:
            # smpl
            num_params = NUM_PARAMS_SMPL
            self.pose = self.labels['pose_cam'][:, :num_params * 3].astype(np.float32)
        elif 'smpl' in self.labels:
            smpl_st = self.labels['smpl'].item()
            body_poses = smpl_st['body_pose'].astype(np.float32)
            global_orient = smpl_st['global_orient'].astype(np.float32)
            self.pose = np.concatenate((global_orient, body_poses), axis=-1)

         # === 2D Keypoints ===
        if 'part' in self.labels:
            self.keypoints = self.labels['part']
        elif 'gtkps' in self.labels:
            self.keypoints = self.labels['gtkps'][:, :NUM_JOINTS]  # 44 joints
        elif 'body_keypoints_2d' in self.labels:
            self.keypoints = self.labels['body_keypoints_2d']
        else:
            raise ValueError('keypoints not Found, set to zero')

        # === Shape (Betas) ===
        if 'shape' in self.labels:
            self.betas = self.labels['shape'].astype(np.float32)[:,:NUM_BETAS] 
        elif 'betas' in self.labels:
            self.betas = self.labels['betas'].astype(np.float32)[:,:NUM_BETAS]
        elif 'smpl' in self.labels:
            smpl_st = self.labels['smpl'].item()
            self.betas = smpl_st['betas'].astype(np.float32)

        # === Gender ===
        try:
            gender = self.labels['gender']
            self.gender = np.array([0 if str(g) == 'm' or str(g)=='male' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1 * np.ones(len(self.imgname)).astype(np.int32)

        # === SMPL and SMPL-X models ===
        self.smpl_gt_male = smplx.SMPL(SMPL_MODEL_DIR,
                                gender='male')
        self.smpl_gt_female = smplx.SMPL(SMPL_MODEL_DIR,
                                    gender='female')
        self.smpl_gt_neutral = smplx.SMPL(SMPL_MODEL_DIR,
                                    gender='neutral')
              
        logger.info(f'Loaded {self.ds_name} dataset, num samples {self.length}')

    def __getitem__(self, index):
        item = {}

        # === Extract metadata ===
        scale = self.scale[index]
        center = self.center[index]
        keypoints_2d = self.keypoints[index]
        center_x = center[0]
        center_y = center[1]
        # === Compute bounding box size ===
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=self.bbox_shape).max()

        if np.any(bbox_size < 1):
            raise ValueError(f"Invalid bbox size: {bbox_size}. It must be >= 1")
        
        # === Load image and convert BGR to RGB ===
        if 'h36m' in self.ds_name:
            imgname =  os.path.join(self.img_dir, self.imgname[index].decode())
        else:
            imgname = os.path.join(self.img_dir, self.imgname[index])

        cv_img = cv2.imread(imgname, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        cv_img = cv_img[:, :, ::-1].copy()
        aspect_ratio, img_full_resized = resize_image(cv_img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                        (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())
        # === Load augmentation configuration ===
        augm_config = copy.deepcopy(self.cfg.datasets.config)

        # === Generate training patch === 
        img_patch_rgba, img_patch_cv, keypoints_2d, \
        img_size, cx, cy, bbox_w, bbox_h, trans, scale_aug = get_example(
                                      img_path = imgname,
                                      center_x = center_x, center_y = center_y,
                                      width = bbox_size, height = bbox_size,
                                      keypoints_2d = keypoints_2d,
                                      patch_width = self.img_size, patch_height = self.img_size,
                                      mean = self.mean, std = self.std, 
                                      do_augment = self.is_train, 
                                      augm_config = augm_config,
                                      is_bgr = True, 
                                      return_trans = True,
                                      use_skimage_antialias = self.use_skimage_antialias,
                                      border_mode = self.border_mode,
                                      ds_name = self.ds_name
                            )

        # === Camera intrinsics ===
        img_h, img_w = img_size[0], img_size[1]

        # === SMPL parameter extraction ===
        pose = self.pose[index].astype(np.float32)
        betas = self.betas[index].astype(np.float32)

        smpl_params = {
            'global_orient': pose[:3],
            'body_pose': pose[3:],
            'betas': betas
        }
      
        # === Load SMPL model based on gender ===
        gender_str = {0: 'male', 1: 'female'}.get(self.gender[index], 'neutral')
        body_model = None
       
        if 'smpl' in self.ds_name:
            if gender_str == 'neutral':
                body_model = self.smpl_gt_neutral
            elif gender_str == 'male':
                body_model = self.smpl_gt_male
            else:
                body_model = self.smpl_gt_female
        
        # === Forward the model with given parameters ===
        if body_model is not None:
            gt_output = body_model(
                global_orient=torch.from_numpy(smpl_params['global_orient']).unsqueeze(0),
                body_pose=torch.from_numpy(smpl_params['body_pose']).unsqueeze(0),
                betas=torch.from_numpy(smpl_params['betas']).unsqueeze(0)
            )
            gt_vertices = gt_output.vertices.detach()  # (1, 6890, 3)

        # === Image and annotation info ===
        item.update({
            'img': img_patch_rgba[:3],  # remove alpha
            'kp2d': keypoints_2d.astype(np.float32),
            'box_center': np.array([cx, cy]),
            'bbox_size' : bbox_w * scale_aug,
            'img_size': np.array([img_h, img_w]),
            'smpl_params': smpl_params,
            'vertices': gt_vertices[0].float(),
            'imgname': imgname,
            'gender': self.gender[index],
            'ds_name': self.ds_name,
            'img_full_resized': img_full_resized
        })
            
        return item

    def __len__(self):
        return int(len(self.imgname))
        

