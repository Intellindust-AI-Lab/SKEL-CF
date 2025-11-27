import os
import time
import cv2
import torch 
import numpy as np
from torchvision.transforms import Normalize
from datasets.utils import convert_cvimg_to_tensor, expand_to_aspect_ratio, generate_image_patch_cv2
from util.constants import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD
from util.pylogger import get_pylogger



logger = get_pylogger(__name__)

class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        cfg, 
        img_cv2: np.array,
        bbox_center: np.array,
        bbox_scale: np.array,
        cam_int: np.array = None,
        train: bool = False,
        img_path = None,
        **kwargs
    ):
        
        super().__init__()
        self.cfg = cfg 
        self.img_cv2 = img_cv2
        self.img_path = img_path

        assert train == False, "ViTDetDataset is only for inference"

        self.train = train
        if cam_int is not None:
            self.cam_int = cam_int
        else:
            self.cam_int = np.array([])
            logger.info("Camera intrinsics not provided, using default values")

        self.mean = 255. * np.array(IMAGE_MEAN)
        self.std = 255. * np.array(IMAGE_STD)
        # self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        self.center = bbox_center
        self.scale = bbox_scale
        self.personid = np.arange(len(self.center), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.personid)
    
    def __getitem__(self, idx: int):
        center = self.center[idx]
        center_x = center[0]
        center_y = center[1]
        
        scale = self.scale[idx]
        BBOX_SHAPE = None
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()

        patch_width = patch_height = 256 # 256
        cvimg = self.img_cv2
        
        img_patch_cv, trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    False, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        
        save_debug_patch = False
        if save_debug_patch:
            timestamp = time.strftime("%H%M%S") 
            debug_filename = f'debug_patch_{idx}_{timestamp}.jpg'

            debug_path = os.path.join(self.cfg.misc.output_folder, debug_filename)
            cv2.imwrite(debug_path, img_patch_cv)
            print(f"Saved debug patch to {debug_path}")
        
        img_patch = convert_cvimg_to_tensor(img_patch_cv[:, :, ::-1])
        
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]
        # img_patch = self.normalize_img(torch.tensor(img_patch))

        item = {
            'img' : img_patch,
            'personid': int(self.personid[idx]),
            'imgname': str(self.img_path),
            'box_center': center,
            'bbox_size': bbox_size,
            'img_size': 1.0*np.array([cvimg.shape[0], cvimg.shape[1]]),
            'cam_int': self.cam_int,
            'img_vis' : img_patch_cv
        }
        return item 
