import torch
import numpy as np
from skimage.filters import gaussian
import random
import cv2
from typing import List, Dict, Tuple
from yacs.config import CfgNode
from typing import Union

from util.pylogger import get_pylogger

logger = get_pylogger(__name__)

def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """
     Increase the size of bounding box to match the target shape.
    """
    if target_aspect_ratio is None:
        return input_shape
    
    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape
    
    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h 
        w_new = max(w, h*w_t/h_t)
    
    assert h_new >= h and w_new >= w, "expanded size should not be smaller"
    return np.array([w_new, h_new])
    
    
def do_augmentation(aug_config: CfgNode) -> Tuple:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        scale (float): Box rescaling factor.
        rot (float): Random image rotation.
        do_flip (bool): Whether to flip image or not.
        do_extreme_crop (bool): Whether to apply extreme cropping (as proposed in EFT).
        color_scale (List): Color rescaling factor
        tx (float): Random translation along the x axis.
        ty (float): Random translation along the y axis. 
    """
    # (-1, 1)
    tx = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.trans_factor
    ty = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.trans_factor
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.scale_factor + 1.0

    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * aug_config.rot_factor if random.random() <= aug_config.rot_aug_rate else 0
    
    do_flip = aug_config.do_flip and random.random() <= aug_config.flip_aug_rate
    do_extreme_crop = random.random() <= aug_config.extreme_crop_aug_rate

    extreme_crop_lvl = aug_config.get('extreme_crop_aug_level', 0)
    # extreme_crop_lvl = 1
    # （0.8, 1.2)
    c_up = 1.0 + aug_config.color_scale
    c_low = 1.0 - aug_config.color_scale
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]
    return scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty

def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    """
    Rotate a 2D point on the x-y plane.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        rot_rad (float): Rotation angle
    Returns:
        np.array: Rotated 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.array:
    """
    Create transformation matrix for the bounding box crop.
    Args:
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        src_width (float): Bounding box width.
        src_height (float): Bounding box height.
        dst_width (float): Output box width.
        dst_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        trans (np.array): Target geometric transformation.
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_points2d_parallel(keypoints_2d: np.array, trans:np.array):

    ones = np.ones((keypoints_2d.shape[0], 1))
    keypoints_augmented = np.hstack([keypoints_2d, ones])
    
    transformed_keypoints = np.dot(trans, keypoints_augmented.T).T
    
    # Extract the transformed 2D points by discarding the last row
    transformed_keypoints_2d = transformed_keypoints[:, :2]  
    return transformed_keypoints_2d



def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w , h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])


def generate_image_patch_cv2(img: np.array, c_x: float, c_y: float,
                             bb_width: float, bb_height: float,
                             patch_width: float, patch_height: float,
                             do_flip: bool, scale: float, rot: float,
                             border_mode=cv2.BORDER_CONSTANT, border_value=0) -> Tuple[np.array, np.array]:
    """
    Crop the input image and return the crop and the corresponding transformation matrix.
    Args:
        img (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        do_flip (bool): Whether to flip image or not.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        img_patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans (np.array): Transformation matrix.
    """

    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1


    trans = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), 
                        flags=cv2.INTER_LINEAR, 
                        borderMode=border_mode,
                        borderValue=border_value,
                )
    # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
    if (img.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        img_patch[:,:,3] = cv2.warpAffine(img[:,:,3], trans, (int(patch_width), int(patch_height)), 
                                            flags=cv2.INTER_LINEAR, 
                                            borderMode=cv2.BORDER_CONSTANT,
                            )

    return img_patch, trans

def convert_cvimg_to_tensor(cvimg: np.array):
    """
    Convert image from HWC to CHW format.
    Args:
        cvimg (np.array): Image of shape (H, W, 3) as loaded by OpenCV.
    Returns:
        np.array: Output image of shape (3, H, W).
    """
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    img = img.astype(np.float32)
    return img

def get_example(
        img_path: Union[str, np.ndarray],
        center_x: float,
        center_y: float,
        width: float,
        height: float,
        keypoints_2d: np.array,
        patch_width: int,
        patch_height: int,
        mean: np.array,
        std: np.array,
        do_augment: bool,
        augm_config: CfgNode,
        is_bgr: bool = True,
        use_skimage_antialias: bool = False,
        border_mode: int = cv2.BORDER_CONSTANT,
        return_trans: bool = False,
        ds_name: str = None):
    
    # 1. load the img which should be str path or np.ndarray
    if isinstance(img_path, str):
        cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # closeup and portrait images are rotated 90 degrees clockwise
        if ds_name is not None and ('closeup' in ds_name or 'portrait' in ds_name):
            cvimg = cv2.rotate(cvimg, cv2.ROTATE_90_CLOCKWISE)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % img_path)  
    elif isinstance(img_path, np.ndarray):
        cvimg = img_path
    else:
        raise TypeError('img_path must be either a string or a numpy array')
    
    # 2. get image size and augmentation parameters
    img_height, img_width, img_channels = cvimg.shape
    img_size = np.array([img_height, img_width])
    if do_augment:
        scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = do_augmentation(augm_config)
    else:
        scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = 1.0, 0, False, False, 0, [1.0, 1.0, 1.0], 0., 0.

    if width < 1 or height < 1:
        raise ValueError(f"Invalid width {width} or height {height} for the bounding box.")
    
    if do_extreme_crop:
        if extreme_crop_lvl == 0:
            center_x1, center_y1, width1, height1 = extreme_cropping(center_x, center_y, width, height, keypoints_2d)
        elif extreme_crop_lvl == 1:
            center_x1, center_y1, width1, height1 = extreme_cropping_aggressive(center_x, center_y, width, height, keypoints_2d)

        THRESH = 4
        if width1 < THRESH or height1 < THRESH:
            pass
        else:
            center_x, center_y, width, height = center_x1, center_y1, width1, height1
    
    center_x += width * tx
    center_y += height * ty

    # 3. generate image patch
    if use_skimage_antialias:
        # Blur image to avoid aliasing artifacts
        downsampling_factor = (patch_width / (width*scale))
        if downsampling_factor > 1.1:
            cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True, truncate=3.0)

    if do_augment and augm_config.use_alb:
        import albumentations as A
        aug_comp = [A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
                    A.ImageCompression(20, 100, p=0.1),
                    A.RandomRain(blur_value=4, p=0.1),
                    A.MotionBlur(blur_limit=(3, 15),  p=0.2),
                    A.Blur(blur_limit=(3, 10), p=0.1),
                    A.RandomSnow(brightness_coeff=1.5,
                    snow_point_lower=0.2, snow_point_upper=0.4)]
        aug_mod = [A.CLAHE((1, 11), (10, 10), p=0.2), A.ToGray(p=0.2),
                   A.RandomBrightnessContrast(p=0.2),
                   A.MultiplicativeNoise(multiplier=[0.5, 1.5],
                   elementwise=True, per_channel=True, p=0.2),
                   A.HueSaturationValue(hue_shift_limit=20,
                   sat_shift_limit=30, val_shift_limit=20,
                   always_apply=False, p=0.2),
                   A.Posterize(p=0.1),
                   A.RandomGamma(gamma_limit=(80, 200), p=0.1),
                   A.Equalize(mode='cv', p=0.1)]
        albumentation_aug = A.Compose([
            A.OneOf(aug_comp, p=augm_config.alb_prob),
            A.OneOf(aug_mod, p=augm_config.alb_prob)
            ])          
        cvimg = albumentation_aug(image=cvimg)['image']
    
    img_patch_cv, trans = generate_image_patch_cv2( cvimg,
                                                    center_x, center_y,
                                                    width, height,
                                                    patch_width, patch_height,
                                                    do_flip, scale, rot, 
                                                    border_mode=border_mode)
    
    image = img_patch_cv.copy()

    if is_bgr:
        image = image[:, :, ::-1]
    img_patch_cv = image.copy()
    img_patch = convert_cvimg_to_tensor(image)

    for n_c in range(min(img_channels, 3)):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]
    
    keypoints_2d[:, :2] = trans_points2d_parallel(keypoints_2d[:, 0:2], trans)
    keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5
    # center_x, center_y, width, height consturcts the bbox 
    if not return_trans:
        return img_patch, img_patch_cv, keypoints_2d, img_size, center_x, center_y, width, height, scale
    else:
        return img_patch, img_patch_cv, keypoints_2d, img_size, center_x, center_y, width, height, trans, scale

def crop_to_hips(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Extreme cropping: Crop the box up to the hip locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24, 25+0, 25+1, 25+4, 25+5]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_to_shoulders(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box up to the shoulder locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    center, scale = get_bbox(keypoints_2d)
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.2 * scale[0]
        height = 1.2 * scale[1]
    return center_x, center_y, width, height

def crop_to_head(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the head.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.3 * scale[0]
        height = 1.3 * scale[1]
    return center_x, center_y, width, height

def crop_torso_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the torso.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nontorso_body_keypoints = [0, 3, 4, 6, 7, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 4, 5, 6, 7, 10, 11, 13, 17, 18]]
    keypoints_2d[nontorso_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_rightarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the right arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightarm_body_keypoints = [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_leftarm_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the left arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftarm_body_keypoints = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_legs_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the legs.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonlegs_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18] + [25 + i for i in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]]
    keypoints_2d[nonlegs_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_rightleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the right leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightleg_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21] + [25 + i for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def crop_leftleg_only(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array):
    """
    Extreme cropping: Crop the box and keep on only the left leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftleg_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16, 17, 18, 22, 23, 24] + [25 + i for i in [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height

def full_body(keypoints_2d: np.array) -> bool:
    """
    Check if all main body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """

    body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
    body_keypoints = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
    return (np.maximum(keypoints_2d[body_keypoints, -1], keypoints_2d[body_keypoints_openpose, -1]) > 0).sum() == len(body_keypoints)

def upper_body(keypoints_2d: np.array):
    """
    Check if all upper body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """
    lower_body_keypoints_openpose = [10, 11, 13, 14]
    lower_body_keypoints = [25 + i for i in [1, 0, 4, 5]]
    upper_body_keypoints_openpose = [0, 1, 15, 16, 17, 18]
    upper_body_keypoints = [25+8, 25+9, 25+12, 25+13, 25+17, 25+18]
    return ((keypoints_2d[lower_body_keypoints + lower_body_keypoints_openpose, -1] > 0).sum() == 0)\
       and ((keypoints_2d[upper_body_keypoints + upper_body_keypoints_openpose, -1] > 0).sum() >= 2)

def get_bbox(keypoints_2d: np.array, rescale: float = 1.2) -> Tuple:
    """
    Get center and scale for bounding box from openpose detections.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center (np.array): Array of shape (2,) containing the new bounding box center.
        scale (float): New bounding box scale.
    """
    valid = keypoints_2d[:,-1] > 0
    valid_keypoints = keypoints_2d[valid][:,:-1]
    center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
    bbox_size = (valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0))
    # adjust bounding box tightness
    scale = bbox_size
    scale *= rescale
    return center, scale

def extreme_cropping(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Perform extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """

    p = torch.rand(1).item()
    # 0.7 probs crop to hip, 0.2 crops to shoulder, 0.1 crops to head
    if full_body(keypoints_2d):
        if p < 0.7:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)

    return center_x, center_y, max(width, height), max(width, height)

def extreme_cropping_aggressive(center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array) -> Tuple:
    """
    Perform aggressive extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_hips(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.3:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.5:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.6:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.7:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_legs_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.9:
            center_x, center_y, width, height = crop_rightleg_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftleg_only(center_x, center_y, width, height, keypoints_2d)
    elif upper_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_shoulders(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.6:
            center_x, center_y, width, height = crop_torso_only(center_x, center_y, width, height, keypoints_2d)
        elif p < 0.8:
            center_x, center_y, width, height = crop_rightarm_only(center_x, center_y, width, height, keypoints_2d)
        else:
            center_x, center_y, width, height = crop_leftarm_only(center_x, center_y, width, height, keypoints_2d)
    return center_x, center_y, max(width, height), max(width, height)