from typing import Dict
import cv2
import os
import json
import torch
from tqdm import tqdm
import trimesh
import numpy as np
from glob import glob
from torchvision.transforms import Normalize
from detectron2.config import LazyConfig
from body_models.skel_utils.transforms import params_rep2q
from datasets.constants import DETECTRON_CFG
from models.cam_model.fl_net import FLNet
from util.constants import CAM_MODEL_CKPT, DETECTRON_CKPT, IMAGE_MEAN, IMAGE_STD
from vis.dataset import Dataset
from vis.renderer_vis import Renderer
from util.data import recursive_to
from body_models.skel_wrapper import SKELWrapper
from util.utils_detectron2 import DefaultPredictor_Lazy
from vis.skelvit_render import SKELViTRender
from vis.color import ColorPalette


def fix_prefix_state_dict(st: Dict):
    new_st = {}
    for k, v in st.items():
        if k.startswith('model.'):
            new_k = k.replace('model.', '', 1)
            new_st[new_k] = v
    return new_st




def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y:start_y + new_height, start_x:start_x + new_width] = resized_img

    return aspect_ratio, final_img

class HumanMeshEstimator:
    def __init__(self, cfg, threshold=0.25):
        self.cfg = cfg 
        self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.model = self.init_model()
        self.detector = self.init_detector(threshold)
        self.cam_model = self.init_cam_model()
        self.normalize_img = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)
        self.skel_model = SKELWrapper(**cfg.hub.body_models.skel_mix_hsmr).to(self.device)

    def init_cam_model(self):
        model = FLNet()
        checkpoint = torch.load(CAM_MODEL_CKPT, map_location='cpu')['state_dict']
        model.load_state_dict(checkpoint)
        # model = model.to(self.device)
        model.eval()
        return model

    def init_model(self):
        model = SKELViTRender(self.cfg)
        checkpoint = torch.load(self.cfg.trainer.ckpt_path, map_location='cpu')['ema_model']
        st = fix_prefix_state_dict(checkpoint)
        model.load_state_dict(st, strict=True)
        model = model.to(self.device)
        model.eval()
        return model
    
    def init_detector(self, threshold):

        detectron2_cfg = LazyConfig.load(str(DETECTRON_CFG))
        detectron2_cfg.train.init_checkpoint = DETECTRON_CKPT
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = threshold
        detector = DefaultPredictor_Lazy(detectron2_cfg)
        return detector

    
    def convert_to_full_img_cam(self, pare_cam, bbox_height, bbox_center, img_w, img_h, focal_length):
        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2. * focal_length / (bbox_height * s)
        cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_output_mesh(self, params, pred_cam, batch):

        skel_output = self.skel_model(**{k: v.float() for k, v in params.items()}, skelmesh=True)
        # pred_keypoints_3d = skel_output.joints
        pred_vertices = skel_output.skin_verts
        pred_skel_verts = skel_output.skel_verts

        img_h, img_w = batch['img_size'][0]
        cam_trans = self.convert_to_full_img_cam(
            pare_cam=pred_cam,
            bbox_height=batch['bbox_size'],
            bbox_center=batch['box_center'],
            img_w=img_w,
            img_h=img_h,
            focal_length=batch['cam_int'][:, 0, 0]
        )
        return pred_vertices, pred_skel_verts, cam_trans

    def get_cam_intrinsics(self, img):
        img_h, img_w, c = img.shape
        aspect_ratio, img_full_resized = resize_image(img, 256)
        img_full_resized = np.transpose(img_full_resized.astype('float32'),
                            (2, 0, 1))/255.0
        img_full_resized = self.normalize_img(torch.from_numpy(img_full_resized).float())

        estimated_fov, _ = self.cam_model(img_full_resized.unsqueeze(0))
        vfov = estimated_fov[0, 1]
        fl_h = (img_h / (2 * torch.tan(vfov / 2))).item()
        # fl_h = (img_w * img_w + img_h * img_h) ** 0.5
        cam_int = np.array([[fl_h, 0, img_w/2], [0, fl_h, img_h / 2], [0, 0, 1]]).astype(np.float32)
        return cam_int


    def process_image(self, img_path, output_img_folder):
        img_cv2 = cv2.imread(str(img_path))
        
        fname, img_ext = os.path.splitext(os.path.basename(img_path))

        # Detect humans in the image
        det_out = self.detector(img_cv2)
        det_instances = det_out['instances']
        
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.7)

        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        bbox_scale = (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0 
        bbox_center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


        # image h & w
        img_h, img_w = img_cv2.shape[:2]

        # bbox center 
        center_x = bbox_center[:, 0]
        center_y = bbox_center[:, 1]

        center_mask = (
            (center_x > 0.1 * img_w) & (center_x < 0.9 * img_w) &
            (center_y > 0.1 * img_h) & (center_y < 0.9 * img_h)
        )
        # keep center area boxes
        if center_mask.sum() > 0 :
            areas = areas[center_mask]
            boxes = boxes[center_mask]
            bbox_scale = bbox_scale[center_mask]
            bbox_center = bbox_center[center_mask]
        
        num_keep = min(self.cfg.misc.num_keep, len(areas))
        keep = areas.argsort()[-num_keep:][::-1]
        boxes = boxes[keep]
        bbox_scale = bbox_scale[keep]
        bbox_center = bbox_center[keep]
        # Get Camera intrinsics using HumanFoV Model
        cam_int = self.get_cam_intrinsics(img_cv2)
        dataset = Dataset(cfg=self.cfg, img_cv2=img_cv2, bbox_center=bbox_center, 
                            bbox_scale=bbox_scale, cam_int=cam_int, train=False, img_path=img_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=10)

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            img_h, img_w = batch['img_size'][0]
            with torch.no_grad():
                _, dec_params, per_layer_params = self.model(batch)

            # for i in range(len(per_layer_params)):     
                # if i < 5: 
                #     continue 
            
            # poses = per_layer_params[i][f'pd_poses_{i}']
            # poses = params_rep2q(poses.reshape(-1, 24, 6)).reshape(-1, 46)
            # skel_params = {
            #     'poses': poses,
            #     'betas': per_layer_params[i][f'pd_betas_{i}']
            # }
            # out_cam = per_layer_params[i][f'pd_cam_{i}']
            skel_params = dec_params['pd_skel_params']
            out_cam = dec_params[f'pd_cam_t']
            focal_length_ = batch['cam_int'][:, 0, 0]
            # 1. get output skin & skel verts
            pred_skin_verts, pred_skel_verts, output_cam_trans = self.get_output_mesh(skel_params, out_cam, batch)
            
            focal_length = (focal_length_[0], focal_length_[0])
            # 4. add camera translation to verts
            pred_skin_verts_array = (pred_skin_verts + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            pred_skel_verts_array = (pred_skel_verts + output_cam_trans.unsqueeze(1)).detach().cpu().numpy()
            
            # obj_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_skin.obj')
            # skin_mesh = trimesh.Trimesh(vertices=pred_skin_verts[0].cpu(), faces=self.skel_model.skin_f.cpu().numpy(), process=False)
            # skin_mesh.export(obj_fname)
            
     
            # skel_obj_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_skel.obj')
            # skel_mesh = trimesh.Trimesh(vertices=pred_skel_verts[0].cpu(), faces=self.skel_model.skel_f.cpu().numpy(), process=False)
            # skel_mesh.export(skel_obj_fname)
            

            # 5. Render overlay
            renderer = Renderer(focal_length=focal_length[0], 
                img_w=img_w, img_h=img_h, same_mesh_color=True)

            
            skin_front_view = renderer.render_front_view(verts=pred_skin_verts_array, color_name='blue', 
                                    faces=self.skel_model.skin_f.cpu(), bg_img_rgb=img_cv2.copy())

            skel_front_view = renderer.render_front_view(verts=pred_skel_verts_array, color_name='human_yellow', 
                                    faces=self.skel_model.skel_f.cpu(), bg_img_rgb=img_cv2.copy())

            # skel_side_view = renderer.render_side_view(verts=pred_skel_verts_array, color_name='human_yellow', 
            #                                 faces=self.skel_model.skel_f.cpu())
                                            
            full_img_blend = cv2.addWeighted(skin_front_view, 0.7, skel_front_view, 0.3, 0)
            
        
            concat_img = np.hstack([img_cv2, skel_front_view, full_img_blend])
            
            skin_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_skin{img_ext}')
            skel_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_skel{img_ext}')
            mixed_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_mixed{img_ext}')
            concat_fname = os.path.join(output_img_folder, f'{os.path.basename(fname)}_concat{img_ext}')
        
            # Write overlay
            # cv2.imwrite(skin_fname, skin_front_view)
            cv2.imwrite(skel_fname, skel_front_view)
            cv2.imwrite(mixed_fname, full_img_blend)
            cv2.imwrite(concat_fname, concat_img)
            
            renderer.delete()


    def run_on_images(self, image_folder, out_folder):
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
        images_list = [image for ext in image_extensions for image in glob(os.path.join(image_folder, ext))]
        for ind, img_path in tqdm(enumerate(images_list), desc="Processing images", total=len(images_list)):
            self.process_image(img_path, out_folder)