from typing import Dict
import torch 
from torch import nn
from omegaconf import DictConfig
from body_models.skel_wrapper import SKELWrapper
from models.backbones.vit import ViT
from models.cam_model.fl_net import FLNet
from models.heads.skel_decoder_base import SKELTransformerDecoderHeadBase
from models.losses.losses import HPE_Loss
from util.constants import CAM_MODEL, DINOV3_DIR, DINOV3B_BACKBONE, DINOV3H_BACKBONE, DINOV3L_BACKBONE, VITPOSE_BACKBONE, VITPOSEB_BACKBONE
from util.geometry import cam_perspective_projection
from util.pylogger import get_pylogger

logger = get_pylogger(__name__)



def fix_prefix_state_dict_vitposeb(st: Dict):
    new_st = {}
    for k, v in st.items():
        if k.startswith('backbone.'):
            new_k = k.replace('backbone.', '', 1)
            new_st[new_k] = v

    for k, v in new_st.items():
        logger.info(f"key: {k}, value: {v.shape}")

    return new_st


class SKELViT(nn.Module):

    def __init__(self, cfg:DictConfig):
        super(SKELViT, self).__init__()
        # SKEL_wrapper 
        
        self.cfg = cfg
        self.backbone = self.setup_backbone()
        # setup the decoder, take pose_init after encoder as input
        self.decoder = self.setup_head()
        self.cam_model = self.setup_cam_model()
        self.skel_model = SKELWrapper(**cfg.hub.body_models.skel_mix_hsmr)

    def setup_cam_model(self):

        ckpt = torch.load(CAM_MODEL, map_location='cpu', weights_only=False)
        cam_model = FLNet()
        cam_model.load_state_dict(ckpt['state_dict'], strict=True)

        for p in cam_model.parameters():
            p.requires_grad = False
            
        cam_model.eval()
        return cam_model

    def setup_backbone(self):
       
        backbone = ViT(**self.cfg.hub.backbones.vit_h)
        ckpt = torch.load(VITPOSE_BACKBONE, map_location='cpu')
        backbone.load_state_dict(ckpt['state_dict'], strict=True)
        logger.info('Vitpose-H initialized !')
        
        return backbone


    def setup_head(self):
       
        skel_head = SKELTransformerDecoderHeadBase(cfg=self.cfg)
        logger.info('SKEL Transformer decoder head initialized !')

        return skel_head

    def get_trainable_parameters(self):
        return list(self.backbone.parameters()) + list(self.decoder.parameters())

    def forward_step(self, x, batch):
      
        B = x.shape[0]
        # B, 192, 1280
        x_norm_patch = self.backbone(x[:, :, :, 32:-32])    
        cam_int, bbox_info = self.get_cam_intrinsic(batch)
        # x_cls = x_norm_patch[:, 0, :]
        # B, 1280
        x_cls = x_norm_patch.mean(1)
        # enc_poses ... / dec_poses
        outputs = self.decoder(x_cls, 
                            x_norm_patch=x_norm_patch, bbox_info=bbox_info)
         
        return outputs, cam_int

    def get_cam_intrinsic(self, batch):
       
        
        img_h, img_w = batch['img_size'][:, 0], batch['img_size'][:, 1]  
        if self.training:
            cam_intrinsics = batch['cam_int']
        else:
            # if evaluation 
            batch_size = img_h.shape[0]
            device = img_h.device
            
            cam_intrinsics = torch.zeros(batch_size, 3, 3, device=device, dtype=torch.float32)
             
            with torch.no_grad():
                cam, _ = self.cam_model(batch['img_full_resized'])
            vfov = cam[:, 1]
            fl_h = img_h / (2 * torch.tan(vfov / 2)) 

            cam_intrinsics[:, 0, 0] = fl_h  # fx
            cam_intrinsics[:, 1, 1] = fl_h  # fy
            cam_intrinsics[:, 0, 2] = img_w / 2.  # cx
            cam_intrinsics[:, 1, 2] = img_h / 2.  # cy
            cam_intrinsics[:, 2, 2] = 1.0  
 
        
        # Original
        b = batch['bbox_size']
        cx, cy = batch['box_center'][:, 0], batch['box_center'][:, 1]
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b],
                                dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / cam_intrinsics[:, 0, 0].unsqueeze(-1)   # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] / cam_intrinsics[:, 0, 0])  # [-1, 1]

        bbox_info = bbox_info.cuda().float()
        return cam_intrinsics, bbox_info

    def convert_to_full_img_cam(
        self, pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length):

        s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
        tz = 2. * focal_length / (bbox_height * s)

        cx = 2. * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
        cy = 2. * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
        return cam_t

    def get_pd_params(self, pd_poses, pd_betas, pd_cam, batch, cam_intrinsics):

        skel_params = {
            'poses': pd_poses,
            'betas': pd_betas
        }
        
        # Get the 3D keypoints, skin vertices and 2D keypoints
        skel_outputs = self.skel_model(**skel_params, skelmesh=False)

        pd_kp3d = skel_outputs.joints  # (B, Q=44, 3)
        pd_skin_verts = skel_outputs.skin_verts.detach().cpu().clone()  # (B, V=6890, 3)

        img_h, img_w = batch['img_size'][:, 0], batch['img_size'][:, 1]

        pd_cam_t = self.convert_to_full_img_cam(
            pare_cam = pd_cam,
            bbox_height = batch['bbox_size'],
            bbox_center = batch['box_center'],
            img_w = img_w,
            img_h = img_h,
            focal_length = cam_intrinsics[:, 0, 0],
        )
        
        batch_size = pd_poses.shape[0]
        # full img translation & resize img focal length  
        pd_kp2d = cam_perspective_projection(
                points       = pd_kp3d, # (B, K=Q=44, 3)
                rotation     = torch.eye(3, device='cuda').unsqueeze(0).expand(batch_size, -1, -1),
                translation  = pd_cam_t,  # (B, 3)
                cam_intrinsics = cam_intrinsics
            )

        total_predict = {
            'pd_cam_t' : pd_cam_t,
            'pd_kp3d' : pd_kp3d,
            'pd_kp2d' : pd_kp2d,
            'pd_skin_verts' : pd_skin_verts,
            'pd_skel_params' : skel_params,
        }

        return total_predict
    

    def forward(self, batch):
        img_patch = batch['img']  # (B, C, H, W)

        outputs, cam_int = self.forward_step(img_patch, batch)  # {...}
        predict_enc = self.get_pd_params(outputs['pd_enc_poses'], outputs['pd_enc_betas'], outputs['pd_enc_cam'], batch, cam_int)
        predict_dec = self.get_pd_params(outputs['pd_dec_poses'], outputs['pd_dec_betas'], outputs['pd_dec_cam'], batch, cam_int)   
        per_layer_params = outputs['per_layer_params']
        
        
        return predict_enc, predict_dec, per_layer_params


def build_model(cfg):
    skelvit = SKELViT(cfg)
    loss_func = HPE_Loss(cfg)
    
    return skelvit, loss_func