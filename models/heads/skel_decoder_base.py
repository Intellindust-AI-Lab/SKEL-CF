import torch
import torch.nn as nn
import numpy as np

from omegaconf import OmegaConf

from body_models.skel_utils.transforms import params_q2rep, params_rep2q
from util.constants import SKEL_MEAN_PARAMS
from util.pylogger import get_pylogger
from .utils.pose_transformer import SKELTransformerDecoder

logger = get_pylogger(__name__)

class SKELTransformerDecoderHeadBase(nn.Module):
    """
        Cross-attention based SKEL Transformer decoder
        1. input cls_token, token_norm_patch, output final pose 
        2. calculate the Loss
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        if cfg.hub.skel_head.pd_poses_repr == 'rotation_6d':
            n_poses = 24 * 6
        elif cfg.hub.skel_head.pd_poses_repr== 'euler_angle':
            n_poses = 46
        else:
            raise ValueError(f"Unknown pose representation: {cfg.hub.skel_head.pd_poses_repr}")

        self.num_decoder_layers = 6 
        n_betas = 10
        n_cam   = 3

        # Build transformer decoder.
        transformer_args = {
                'num_tokens' : 1,
                'dim'        : cfg.hub.skel_head.transformer_decoder.context_dim,
                'token_dim'  : cfg.hub.skel_head.transformer_decoder.context_dim,
                'skip_token_embedding' : True,
            }
        transformer_args.update(OmegaConf.to_container(cfg.hub.skel_head.transformer_decoder, resolve=True))  # type: ignore
          
        self.transformer = SKELTransformerDecoder(**transformer_args)
        
        # Build decoders for parameters.
        dim = transformer_args['dim']

        self.cam_decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, n_cam),
        )

        self.betas_decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, n_betas),
        )

        # free decoder 
        self.poses_decoder = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, n_poses),
        )
        

        self.to_pose_embedding = nn.Linear(24 * 6 + 3, dim)
        self.to_betas_embedding = nn.Linear(10 + 3, dim)
        self.to_cam_embedding = nn.Linear(3 + 3, dim)

        logger.info("seperate embedding layers for pose, betas, cam !")
       

        mean_params = np.load(SKEL_MEAN_PARAMS, allow_pickle=True)
        mean_poses = torch.from_numpy(mean_params['poses']).unsqueeze(0)
        mean_poses = params_q2rep(mean_poses).reshape(-1, 24*6)
        mean_betas = torch.from_numpy(mean_params['betas']).unsqueeze(0)
        mean_cam = torch.from_numpy(mean_params['cam']).unsqueeze(0)

        self.register_buffer('mean_poses', mean_poses)
        self.register_buffer('mean_betas', mean_betas)
        self.register_buffer('mean_cam', mean_cam)

        
    @property
    def layers(self):
        return self.transformer.layers
    
    
    @property
    def dropout(self):
        return self.transformer.dropout
    
    @property
    def pos_embedding(self):
        return self.transformer.pos_embedding


    def forward(self, x, **kwargs):
        # x_norm_patch
        B = x.shape[0]
       
        x_cls = x 
        x_norm_patch = kwargs.get('x_norm_patch', None)
        bbox_info = kwargs.get('bbox_info', None)

        assert x_norm_patch is not None and bbox_info is not None, 'x_norm_patch and bbox_info are required'

        # 1. initial pose, betas, cam
        poses_init = self.poses_decoder(x_cls) + self.mean_poses
        betas_init = self.betas_decoder(x_cls) + self.mean_betas
        cam_init = self.cam_decoder(x_cls) + self.mean_cam

        pose_i = poses_init
        betas_i = betas_init
        cam_i = cam_init
     
        # Pass through transformer.        
        context_list = [x_norm_patch] * len(self.layers)        

        poses_token = torch.cat([poses_init,  bbox_info], dim=-1)[:, None,:]  # B, 1, 147
        betas_token = torch.cat([betas_init, bbox_info], dim=-1)[:, None,:]  # B, 1, 13
        cam_token = torch.cat([cam_init, bbox_info], dim=-1)[:, None,:]  # B, 1, 6
    
        poses_token = self.to_pose_embedding(poses_token)
        betas_token = self.to_betas_embedding(betas_token)
        cam_token = self.to_cam_embedding(cam_token)

        # B, 3, dim
        token = torch.cat([poses_token, betas_token, cam_token], dim=1)
        b, n, _ = token.shape
        token = self.dropout(token)
        token = token + self.pos_embedding[:, :n]
        
        per_layer_params = []
        for i, (self_attn, cross_attn, ff) in enumerate(self.layers):
            # 1. predict the residual
            token = self_attn(token) + token
            token = cross_attn(token, context=context_list[i]) + token
            token = ff(token) + token

            # 2. trans token to the human params && add the residual
            pose_i = pose_i + self.poses_decoder(token[:, 0]) 
            betas_i = betas_i + self.betas_decoder(token[:, 1])
            cam_i = cam_i + self.cam_decoder(token[:, 2])

            # 3. add the layers' params to the output
            per_layer_params.append({
                f'pd_poses_{i}' : pose_i,
                f'pd_betas_{i}' : betas_i,
                f'pd_cam_{i}' : cam_i
            })

            # 4. trans to token
            poses_token = torch.cat([pose_i,  bbox_info], dim=-1)[:, None,:]  # 1, 1, 147
            betas_token = torch.cat([betas_i, bbox_info], dim=-1)[:, None,:]  # 1, 1, 13
            cam_token = torch.cat([cam_i, bbox_info], dim=-1)[:, None,:]  # 1, 1, 6
        
            poses_token = self.to_pose_embedding(poses_token)
            betas_token = self.to_betas_embedding(betas_token)
            cam_token = self.to_cam_embedding(cam_token)

            token = torch.cat([poses_token, betas_token, cam_token], dim=1) # B, 3, 1024
        
        # 1.final output 
        poses_init = params_rep2q(poses_init.reshape(-1, 24, 6)).reshape(-1, 46)

        # 3.decoder_output
        pd_decoder_poses = pose_i 
        pd_decoder_poses = params_rep2q(pd_decoder_poses.reshape(-1, 24, 6)).reshape(-1, 46)
        pd_decoder_betas = betas_i
        pd_decoder_cam = cam_i

        return {
            'pd_enc_poses' : poses_init,
            'pd_enc_betas' : betas_init,
            'pd_enc_cam' : cam_init,
            'pd_dec_poses' : pd_decoder_poses,
            'pd_dec_betas' : pd_decoder_betas,
            'pd_dec_cam' : pd_decoder_cam, 
            'per_layer_params' : per_layer_params
        }
