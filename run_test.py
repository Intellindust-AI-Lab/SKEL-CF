from collections import defaultdict
from pathlib import Path
import hydra
import numpy as np
import torch 
from typing import Dict
from smplx import SMPL
from tqdm import tqdm
from body_models.skel_utils.transforms import params_rep2q
from body_models.skel_wrapper import SKELWrapper
from datasets import build_dataset
from datasets.constants import H36M_TO_J14, REGRESSOR_H36M, SMPL_MODEL_DIR
from datasets.moyo_datasets import EvalMoyoDataset
from models.skelvit import SKELViT
from util.pylogger import get_pylogger
from torch.utils.data import DataLoader

logger = get_pylogger(__name__)

def axis_angle_to_matrix(vec):
    # vec: (N,3)
    theta = torch.norm(vec + 1e-8, dim=1, keepdim=True)  # (N,1)
    axis = vec / theta
    axis = axis.view(-1, 3, 1)  # (N,3,1)

    cos = torch.cos(theta).view(-1, 1, 1)
    sin = torch.sin(theta).view(-1, 1, 1)
    I = torch.eye(3, device=vec.device).unsqueeze(0)

    outer = axis @ axis.transpose(1,2)  # (N,3,3)
    skew = torch.zeros_like(outer)
    skew[:,0,1], skew[:,0,2], skew[:,1,0] = -axis[:,2,0], axis[:,1,0], axis[:,2,0]
    skew[:,1,2], skew[:,2,0], skew[:,2,1] = -axis[:,0,0], -axis[:,1,0], axis[:,0,0]

    R = cos * I + (1-cos) * outer + sin * skew
    return R


def to_device(data, device='cuda'):
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).to(device)
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    elif isinstance(data, Dict):
        data = {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, (np.generic, float, int)):
        data = torch.tensor(data).to(device)
    
    return data 


class Evaluator():
    def __init__(self):
        self.val_list = defaultdict(list)
        self.ds_name = None
    
    def eval(self, pd, gt):
        if 'pd_verts' in pd:
            pd_verts = pd['pd_verts']
            gt_verts = gt['gt_verts']
            pve = self.eval_verts(pd_verts, gt_verts)
            papve = self.eval_papve(pd_verts, gt_verts)

            self.val_list['papve'].append(papve)
            self.val_list['pve'].append(pve)

        if 'pd_kp3d' in pd:
            pd_j3d = pd['pd_kp3d']
            gt_j3d = gt['gt_kp3d']
            mpjpe = self.eval_j3d(pd_j3d, gt_j3d)
            pa_mpjpe = self.eval_paj3d(pd_j3d, gt_j3d)
            self.val_list['mpjpe'].append(mpjpe)
            self.val_list['pa_mpjpe'].append(pa_mpjpe)
        
        if 'kp2d' in pd:
            pass 

    def eval_verts(self, pd_verts, gt_verts):
        scale = 1000
        result = torch.sqrt(torch.sum((gt_verts - pd_verts) ** 2, dim=-1)).mean(dim=-1)
        return result * scale

    def eval_papve(self, pd_verts, gt_verts):
        pd_aligned = self.similarity_align_to(pd_verts, gt_verts)
        return self.eval_j3d(pd_aligned, gt_verts)

    def eval_j3d(self, pd_j3d, gt_j3d):
        scale = 1000
        result = torch.sqrt(torch.sum((gt_j3d - pd_j3d) ** 2, dim=-1)).mean(dim=-1)
        return result * scale
    
    def eval_paj3d(self, pd_j3d, gt_j3d):
        pd_aligned = self.similarity_align_to(pd_j3d, gt_j3d)
        return self.eval_j3d(pd_aligned, gt_j3d)

    
    def similarity_align_to(
        self,
        S1 : torch.Tensor,
        S2 : torch.Tensor,
    ):
        
        assert (S1.shape[-1] == 3 and S2.shape[-1] == 3), 'The last dimension of `S1` and `S2` must be 3.'
        assert (S1.shape[:-2] == S2.shape[:-2]), 'The batch size of `S1` and `S2` must be the same.'
        original_BN3 = S1.shape
        N = original_BN3[-2]
        S1 = S1.reshape(-1, N, 3) # (B', N, 3) <- (...B, N, 3)
        S2 = S2.reshape(-1, N, 3) # (B', N, 3) <- (...B, N, 3)
        B = S1.shape[0]

        S1 = S1.transpose(-1, -2) # (B', 3, N) <- (B', N, 3)
        S2 = S2.transpose(-1, -2) # (B', 3, N) <- (B', N, 3)
        _device = S2.device
        S1 = S1.to(_device)

        # 1. Remove mean.
        mu1 = S1.mean(axis=-1, keepdims=True) # (B', 3, 1)
        mu2 = S2.mean(axis=-1, keepdims=True) # (B', 3, 1)
        X1 = S1 - mu1 # (B', 3, N)
        X2 = S2 - mu2 # (B', 3, N)

        # 2. Compute variance of X1 used for scales.
        var1 = torch.einsum('...BDN->...B', X1**2) # (B',)

        # 3. The outer product of X1 and X2.
        K = X1 @ X2.transpose(-1, -2) # (B', 3, 3)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
        U, s, V = torch.svd(K) # (B', 3, 3), (B', 3), (B', 3, 3)

        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = torch.eye(3, device=_device)[None].repeat(B, 1, 1) # (B', 3, 3)
        Z[:, -1, -1] *= (U @ V.transpose(-1, -2)).det().sign()

        # Construct R.
        R = V @ (Z @ U.transpose(-1, -2)) # (B', 3, 3)

        # 5. Recover scales.
        traces = [torch.trace(x)[None] for x in (R @ K)]
        scales = torch.cat(traces) / var1 # (B',)
        scales = scales[..., None, None] # (B', 1, 1)

        # 6. Recover translation.
        t = mu2 - (scales * (R @ mu1)) # (B', 3, 1)

        # 7. Error:
        S1_aligned = scales * (R @ S1) + t # (B', 3, N)

        S1_aligned = S1_aligned.transpose(-1, -2) # (B', N, 3) <- (B', 3, N)
        S1_aligned = S1_aligned.reshape(original_BN3) # (...B, N, 3)
        return S1_aligned # (...B, N, 3)


    def compute_the_results(self):
        if len(self.val_list) == 0:
            logger.info(f'{self.ds_name} val_list is empty!!')

        logger.info(f'Aggregate the {self.ds_name} results!')
        for key in self.val_list.keys():
            results = self.val_list[key]
            results = torch.cat(results, dim=0) # (B, B, B, ...)
            results = results.mean()
            logger.info(f'{key}: {results}')



def fix_prefix_state_dict(st: Dict):
    new_st = {}
    for k, v in st.items():
        if k.startswith('model.'):
            new_k = k.replace('model.', '', 1)
            new_st[new_k] = v
    return new_st


def evaluate_emdb(batch, total_predict, evaluator, device):
    gt_vertices = batch['vertices']
    device = gt_vertices.device

    smpl = SMPL(SMPL_MODEL_DIR, gender='neutral')
    smpl_J_regressor = smpl.J_regressor.to(device)

    gt_keypoints_3d = torch.matmul(smpl_J_regressor, gt_vertices)
    gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
    # gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
    gt_vertices = gt_vertices - gt_pelvis

    pred_vertices = total_predict['pd_skin_verts'].to(device)
    pred_keypoints_3d = torch.matmul(smpl_J_regressor, pred_vertices) # B, 24
    pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
    # pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
    pred_vertices = pred_vertices - pred_pelvis

    evaluator.eval(
        {'pd_verts': pred_vertices, 'pd_kp3d': pred_keypoints_3d},
        {'gt_verts': gt_vertices, 'gt_kp3d': gt_keypoints_3d}
    )


def evaluate_moyo(skel_model, smpl_model, batch, total_predict, evaluator, device):
    gt_smpl_params = batch['smpl']

    smpl_output = smpl_model(**gt_smpl_params)
    gt_vertices = smpl_output.vertices
    gt_joints = smpl_output.joints[:, :24] # (B, 24, 3)

    gt_pelvis = gt_joints[:, [0], :].clone()
    gt_joints = gt_joints - gt_pelvis
    gt_vertices = gt_vertices - gt_pelvis

    skel_output = skel_model(**total_predict['pd_skel_params'], skelmesh=False)

    pd_vertices = skel_output.skin_verts
    pd_joints = skel_output.joints_custom
    pred_pelvis = pd_joints[:, [0], :].clone()
    pd_joints = pd_joints - pred_pelvis
    pd_vertices = pd_vertices - pred_pelvis
    
    evaluator.eval(
        {'pd_verts': pd_vertices, 'pd_kp3d': pd_joints},
        {'gt_verts': gt_vertices, 'gt_kp3d': gt_joints}
    )



def evaluate_spec(batch, total_predict, evaluator, device):
    gt_vertices = batch['vertices']
    device = gt_vertices.device

    J_regressor_batch_smpl = torch.from_numpy(np.load(REGRESSOR_H36M))
    J_regressor_batch_smpl = J_regressor_batch_smpl[None, :].expand(batch['img'].shape[0], -1, -1).float().cuda().to(device)

    gt_keypoints_3d = torch.matmul(J_regressor_batch_smpl, gt_vertices)
    gt_pelvis = (gt_keypoints_3d[:, [1], :] + gt_keypoints_3d[:, [2], :]) / 2.0
    # gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
    gt_vertices = gt_vertices - gt_pelvis

    pred_vertices = total_predict['pd_skin_verts'].to(device)
    pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_vertices) # B, 24
    pred_pelvis = (pred_keypoints_3d[:, [1], :] + pred_keypoints_3d[:, [2], :]) / 2.0
    # pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
    pred_vertices = pred_vertices - pred_pelvis

    evaluator.eval(
        {'pd_verts': pred_vertices, 'pd_kp3d': pred_keypoints_3d},
        {'gt_verts': gt_vertices, 'gt_kp3d': gt_keypoints_3d}
    )


def evaluate_3dpw(batch, total_predict, evaluator, device):
    gt_vertices = batch['vertices']
    device = gt_vertices.device

    J_regressor_batch_smpl = torch.from_numpy(np.load(REGRESSOR_H36M))
    J_regressor_batch_smpl = J_regressor_batch_smpl[None, :].expand(batch['img'].shape[0], -1, -1).float().cuda().to(device)
    joint_mapper_h36m = H36M_TO_J14
    
    gt_keypoints_3d = torch.matmul(J_regressor_batch_smpl, gt_vertices)
    gt_pelvis = gt_keypoints_3d[:, [0], :].clone()
    gt_keypoints_3d = gt_keypoints_3d[:, joint_mapper_h36m, :]
    gt_keypoints_3d = gt_keypoints_3d - gt_pelvis
    gt_vertices = gt_vertices - gt_pelvis
    
    # pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
    pred_vertices = total_predict['pd_skin_verts'].to(device)

    pred_keypoints_3d = torch.matmul(J_regressor_batch_smpl, pred_vertices) # B, 24
    pred_pelvis = pred_keypoints_3d[:, [0], :].clone()
    pred_keypoints_3d = pred_keypoints_3d[:, joint_mapper_h36m, :]
    pred_keypoints_3d = pred_keypoints_3d - pred_pelvis
    pred_vertices = pred_vertices - pred_pelvis

    evaluator.eval(
        {'pd_verts': pred_vertices, 'pd_kp3d': pred_keypoints_3d},
        {'gt_verts': gt_vertices, 'gt_kp3d': gt_keypoints_3d}
    )


@hydra.main(version_base='1.2', config_path="config", config_name="eval.yaml")
@torch.no_grad()
def test(cfg):
    val_dataset_list = build_dataset(cfg, 'val')
    root_dir = Path(cfg.paths.data_inputs)
    if 'moyo' in cfg.DATASETS.EXTRA_DATASETS:
        # Load MOYO-HARD dataset
        npz_file = root_dir / 'skel-evaluation-labels' / 'moyo_hard.npz'
        moyo_hard_dataset = EvalMoyoDataset(cfg, npz_fn=npz_file, ignore_img=False)
        val_dataset_list.append(moyo_hard_dataset)
        
        # Load MOYO-all dataset
        npz_file = root_dir / 'skel-evaluation-labels' / 'moyo_v2.npz'
        moyo_dataset = EvalMoyoDataset(cfg, npz_fn=npz_file, ignore_img=False)
        val_dataset_list.append(moyo_dataset)

        logger.info(f'Loaded MOYO & MOYO-HARD datasets')


    device = torch.device('cuda')
   
    model = SKELViT(cfg) 
    ckpt_paths = cfg.trainer.ckpt_path
    
    # load ema_model
    st = torch.load(ckpt_paths, map_location='cpu')['ema_model']
    st = fix_prefix_state_dict(st)
    model.load_state_dict(st, strict=True)
    model = model.to(device)
    model.eval()

    data_inputs = Path(cfg.paths.data_inputs)

    skel_model = SKELWrapper(
        gender = 'male',
        model_path = data_inputs / 'body_models' / 'skel',
        joint_regressor_extra = data_inputs / 'body_models'  / 'SMPL_to_J19.pkl',
        joint_regressor_custom = data_inputs / 'body_models' / 'J_regressor_SMPL_MALE.pkl',
    ).to(device)

    smpl_model = SMPL(
        gender = 'neutral',
        model_path = data_inputs / 'body_models' / 'SMPL',
    ).to(device)

    evaluator = Evaluator()
      
    for dataset_val in val_dataset_list:
        data_loader_val = DataLoader(dataset_val, cfg.trainer.test_batch_size, drop_last=False, num_workers=cfg.general.num_workers)
    
        ds_name = None 
        for i, batch in enumerate(tqdm(data_loader_val, total=len(data_loader_val), desc='Testing')):
            batch = {k: to_device(v, device) for k, v in batch.items()}
            img = batch['img']
            ds_name = batch['ds_name']

            if evaluator.ds_name is None:
                evaluator.ds_name = ds_name[0]

            B = img.shape[0]  
            # enc, dec, layer          
            _, total_predict, _ = model(batch)

            if 'emdb' in ds_name[0]:
                evaluate_emdb(batch, total_predict, evaluator, device)
            elif 'moyo' in ds_name[0] or 'moyo_hard' in ds_name[0]:
                evaluate_moyo(skel_model, smpl_model, batch, total_predict, evaluator, device)
            elif '3dpw' in ds_name[0]:
                evaluate_3dpw(batch, total_predict, evaluator, device)
            elif 'spec' in ds_name[0]:
                evaluate_spec(batch, total_predict, evaluator, device)
            else:
                raise ValueError(f'UnKnown dataset {ds_name[0]}')
                                                            
        evaluator.compute_the_results()
        evaluator.ds_name = None
        evaluator.val_list.clear()



if __name__ == '__main__':
    test()

