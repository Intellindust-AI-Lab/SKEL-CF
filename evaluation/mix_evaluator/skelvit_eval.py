
from evaluation import HSMR3DEvaluator, SKELRealityEvaluator
from evaluation.hmr2_utils import Evaluator as HMR2Evaluator
from smplx import SMPL
from body_models.skel_wrapper import SKELWrapper as SKEL
# ================== Body Models Tools ==================



def make_SMPL(gender='neutral', device='cuda:0'):
    return SMPL(
        gender = gender,
        model_path = PM.inputs / 'body_models' / 'smpl',
    ).to(device)



def make_SKEL_smpl_joints(gender='male', device='cuda:0'):
    ''' We don't have neutral model for SKEL, so use male for now. '''
    return SKEL(
        gender = gender,
        model_path = PM.inputs / 'body_models' / 'skel',
        joint_regressor_extra = PM.inputs / 'body_models' / 'SMPL_to_J19.pkl',
        joint_regressor_custom = PM.inputs / 'body_models' / 'J_regressor_SMPL_MALE.pkl',
    ).to(device)


def _reset_root(v3d, j3d):
    ''' Reset the root position to the origin. '''
    r3d = j3d[:, [0]]  # (B, 1, 3)
    v3d -= r3d  # (B, V, 3)
    j3d -= r3d  # (B, J, 3)
    return v3d, j3d


def _forward_gt_smpl24(gt):
    ''' Inference SMPL parameters to standard SMPL vertices and joints(24). '''
    # Lazily create body_model.
    if not hasattr(_forward_gt_smpl24, 'smpl_model'):
        _forward_gt_smpl24.smpl_model = make_SMPL(gender='neutral', device=gt['smpl']['betas'].device)
    smpl_model = _forward_gt_smpl24.smpl_model
    # SMPL inference.
    smpl_output = smpl_model(**gt['smpl'])
    gt_v = smpl_output.vertices
    gt_j = smpl_output.joints[:, :24]  # (B, J=24, 3)
    return _reset_root(gt_v, gt_j)


def _forward_pd_smpl24(pd):
    ''' Inference SKEL parameters to standard SMPL vertices and joints(24). '''
    # Lazily create body_model.
    if not hasattr(_forward_pd_smpl24, 'skel_model'):
        _forward_pd_smpl24.skel_model = make_SKEL_smpl_joints(device=pd['pd_params']['betas'].device)
    skel_model = _forward_pd_smpl24.skel_model
    # SKEL inference.
    skel_output = skel_model(**pd['pd_params'], skelmesh=False)
    pd_v = skel_output.skin_verts  # (B, V=6890, 3)
    pd_j = skel_output.joints_custom  # (B, J=24, 3)
    return _reset_root(pd_v, pd_j)



class UniformEvaluator():
    MODE_STD = 'std'
    MODE_EXT = 'ext'


    def __init__(self, data, device='cuda:0'):
        ''' Determine which evaluator to use. '''
        self.device = device
        if data['name'] in ['MOYO']:
            self.mode = self.MODE_EXT
            self.accuracy_ext = HSMR3DEvaluator()
            self.reality = SKELRealityEvaluator()
        else:
            self.mode = self.MODE_STD
            if data['name'] in ['H36M-VAL-P2','3DPW-TEST']:
                metrics = ['mode_re', 'mode_mpjpe']
                pck_thresholds = None
            elif data['name'] in ['LSP-EXTENDED', 'POSETRACK-VAL', 'COCO-VAL']:
                metrics = ['mode_kpl2']
                pck_thresholds = [0.05, 0.1]

            self.accuracy_std = HMR2Evaluator(
                dataset_length = int(1e8),
                keypoint_list  = data['dataset']._kp_list_,
                pelvis_ind     = 39,
                metrics        = metrics,
                pck_thresholds = pck_thresholds,
            )


    def eval(self, pd, gt):
        ''' Uniform evaluation interface. '''
        if self.mode == self.MODE_EXT:
            pd_v, pd_j = _forward_pd_smpl24(pd)
            gt_v, gt_j = _forward_gt_smpl24(gt)
            self.accuracy_ext.eval(
                pd = {'v3d_pose': pd_v, 'j3d_pose': pd_j},
                gt = {'v3d_pose': gt_v, 'j3d_pose': gt_j},
            )
            self.reality.eval(
                pd = {'poses': pd['pd_params']['poses']},
            )
        elif self.mode == self.MODE_STD:
            self.accuracy_std(pd, gt)


    def get_results(self):
        ''' Uniform results interface. '''
        if self.mode == self.MODE_EXT:
            results = {
                **self.accuracy_ext.get_results(),
                **self.reality.get_results(),
            }
        elif self.mode == self.MODE_STD:
            results = self.accuracy_std.get_metrics_dict()
        return results