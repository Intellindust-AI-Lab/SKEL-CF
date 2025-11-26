



import pickle
from smplx import SMPL
from smplx.utils import SMPLOutput
import torch

from datasets.constants import SMPL_to_J19, smpl_to_openpose
from smplx.lbs import vertices2joints




class SMPLWrapper(SMPL):
    def __init__(self, *args, **kwargs):
        super(SMPLWrapper, self).__init__(*args, **kwargs)
        J_regressor_extra = pickle.load(open(SMPL_to_J19,'rb'))
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.register_buffer('joint_map', torch.tensor(smpl_to_openpose, dtype=torch.long))

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPLWrapper, self).forward(*args, **kwargs)
        print(smpl_output.joints.shape)
        joints = smpl_output.joints[:, self.joint_map, :]
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([joints, extra_joints], dim=1)
        output = SMPLOutput(vertices=smpl_output.vertices,
                            global_orient=smpl_output.global_orient,
                            body_pose=smpl_output.body_pose,
                            joints=joints,
                            betas=smpl_output.betas,
                            full_pose=smpl_output.full_pose)
        return output