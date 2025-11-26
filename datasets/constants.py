from body_models.skel.osim_rot import ConstantCurvatureJoint, CustomJoint, EllipsoidJoint, PinJoint, WalkerKnee
from torch import nn 

NUM_POSE_PARAMS = 23
NUM_BETAS = 10
NUM_JOINTS = 44
NUM_PARAMS_SMPL = 24
NUM_PARAMS_SMPLX = 22 # Only body
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]

body_permutation = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]
extra_permutation = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18]
smpl_to_openpose = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                    7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
FLIP_KEYPOINT_PERMUTATION = body_permutation + [25 + i for i in extra_permutation]

SMPL_TO_OPENPOSE = [24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                    7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
SMPL_to_J19 = 'data_inputs/body_models/train-eval-utils/SMPL_to_J19.pkl'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data_inputs/body_models/train-eval-utils/J_regressor_extra.npy'
REGRESSOR_H36M='data_inputs/body_models/train-eval-utils/J_regressor_h36m.npy'

SMPL_MODEL_DIR='data_inputs/body_models/SMPL'
VITPOSE_BACKBONE='data_inputs/backbone/vitpose_backbone.pth'
DETECTRON_CFG='util/cascade_mask_rcnn_vitdet_h_75ep.py'

JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}


SKEL_JOINTS_DICT = nn.ModuleList([
    CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]), # 0 pelvis
    CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]), # 1 femur_r
    WalkerKnee(), # 2 tibia_r
    PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]), # 3 talus_r Field taken from .osim Joint-> frames -> PhysicalOffsetFrame -> orientation
    PinJoint(parent_frame_ori = [-1.76818999, 0.906223, 1.8196000]), # 4 calcn_r
    PinJoint(parent_frame_ori = [-3.141589999, 0.6199010, 0]), # 5 toes_r
    CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, -1, -1]), # 6 femur_l
    WalkerKnee(), # 7 tibia_l
    PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]), # 8 talus_l
    PinJoint(parent_frame_ori = [1.768189999 ,-0.906223, 1.8196000]), # 9 calcn_l
    PinJoint(parent_frame_ori = [-3.141589999, -0.6199010, 0]), # 10 toes_l
    ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]), # 11 lumbar
    ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]), # 12 thorax
    ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]), # 13 head
    EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, -1, -1]), # 14 scapula_r
    CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]), # 15 humerus_r
    CustomJoint(axis=[[0.0494, 0.0366, 0.99810825]], axis_flip=[[1]]), # 16 ulna_r
    CustomJoint(axis=[[-0.01716099, 0.99266564, -0.11966796]], axis_flip=[[1]]), # 17 radius_r
    CustomJoint(axis=[[1,0,0], [0,0,-1]], axis_flip=[1, 1]), # 18 hand_r
    EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, 1, 1]), # 19 scapula_l
    CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]), # 20 humerus_l
    CustomJoint(axis=[[-0.0494, -0.0366, 0.99810825]], axis_flip=[[1]]), # 21 ulna_l
    CustomJoint(axis=[[0.01716099, -0.99266564, -0.11966796]], axis_flip=[[1]]), # 22 radius_l
    CustomJoint(axis=[[-1,0,0], [0,0,-1]], axis_flip=[1, 1]), # 23 hand_l
])