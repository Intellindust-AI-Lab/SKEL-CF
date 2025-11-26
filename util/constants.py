from body_models.skel.osim_rot import ConstantCurvatureJoint, CustomJoint, EllipsoidJoint, PinJoint, WalkerKnee


DINOV2B_BACKBONE = 'data_inputs/backbone/dinov2_vitb14_reg4_pretrain.pth'
DINOV2L_BACKBONE = 'data_inputs/backbone/dinov2_vitl14_reg4_pretrain.pth'
DINOV3L_BACKBONE = 'data_inputs/backbone/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'
DINOV3B_BACKBONE = 'data_inputs/backbone/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
DINOV3H_BACKBONE = 'data_inputs/backbone/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth'
CAM_MODEL = 'data_inputs/backbone/cam_model_cleaned.ckpt'

DINOV3_DIR = 'models/backbones/dinov3-main'
VITPOSE_BACKBONE = 'data_inputs/backbone/vitpose_backbone.pth'
VITPOSEB_BACKBONE = 'data_inputs/backbone/vitpose-b.pth'
CAM_MODEL_CKPT = 'data_inputs/backbone/cam_model_cleaned.ckpt'
DETECTRON_CKPT='data_inputs/backbone/model_final_f05665.pkl'

SKEL_MEAN_PARAMS = 'models/heads/SKEL_mean.npz'
SKEL_TEMPLATE_PARMAS = 'models/heads/pose_rep_posepriork12.npy'

IMAGE_SIZE = 256
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

JOINTS_DEF = [
    CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]),             #  0 pelvis
    CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, 1, 1]),             #  1 femur_r
    WalkerKnee(),                                                                   #  2 tibia_r
    PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]),                  #  3 talus_r Field taken from .osim Joint-> frames -> PhysicalOffsetFrame -> orientation
    PinJoint(parent_frame_ori = [-1.76818999, 0.906223, 1.8196000]),                #  4 calcn_r
    PinJoint(parent_frame_ori = [-3.141589999, 0.6199010, 0]),                      #  5 toes_r
    CustomJoint(axis=[[0,0,1], [1,0,0], [0,1,0]], axis_flip=[1, -1, -1]),           #  6 femur_l
    WalkerKnee(),                                                                   #  7 tibia_l
    PinJoint(parent_frame_ori = [0.175895, -0.105208, 0.0186622]),                  #  8 talus_l
    PinJoint(parent_frame_ori = [1.768189999 ,-0.906223, 1.8196000]),               #  9 calcn_l
    PinJoint(parent_frame_ori = [-3.141589999, -0.6199010, 0]),                     # 10 toes_l
    ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]),  # 11 lumbar
    ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]),  # 12 thorax
    ConstantCurvatureJoint(axis=[[1,0,0], [0,0,1], [0,1,0]], axis_flip=[1, 1, 1]),  # 13 head
    EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, -1, -1]),        # 14 scapula_r
    CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]),             # 15 humerus_r
    CustomJoint(axis=[[0.0494, 0.0366, 0.99810825]], axis_flip=[[1]]),              # 16 ulna_r
    CustomJoint(axis=[[-0.01716099, 0.99266564, -0.11966796]], axis_flip=[[1]]),    # 17 radius_r
    CustomJoint(axis=[[1,0,0], [0,0,1]], axis_flip=[1, -1]),                        # 18 hand_r
    EllipsoidJoint(axis=[[0,1,0], [0,0,1], [1,0,0]], axis_flip=[1, 1, 1]),          # 19 scapula_l
    CustomJoint(axis=[[1,0,0], [0,1,0], [0,0,1]], axis_flip=[1, 1, 1]),             # 20 humerus_l
    CustomJoint(axis=[[-0.0494, -0.0366, 0.99810825]], axis_flip=[[1]]),            # 21 ulna_l
    CustomJoint(axis=[[0.01716099, -0.99266564, -0.11966796]], axis_flip=[[1]]),    # 22 radius_l
    CustomJoint(axis=[[1,0,0], [0,0,1]], axis_flip=[-1, -1]),                       # 23 hand_l
]

N_JOINTS = len(JOINTS_DEF)  # 24