import os
curr_dir = os.path.abspath(os.path.dirname(__file__))
base_dir = os.path.join(curr_dir, '../')


DATASET_FOLDERS = {
     # testset-images
    '3dpw-cameraHMR-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/3dpw'),
    'coco-val-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/coco'),
    'emdb-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/emdb'),
    'moyo-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/moyo'),
    'spec-test-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/spec-syn'),
    # 'rich-smplx': os.path.join(base_dir, 'data/test-images/RICH'),
    'h36m-val' : os.path.join(base_dir, 'data_inputs/skel-evaluation-data/h36m-select'),
    'lsp-val' : os.path.join(base_dir, 'data_inputs/skel-evaluation-data/hr-lspet'),
    'posetrack-val' : os.path.join(base_dir, 'data_inputs/skel-evaluation-data/posetrack'),
    # training-images
    'insta-1': os.path.join(base_dir, 'data_inputs/skel-training-images/insta-train/'),
    'insta-2': os.path.join(base_dir, 'data_inputs/skel-training-images/insta-train/'),
    'aic': os.path.join(base_dir, 'data_inputs/skel-training-images/aic-train/'),
    'mpii-train':  os.path.join(base_dir, 'data_inputs/skel-training-images/mpii-train/'),
    'coco-train':  os.path.join(base_dir, 'data_inputs/skel-training-images/coco-train/'),
    'h36m-train':  os.path.join(base_dir, 'data_inputs/skel-training-images/h36m-train/'),
    'mpi-inf-train':  os.path.join(base_dir, 'data_inputs/skel-training-images/mpi-inf-train/'),
    'agora-train':  os.path.join(base_dir, '/public_hw/home/cit_yuzeng/skelvit/data_inputs/skel-training-images/bedlam/images'),
    # 2022 1010 - 2
    '20221010_3_1000_batch01hand_6fps': 
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221010_3_1000_batch01hand_6fps/png'),
    '20221010_3-10_500_batch01hand_zoom_suburb_d_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps/png'),
    # 2022 1011 - 4
    '20221011_1_250_batch01hand_closeup_suburb_a_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221011_1_250_batch01hand_closeup_suburb_a_6fps/png'),
    '20221011_1_250_batch01hand_closeup_suburb_b_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221011_1_250_batch01hand_closeup_suburb_b_6fps/png'),
    '20221011_1_250_batch01hand_closeup_suburb_c_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221011_1_250_batch01hand_closeup_suburb_c_6fps/png'),
    '20221011_1_250_batch01hand_closeup_suburb_d_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221011_1_250_batch01hand_closeup_suburb_d_6fps/png'),
    # 2022 1012 - 3
    '20221012_1_500_batch01hand_closeup_highSchoolGym_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps/png'),
    '20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps/png'),
    '20221013_3_250_batch01hand_orbit_bigOffice_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221013_3_250_batch01hand_orbit_bigOffice_6fps/png'),
    # 2022 1013 - 2
    '20221013_3_250_batch01hand_static_bigOffice_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221013_3_250_batch01hand_static_bigOffice_6fps/png'),
    '20221013_3-10_500_batch01hand_static_highSchoolGym_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps/png'),
    # 2022 1014 - 1
    '20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png'),
    # 2022 1015 - 3
    '20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps/png'),
    '20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps/png'),
    '20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps/png'),
    # 2022 1017 - 1
    '20221017_3_1000_batch01hand_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221017_3_1000_batch01hand_6fps/png'),
     # 2022 1018 - 5
    '20221018_1_250_batch01hand_zoom_suburb_b_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221018_1_250_batch01hand_zoom_suburb_b_6fps/png'),
    '20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps/png'),
    '20221018_3-8_250_batch01hand_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221018_3-8_250_batch01hand_6fps/png'),
    '20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps/png'),
    '20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps/png'),
    # 2022 1019 - 4
    '20221019_1_250_highbmihand_closeup_suburb_b_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221019_1_250_highbmihand_closeup_suburb_b_6fps/png'),
    '20221019_1_250_highbmihand_closeup_suburb_c_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221019_1_250_highbmihand_closeup_suburb_c_6fps/png'),
    '20221019_3-8_1000_highbmihand_static_suburb_d_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221019_3-8_1000_highbmihand_static_suburb_d_6fps/png'),
    '20221019_3_250_highbmihand_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221019_3_250_highbmihand_6fps/png'),
    # 2022 1020 - 1
    '20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps/png'),
    # 2022 1022 - 1
    '20221022_3_250_batch01handhair_static_bigOffice_30fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221022_3_250_batch01handhair_static_bigOffice_30fps/png'),
    # 2022 1024 - 2
    '20221024_10_100_batch01handhair_zoom_suburb_d_30fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221024_10_100_batch01handhair_zoom_suburb_d_30fps/png'),
    '20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-images/bedlam/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps/png'),
}

DATASET_FILES = [
    {
        # test labels
        '3dpw-cameraHMR-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/3dpw_cameraHMR.npz'),
        'emdb-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/emdb_test.npz'),
        'moyo-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/moyo_v2.npz'),
        # 'rich-smplx': os.path.join(base_dir, 'data/test-labels/rich_test.npz'),
        'h36m-val': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/h36m_val_p2.npz'),
        'spec-test-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/spec_test.npz'),
        'coco-val-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/coco_val.npz'),
        'lsp-val' : os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/hr-lspet_train.npz'),
        'posetrack-val' : os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/posetrack_2018_val.npz'),
    },
    {
        # training labels
        'aic': os.path.join(base_dir, 'data_inputs/skel-training-labels/aic-release-skel-no-trans.npz'),
        'insta-1': os.path.join(base_dir, 'data_inputs/skel-training-labels/insta1-release-skel-no-trans.npz'),
        'insta-2': os.path.join(base_dir, 'data_inputs/skel-training-labels/insta2-release-skel-no-trans.npz'),
        'coco-train': os.path.join(base_dir, 'data_inputs/skel-training-labels/coco-release-skel-no-trans.npz'),
        'mpii-train': os.path.join(base_dir, 'data_inputs/skel-training-labels/mpii-release-skel-no-trans.npz'),
        'h36m-train': os.path.join(base_dir, 'data_inputs/skel-training-labels/h36m-release-skel-no-trans.npz'),
        'mpi-inf-train': os.path.join(base_dir, 'data_inputs/skel-training-labels/mpi-inf-release-skel-no-trans.npz'),
        'agora-train': os.path.join(base_dir, 'data_inputs/skel-training-labels/agora_skel_fit.npz'),
        # bedlam
        # 2022 1010 - 2
        '20221010_3_1000_batch01hand_6fps': 
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221010_3_1000_batch01hand_6fps_skel_joints.npz'),
        '20221010_3-10_500_batch01hand_zoom_suburb_d_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221010_3-10_500_batch01hand_zoom_suburb_d_6fps_skel_joints.npz'),
        # 2022 1011 - 4
        '20221011_1_250_batch01hand_closeup_suburb_a_6fps' :
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221011_1_250_batch01hand_closeup_suburb_a_6fps_skel_joints.npz'),
        '20221011_1_250_batch01hand_closeup_suburb_b_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221011_1_250_batch01hand_closeup_suburb_b_6fps_skel_joints.npz'),
        '20221011_1_250_batch01hand_closeup_suburb_c_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221011_1_250_batch01hand_closeup_suburb_c_6fps_skel_joints.npz'),
        '20221011_1_250_batch01hand_closeup_suburb_d_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221011_1_250_batch01hand_closeup_suburb_d_6fps_skel_joints.npz'),
        # 2022 1012 - 2
        '20221012_1_500_batch01hand_closeup_highSchoolGym_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221012_1_500_batch01hand_closeup_highSchoolGym_6fps_skel_joints.npz'),
        '20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221012_3-10_500_batch01hand_zoom_highSchoolGym_6fps_skel_joints.npz'),
        # 2022 1013 - 3
        '20221013_3_250_batch01hand_orbit_bigOffice_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221013_3_250_batch01hand_orbit_bigOffice_6fps_skel_joints.npz'),
        '20221013_3_250_batch01hand_static_bigOffice_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221013_3_250_batch01hand_static_bigOffice_6fps_skel_joints.npz'),
        '20221013_3-10_500_batch01hand_static_highSchoolGym_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221013_3-10_500_batch01hand_static_highSchoolGym_6fps_skel_joints.npz'),
        # 2022 1014 - 1
        '20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221014_3_250_batch01hand_orbit_archVizUI3_time15_6fps_skel_joints.npz'),
        # 2022 1015 - 3
        '20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221015_3_250_batch01hand_orbit_archVizUI3_time10_6fps_skel_joints.npz'),
        '20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221015_3_250_batch01hand_orbit_archVizUI3_time12_6fps_skel_joints.npz'),
        '20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221015_3_250_batch01hand_orbit_archVizUI3_time19_6fps_skel_joints.npz'),
        # 2022 1017 - 1
        '20221017_3_1000_batch01hand_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221017_3_1000_batch01hand_6fps_skel_joints.npz'),
        # 2022 1018 - 5 - 3 = 2
        '20221018_1_250_batch01hand_zoom_suburb_b_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221018_1_250_batch01hand_zoom_suburb_b_6fps_skel_joints.npz'),
        '20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221018_3_250_batch01hand_orbit_archVizUI3_time15_6fps_skel_joints.npz'),
        '20221018_3-8_250_batch01hand_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221018_3-8_250_batch01hand_6fps_skel_joints.npz'),
        '20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221018_3-8_250_batch01hand_pitchDown52_stadium_6fps_skel_joints.npz'),
        '20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221018_3-8_250_batch01hand_pitchUp52_stadium_6fps_skel_joints.npz'),
        # 2022 1019 - 4
        '20221019_1_250_highbmihand_closeup_suburb_b_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221019_1_250_highbmihand_closeup_suburb_b_6fps_skel_joints.npz'),
        '20221019_1_250_highbmihand_closeup_suburb_c_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221019_1_250_highbmihand_closeup_suburb_c_6fps_skel_joints.npz'),
        '20221019_3-8_1000_highbmihand_static_suburb_d_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221019_3-8_1000_highbmihand_static_suburb_d_6fps_skel_joints.npz'),
        '20221019_3_250_highbmihand_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221019_3_250_highbmihand_6fps_skel_joints.npz'),
        # 2022 1020 - 1
        '20221020-3-8_250_highbmihand_zoom_highSchoolGym_a_6fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221020_3-8_250_highbmihand_zoom_highSchoolGym_a_6fps_skel_joints.npz'),
        # 2022 1022 - 1
        '20221022_3_250_batch01handhair_static_bigOffice_30fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221022_3_250_batch01handhair_static_bigOffice_30fps_skel_joints.npz'),
        # 2022 1024 - 2
        '20221024_10_100_batch01handhair_zoom_suburb_d_30fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221024_10_100_batch01handhair_zoom_suburb_d_30fps_skel_joints.npz'),
        '20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps':
           os.path.join(base_dir, 'data_inputs/skel-training-labels/bedlam/20221024_3-10_100_batch01handhair_static_highSchoolGym_30fps_skel_joints.npz'),
    }
]

