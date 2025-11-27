import os
curr_dir = os.path.abspath(os.path.dirname(__file__))
base_dir = os.path.join(curr_dir, '../')


DATASET_FOLDERS = {
     # testset-images
    '3dpw-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/3dpw'),
    'coco-val-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/coco'),
    'emdb-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/emdb'),
    'moyo-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/moyo'),
    'spec-test-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-data/spec-syn'),
    'h36m-val' : os.path.join(base_dir, 'data_inputs/skel-evaluation-data/h36m-select'),

    # training-images
    'insta-1': os.path.join(base_dir, 'data_inputs/skel-training-images/insta-train/'),
    'insta-2': os.path.join(base_dir, 'data_inputs/skel-training-images/insta-train/'),
    'aic': os.path.join(base_dir, 'data_inputs/skel-training-images/aic-train/'),
    'mpii-train':  os.path.join(base_dir, 'data_inputs/skel-training-images/mpii-train/'),
    'coco-train':  os.path.join(base_dir, 'data_inputs/skel-training-images/coco-train/'),
    'h36m-train':  os.path.join(base_dir, 'data_inputs/skel-training-images/h36m-train/'),
    'mpi-inf-train':  os.path.join(base_dir, 'data_inputs/skel-training-images/mpi-inf-train/'),
}

DATASET_FILES = [
    {
        # test labels
        '3dpw-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/3dpw_val.npz'),
        'emdb-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/emdb_test.npz'),
        'moyo-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/moyo_v2.npz'),
        'h36m-val': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/h36m_val_p2.npz'),
        'spec-test-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/spec_test.npz'),
        'coco-val-smpl': os.path.join(base_dir, 'data_inputs/skel-evaluation-labels/coco_val.npz'),
    },
    {
        # training labels
        'aic': os.path.join(base_dir, 'data_inputs/skel-training-labels/aic-release-skel.npz'),
        'insta-1': os.path.join(base_dir, 'data_inputs/skel-training-labels/insta1-release-skel.npz'),
        'insta-2': os.path.join(base_dir, 'data_inputs/skel-training-labels/insta2-release-skel.npz'),
        'coco-train': os.path.join(base_dir, 'data_inputs/skel-training-labels/coco-release-skel.npz'),
        'mpii-train': os.path.join(base_dir, 'data_inputs/skel-training-labels/mpii-release-skel.npz'),
        'h36m-train': os.path.join(base_dir, 'data_inputs/skel-training-labels/h36m-release-skel.npz'),
        'mpi-inf-train': os.path.join(base_dir, 'data_inputs/skel-training-labels/mpi-inf-release-skel.npz'),
    }
]

