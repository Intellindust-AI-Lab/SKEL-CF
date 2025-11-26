import argparse
import os
import pickle

import numpy as np

from skelfitter.aligner import SKELFitter
from skelfitter.utils import load_smpl_seq




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align SKEL to a SMPL frame')

    parser.add_argument('--smpl_mesh_path', type=str, help='Path to the SMPL mesh to align to', default=None)
    parser.add_argument('--smpl_data_path', type=str, help='Path to the SMPL dictionary to align to (.pkl or .npz)', default='')
    parser.add_argument('-o', '--out_dir', type=str, help='Output directory', default='data_outputs/skelfitter_results/')
    parser.add_argument('-F', '--force-recompute', help='Force recomputation of the alignment', default=True)
    parser.add_argument('--gender', type=str, help='Gender of the subject (only needed if not provided with smpl_data_path)', default='male')
    parser.add_argument('--config', help='Yaml config file containing parameters for training. \
                    You can create a config tailored to align a specific sequence. When left to None, \
                        the default config will be used', default=None)

    args = parser.parse_args()
    smpl_data = load_smpl_seq(args.smpl_data_path, gender=args.gender, straighten_hands=False)

    if args.smpl_mesh_path is not None:
        subj_name = os.path.basename(args.smpl_seq_path).split(".")[0]
    elif args.smpl_data_path is not None:
        subj_name = os.path.basename(args.smpl_data_path).split(".")[0]
    else:
        raise ValueError('Either smpl_mesh_path or smpl_data_path must be provided')

    subj_dir = os.path.join(args.out_dir, subj_name)
    os.makedirs(subj_dir, exist_ok=True)
    npz_path = os.path.join(subj_dir, subj_name + f'_skel_fit.npz')

    # resume fitting
    if os.path.exists(npz_path) and not args.force_recompute:
        print('Previous aligned SKEL sequence found at {}. Will be used as initialization.'.format(subj_dir))
        skel_data_init = pickle.load(open(npz_path, 'rb'))
    else:
        skel_data_init = None
    
    skel_fitter = SKELFitter(args.gender, # smpl_data['gender']
                             device='cuda:0',
                             export_meshes=False,
                             config_path = args.config)
    
    skel_seq = skel_fitter.run_fit(smpl_data['trans'], 
                               smpl_data['betas'], 
                               smpl_data['poses'],
                               batch_size=30000,
                               skel_data_init=skel_data_init, 
                               force_recompute=args.force_recompute)
    
    # postprocessing to npz
    # skel_seq contains [poses,betas,trans,skel_v,skin_v...]
    # smpl_data contains trans, poses, betas ,gender,imgname, cam_int,center,scale,keypoints3d, gtkps
    # skel_poses, skel_betas,skel_trans
    store_dict = {}
    # 1. store the transformed skel_parmas
    for k, v in skel_seq.items():
        if k in ('poses', 'betas', 'trans'):
            store_dict[f'skel_{k}'] = v
    # 2. maintain the original smpl_data info
    for k, v in smpl_data.items():
        if k not in ('skel_poses', 'skel_trans', 'skel_betas', 'gender'):
            store_dict[k] = v

    for k, v in store_dict.items():
        print(f'{k}: {v.shape}')

    np.savez(npz_path, **store_dict)
    print('Saved aligned SKEL to {}'.format(subj_dir))