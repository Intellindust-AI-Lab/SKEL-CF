import math
import os
import pickle
import omegaconf
from smplx import SMPL
import torch
import skelfitter.fitter_config as cg
from body_models.skel.skel_model import SKEL
from tqdm import trange

from skelfitter.losses import compute_anchor_pose, compute_anchor_trans, compute_pose_loss, compute_scapula_loss, compute_spine_loss, pretty_loss_print
from skelfitter.utils import to_params, to_torch
import torch.nn.functional as F

from util.geometry import axis_angle_to_matrix, matrix_to_euler_angles

class SKELFitter(object):

    def __init__(self, gender, device, num_betas=10, export_meshes=False, config_path=None) -> None:

        # self.smpl = smplx.create(cg.smpl_folder, model_type='smpl', gender=gender, num_betas=num_betas, batch_size=1, export_meshes=False)
        self.smpl = SMPL(cg.smpl_folder, gender=gender, num_betas=num_betas, batch_size=1, export_meshes=False).to(device)
        self.skel = SKEL(gender, model_path=cg.skel_folder).to(device)
        self.gender = gender
        self.device = device
        self.num_betas = num_betas

        # Instanciate masks used for the vertex to vertex fitting
        # choose some the vertices used for fitting
        fitting_indices = pickle.load(open(cg.fitting_mask_file, 'rb'))
        fitting_mask = torch.zeros(6890, dtype=torch.bool, device=self.device)
        fitting_mask[fitting_indices] = 1 
        # 1xVx1 to be applied to verts that are BxVx3
        self.fitting_mask = fitting_mask.reshape(1, -1, 1).to(self.device)

        smpl_torso_joints = [0, 3]
        # vertices that affect the torso joints
        verts_mask = (self.smpl.lbs_weights[:, smpl_torso_joints] > 0.5).sum(dim=-1) > 0
        # Because verts are of shape BxVx3
        self.torso_verts_mask = verts_mask.unsqueeze(0).unsqueeze(-1).to(device)

        self.export_meshes = export_meshes

        if config_path is None:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(package_directory, 'config.yaml')

        self.cfg = omegaconf.OmegaConf.load(config_path)
    
    def run_fit(self,
                trans_in,
                betas_in,
                poses_in,
                batch_size=20,
                skel_data_init=None,
                force_recompute=False,
                debug=False,
                watch_frame=0,
                freevert_mesh=None):
        """Align SKEL to a SMPL sequence."""
        self.nb_frames = poses_in.shape[0]
        self.watch_frame = watch_frame
        if self.watch_frame >= self.nb_frames:
            raise ValueError(f'watch_frame {self.watch_frame} is larger than the number of frames {self.nb_frames}. Please provide a watch frame index smaller than the number of frames ({self.nb_frames})')

        self.is_skel_data_init = skel_data_init is not None
        self.force_recompute = force_recompute

        print('Fitting {} frames'.format(self.nb_frames))
        print('Watching frame: {}'.format(watch_frame))

        # Initialize SKEL torch params
        body_params = self._init_params(betas_in, poses_in, trans_in, skel_data_init)

        # we cut the whole sequence in batches for parallel optimization
        if batch_size > self.nb_frames:
            batch_size = self.nb_frames
            print('Batch size is larger than the number of frames. Setting batch size to {}'.format(batch_size))
        
        n_batch = math.ceil(self.nb_frames / batch_size)
        pbar = trange(n_batch, desc="Running batch optimization")

        # Initialize the res dict to store the per frame result skel parameters
        out_keys = ['poses', 'betas', 'trans']
        if self.export_meshes:
            out_keys += ['skel_v', 'skin_v', 'smpl_v']
        res_dict = {key: [] for key in out_keys}

        res_dict['gender'] = self.gender
        if self.export_meshes:
            res_dict['skel_f'] = self.skel.skel_f.cpu().numpy().copy()
            res_dict['skin_f'] = self.skel.skin_f.cpu().numpy().copy()
            res_dict['smpl_f'] = self.smpl.faces

        # Iterate over the batches to fit the whole sequence
        for i in pbar:  

            if debug:
                # Only run the first batch to test, ignore the rest
                if i > 1:
                    continue
            
            # Get batch start and end indices
            i_start =  i * batch_size
            i_end = min((i+1) * batch_size, self.nb_frames)
            
            # fit the batch
            betas, poses, trans, verts = self._fit_batch(body_params, i, i_start, i_end)

            # store the results
            res_dict['poses'].append(poses)
            res_dict['betas'].append(betas)
            res_dict['trans'].append(trans)

            if self.export_meshes:
                # Store the meshes vertices
                skel_output = self.skel.forward(poses=poses, betas=betas, trans=trans, poses_type='skel', skelmesh=True)
                res_dict['skel_v'].append(skel_output.skel_verts)
                res_dict['skin_v'].append(skel_output.skin_verts)
                res_dict['smpl_v'].append(verts)

            # per_frame
            
            # # Initialize the next frames with current frame
            # body_params['poses_skel'][i_end:] = poses[-1:]
            # body_params['trans_skel'][i_end:] = trans[-1]
            # body_params['betas_skel'][i_end:] = betas[-1:]

        for key, val in res_dict.items():
            if isinstance(val, list):
                 res_dict[key] = torch.cat(val, dim=0).detach().cpu().numpy()
        
        return res_dict
    
    def _init_params(self, betas_smpl, poses_smpl, trans_smpl, skel_data_init=None):
        """ Return initial SKEL parameters from SMPL data dictionary and an optional SKEL data dictionary."""

        # prepare smpl_params
        betas_smpl = to_torch(betas_smpl, self.device)
        poses_smpl = to_torch(poses_smpl, self.device)
        trans_smpl = to_torch(trans_smpl, self.device)


        if skel_data_init is None or self.force_recompute:
            poses_skel = torch.zeros((self.nb_frames, self.skel.num_q_params), device=self.device)
            # poses_skel[:, :3] = poses_smpl[:, :3] # Global orient are similar between SMPL and SKEL, so init with SMPL angles
            
            # transform axis angle to euler angles
            smpl_global_orient = poses_smpl[:, :3]
            rot_mat = axis_angle_to_matrix(smpl_global_orient)
            skel_global_orient = matrix_to_euler_angles(rot_mat, 'XYZ')
            poses_skel[:, :3] = skel_global_orient

            poses_skel[:, 0] = -poses_smpl[:, 0] # axis deffinition is different in SKEL

            betas_skel = torch.zeros((self.nb_frames, 10), device=self.device)
            betas_skel[:] = betas_smpl[..., :10]
            
            trans_skel = trans_smpl # Translation is similar between SMPL and SKEL, so init with SMPL translation
        else:
            # Load from previous alignment
            betas_skel = to_torch(skel_data_init['betas'], self.device)
            poses_skel = to_torch(skel_data_init['poses'], self.device)
            trans_skel = to_torch(skel_data_init['trans'], self.device)
        
        # Make a dictionary out of the necessary body parameters
        body_params = {
            'betas_skel': betas_skel,
            'poses_skel': poses_skel,
            'trans_skel': trans_skel,
            'betas_smpl': betas_smpl,
            'poses_smpl': poses_smpl,
            'trans_smpl': trans_smpl
        }

        return body_params
    

    def _fit_batch(self, body_params_in, i, i_start, i_end):
        """ Create parameters for the batch and run the optimization."""

        # Sample a batch
        # assert body_params_in['betas_smpl'].shape[0] == 1, f"beta_smpl should be of shape 1xF where F is the number of frames, got {body_params_in['betas_smpl'].shape}"
        # body_params -> (i_start: i_end)
        # body_params = {key: val[i_start: i_end] for key, val in body_params_in.items() if key != 'betas_smpl'}
        body_params = {key: val[i_start: i_end] for key, val in body_params_in.items()}
        # body_params['betas_smpl'] = body_params_in['betas_smpl'].clone()

        # SMPL params
        betas_smpl = body_params['betas_smpl']
        poses_smpl = body_params['poses_smpl']
        trans_smpl = body_params['trans_smpl']

        # SKEL params
        betas = to_params(body_params['betas_skel'], device=self.device)
        poses = to_params(body_params['poses_skel'], device=self.device)
        trans = to_params(body_params['trans_skel'], device=self.device)

        if 'verts' in body_params:
            verts = body_params['verts']
        else:
            smpl_output = self.smpl(betas=betas_smpl, body_pose=poses_smpl[:,3:], transl=trans_smpl, global_orient=poses_smpl[:,:3])
            verts = smpl_output.vertices

        # Optimize
        config = self.cfg.optim_steps
        current_cfg = config[0]
        if not self.is_skel_data_init:
        #     # Optimize the global rotation and translation for the initial fitting
            
            with open('log.txt', 'a', encoding='utf-8') as f:
                 f.write(f'Step 0: {current_cfg.description}\n')
            self._optim([trans,poses], poses, betas, trans, verts, current_cfg)
        
        for ci, cfg in enumerate(config[1:]):
            current_cfg.update(cfg)
            with open('log.txt', 'a', encoding='utf-8') as f:
                f.write(f'Step {ci+1}: {current_cfg.description}\n')
            self._optim([poses], poses, betas, trans, verts, current_cfg)
        
        return betas, poses, trans, verts
    
    def _optim(self,
            params,
            poses,
            betas,
            trans,
            verts,
            cfg):
        # regress anatomical joints from SMPL's vertices 
        anat_joints = torch.einsum('bik,ji->bjk', [verts, self.skel.J_regressor_osim])
        dJ=torch.zeros((poses.shape[0], 24, 3), device=betas.device)
        # Create the optimizer -> first is trans, poses then poses
        optimizer = torch.optim.LBFGS(params, 
                                        lr=cfg.lr, 
                                        max_iter=cfg.max_iter, 
                                        line_search_fn=cfg.line_search_fn,  
                                        tolerance_change=cfg.tolerance_change)
        
        poses_init = poses.detach().clone()
        trans_init = trans.detach().clone()

        def closure():
            optimizer.zero_grad()
            # fi = self.watch_frame #frame of the batch to display
            # output = self.skel.forward(poses=poses[fi:fi+1], 
            #                                 betas=betas[fi:fi+1], 
            #                                 trans=trans[fi:fi+1], 
            #                                 poses_type='skel', 
            #                                 dJ=dJ[fi:fi+1],
            #                                 skelmesh=True)
            # self._fstep_plot(output, cfg, verts[fi:fi+1], anat_joints[fi:fi+1], )

            loss_dict = self._fitting_loss(poses,
                                        poses_init,
                                        betas,
                                        trans,
                                        trans_init,
                                        dJ,
                                        anat_joints,
                                        verts,
                                        cfg)
            
            with open('log.txt', 'a', encoding='utf-8') as f:
                 print(pretty_loss_print(loss_dict), file=f, flush=True)

            loss = sum(loss_dict.values())                     
            loss.backward()

            return loss
        
        for step_i in range(cfg.num_steps):
            loss = optimizer.step(closure).item()

    def _fstep_plot(self, output, cfg, verts, anat_joints):
        "Function to plot each step"
        if('DISABLE_VIEWER' in os.environ):
            return
        pass 

    def _get_masks(self, cfg):
        pose_mask = torch.ones((self.skel.num_q_params)).to(self.device).unsqueeze(0)
        verts_mask = torch.ones_like(self.fitting_mask)
        joint_mask = torch.ones((self.skel.num_joints, 3)).to(self.device).unsqueeze(0).bool()

        # Mask vertices
        if cfg.mode == 'root_only':
            # Only optimize the global rotation of the body, i.e. the first 3 angles of the pose
            pose_mask[:] = 0 # Only optimize for the global rotation
            pose_mask[:, :3] = 1

            # Only fit the thorax vertices to recovery the proper body orientation and translation
            verts_mask = self.torso_verts_mask
        
        elif cfg.mode == 'fixed_upper_limbs':
            upper_limbs_joints = [0,1,2,3,6,9,12,15,16,17]
            verts_mask = (self.smpl.lbs_weights[:, upper_limbs_joints] > 0.5).sum(dim=-1) > 0
            verts_mask = verts_mask.unsqueeze(0).unsqueeze(-1)

            joint_mask[:, [3,4,5,8,9,10,18,23], :] = 0 

            pose_mask[:] = 1
            pose_mask[:, :3] = 0  # block the global rotation
            pose_mask[:, 19] = 0  # block the lumbar twist

        elif cfg.mode == 'fixed_root':
            pose_mask[:] = 1  
            pose_mask[:,:3] = 0  # Block the global rotation
                
            # The orientation of the upper limbs is often wrong in SMPL so ignore these vertices for the finale step
            upper_limbs_joints = [1, 2, 16, 17]
            verts_mask = (self.smpl.lbs_weights[:, upper_limbs_joints] > 0.5).sum(dim=-1) > 0
            verts_mask = torch.logical_not(verts_mask)
            verts_mask = verts_mask.unsqueeze(0).unsqueeze(-1)
        
        elif cfg.mode == "free":
            verts_mask = torch.ones_like(self.fitting_mask )

            joint_mask[:]=0
            joint_mask[:, [19,14], :] = 1 # Only fir the scapula join to avoid collapsing shoulders
        
        else:
            raise ValueError(f'Unknown mode {cfg.mode}')

        return pose_mask, verts_mask, joint_mask
    
    def _fitting_loss(self,
                      poses,
                      poses_init,
                      betas,
                      trans,
                      trans_init,
                      dJ,
                      anat_joints,
                      verts,
                      cfg):
        loss_dict = {}
        pose_mask, verts_mask, joint_mask = self._get_masks(cfg) 
        poses = poses * pose_mask + poses_init * (1-pose_mask)

        # Mask joints to not optimize before computing the losses 
        output = self.skel.forward(poses=poses, betas=betas, trans=trans, poses_type='skel', dJ=dJ, skelmesh=False)

        # Fit the SMPL vertices
        # We know the skinning of the forearm and the neck are not perfect,
        # so we create a mask of the SMPL vertices that are important to fit, like the hands and the head
        loss_dict['verts_loss_loose'] = cfg.l_verts_loose \
                *(verts_mask  * (output.skin_verts - verts)**2).sum() / (((verts_mask).sum()*self.nb_frames)) 
        
        # Fit the regressed joints, this avoids collapsing shoulders
        loss_dict['joint_loss'] = cfg.l_joint * (joint_mask * (output.joints - anat_joints)**2).mean()

        # Time consistancy
        if poses.shape[0] > 1:
            # This avoids unstable hips orientationZ
            loss_dict['time_loss'] = cfg.l_time_loss * F.mse_loss(poses[1:], poses[:-1])

        loss_dict['pose_loss'] = cfg.l_pose_loss * compute_pose_loss(poses, poses_init)

        if cfg.use_basic_loss is False:
            # These losses can be used to regularize the optimization but are not always necessary
            loss_dict['anch_rot'] = cfg.l_anch_pose * compute_anchor_pose(poses, poses_init)
            loss_dict['anch_trans'] = cfg.l_anch_trans * compute_anchor_trans(trans, trans_init)

            # Regularize the pose
            loss_dict['scapula_loss'] = cfg.l_scapula_loss * compute_scapula_loss(poses)
            loss_dict['spine_loss'] = cfg.l_spine_loss * compute_spine_loss(poses)

             # Adjust the losses of all the pose regularizations sub losses with the pose_reg_factor value
            for key in ['scapula_loss', 'spine_loss', 'pose_loss']:
                loss_dict[key] = cfg.pose_reg_factor * loss_dict[key]
        
        return loss_dict


    # def _get_mask_()