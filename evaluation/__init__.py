from .metrics import *
from .evaluators import EvaluatorBase

# from lib.body_models.smpl_utils.reality import eval_rot_delta as smpl_eval_rot_delta

class HSMR3DEvaluator(EvaluatorBase):

    def eval(self, **kwargs):
        # Get the predictions and ground truths.
        pd, gt = kwargs['pd'], kwargs['gt']
        v3d_pd, v3d_gt = pd['v3d_pose'], gt['v3d_pose']
        j3d_pd, j3d_gt = pd['j3d_pose'], gt['j3d_pose']

        # Compute the metrics.
        mpjpe    = eval_MPxE(j3d_pd, j3d_gt)
        pa_mpjpe = eval_PA_MPxE(j3d_pd, j3d_gt)
        mpve     = eval_MPxE(v3d_pd, v3d_gt)
        pa_mpve  = eval_PA_MPxE(v3d_pd, v3d_gt)

        # Append the results.
        self.accumulator['MPJPE'].append(mpjpe.detach().cpu())
        self.accumulator['PA-MPJPE'].append(pa_mpjpe.detach().cpu())
        self.accumulator['MPVE'].append(mpve.detach().cpu())
        self.accumulator['PA-MPVE'].append(pa_mpve.detach().cpu())



class SMPLRealityEvaluator(EvaluatorBase):

    def eval(self, **kwargs):
        # Get the predictions and ground truths.
        pd = kwargs['pd']
        body_pose = pd['body_pose']  # (..., 23, 3)

        # Compute the metrics.
        vio_d0  = smpl_eval_rot_delta(body_pose, tol_deg=0 )  # {k: (N, 3)}
        vio_d5  = smpl_eval_rot_delta(body_pose, tol_deg=5 )  # {k: (N, 3)}
        vio_d10 = smpl_eval_rot_delta(body_pose, tol_deg=10)  # {k: (N, 3)}
        vio_d20 = smpl_eval_rot_delta(body_pose, tol_deg=20)  # {k: (N, 3)}
        vio_d30 = smpl_eval_rot_delta(body_pose, tol_deg=30)  # {k: (N, 3)}

        # Append the results.
        parts = vio_d5.keys()
        for part in parts:
            self.accumulator[f'VD0_{part}' ].append(vio_d0[part].max(-1)[0].detach().cpu())   # (N,)
            self.accumulator[f'VD5_{part}' ].append(vio_d5[part].max(-1)[0].detach().cpu())   # (N,)
            self.accumulator[f'VD10_{part}'].append(vio_d10[part].max(-1)[0].detach().cpu())  # (N,)
            self.accumulator[f'VD20_{part}'].append(vio_d20[part].max(-1)[0].detach().cpu())  # (N,)
            self.accumulator[f'VD30_{part}'].append(vio_d30[part].max(-1)[0].detach().cpu())  # (N,)

    def get_results(self, chosen_metric=None):
        ''' Get the current mean results. '''
        # Only chosen metrics will be compacted and returned.
        compacted = self._compact_accumulator(chosen_metric)
        ret = {}
        for k, v in compacted.items():
            vio_max = v.max()
            vio_mean = v.mean()
            vio_median = v.median()
            tot_cnt = len(v)
            vio_cnt = (v > 0).float().sum()
            vio_p = vio_cnt / tot_cnt
            ret[f'{k}_max'] = vio_max.item()
            ret[f'{k}_mean'] = vio_mean.item()
            ret[f'{k}_median'] = vio_median.item()
            ret[f'{k}_percentage'] = vio_p.item()
        return ret


# from lib.body_models.skel_utils.reality import eval_rot_delta as skel_eval_rot_delta

class SKELRealityEvaluator(SMPLRealityEvaluator):

    def eval(self, **kwargs):
        # Get the predictions and ground truths.
        pd = kwargs['pd']
        poses = pd['poses']  # (..., 46)

        # Compute the metrics.
        vio_d0  = skel_eval_rot_delta(poses, tol_deg=0 )  # {k: (N, 3)}
        vio_d5  = skel_eval_rot_delta(poses, tol_deg=5 )  # {k: (N, 3)}
        vio_d10 = skel_eval_rot_delta(poses, tol_deg=10)  # {k: (N, 3)}
        vio_d20 = skel_eval_rot_delta(poses, tol_deg=20)  # {k: (N, 3)}
        vio_d30 = skel_eval_rot_delta(poses, tol_deg=30)  # {k: (N, 3)}

        # Append the results.
        parts = vio_d5.keys()
        for part in parts:
            self.accumulator[f'VD0_{part}' ].append(vio_d0[part].max(-1)[0].detach().cpu())   # (N,)
            self.accumulator[f'VD5_{part}' ].append(vio_d5[part].max(-1)[0].detach().cpu())   # (N,)
            self.accumulator[f'VD10_{part}'].append(vio_d10[part].max(-1)[0].detach().cpu())  # (N,)
            self.accumulator[f'VD20_{part}'].append(vio_d20[part].max(-1)[0].detach().cpu())  # (N,)
            self.accumulator[f'VD30_{part}'].append(vio_d30[part].max(-1)[0].detach().cpu())  # (N,)
