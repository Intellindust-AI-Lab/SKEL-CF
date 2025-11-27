from typing import Dict
import hydra
import torch 
from tqdm import tqdm
from datasets.hmr_dataset import HmrDataset
from evaluation.mix_evaluator.skelvit_eval import UniformEvaluator
from models.skelvit import SKELViT
from util.data import recursive_to
from util.misc import trans_points2d_parallel
from util.pylogger import get_pylogger
from torch.utils.data import DataLoader

logger = get_pylogger(__name__)

H36M_DATASET_FILE = 'data_inputs/skel-evaluation-labels/h36m_val_p2.npz'
H36M_IMG_FILE = 'data_inputs/skel-evaluation-data/h36m-select'

COCO_IMG_FILE = 'data_inputs/skel-evaluation-data/coco'
COCO_DATASET_FILE = 'data_inputs/skel-evaluation-labels/coco_val.npz'

def get_data(ds_name, cfg):
    assert ds_name in ['H36M-VAL-P2', 'COCO-VAL']

    if 'H36M' in ds_name:
        dataset_file = H36M_DATASET_FILE
        img_file = H36M_IMG_FILE
    elif 'COCO' in ds_name:
        dataset_file = COCO_DATASET_FILE
        img_file = COCO_IMG_FILE
    else: 
        raise ValueError('UnKnown dataset!')
    dataset = HmrDataset(cfg=cfg, dataset_file=dataset_file, img_dir=img_file, ds_name=ds_name, train=False)
    dataset_map = {d.name: d.item for d in cfg.data_configs.eval.datasets}
    dataset._kp_list_ = dataset_map[ds_name].kp_list


    data_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=4)
    return {
        'name': ds_name,
        'dataset': dataset,
        'data_loader': data_loader,
    }


def fix_prefix_state_dict(st: Dict):
    new_st = {}
    for k, v in st.items():
        if k.startswith('model.'):
            new_k = k.replace('model.', '', 1)
            new_st[new_k] = v
    return new_st


def eval_pipeline(model, data_list):
    for data in data_list:
        evaluator = UniformEvaluator(data, 'cuda')

        for batch in tqdm(data['data_loader']):
            batch = recursive_to(batch, 'cuda')

            with torch.no_grad():
                _, out, _ = model(batch)

            pd_kp2d = out['pd_kp2d']
            pd_kp2d_cropped = trans_points2d_parallel(pd_kp2d, batch['_trans'])
            pd_kp2d_cropped = pd_kp2d_cropped / 256 - 0.5
            output = {
                'pred_keypoints_2d' : pd_kp2d_cropped,  
                'pred_keypoints_3d' : out['pd_kp3d']
            }
            evaluator.eval(pd=output, gt=batch)
            
        ds_name = data['name']
        results = evaluator.get_results()
        logger.info(f'{ds_name}:\n{results}')





@hydra.main(version_base='1.2', config_path="config", config_name="eval.yaml")
def main(cfg):
    ds_conf = cfg.eval_list.datasets
    # 1. Load data.
    data_list = []
    ds_list = ds_conf.split('_')
    for ds_name in ds_list:
        data = get_data(ds_name, cfg)
        data_list.append(data)

        logger.info(f'Will test on {ds_name}')

    # 2. Prepare the pipeline.
    ckpt_paths = cfg.trainer.ckpt_path
    model = SKELViT(cfg)
    
    st = torch.load(ckpt_paths, map_location='cpu', weights_only=False)['ema_model']
    st = fix_prefix_state_dict(st)
    missing, unexpected = model.load_state_dict(st, strict=False)
    logger.info("Missing: " + str(missing))
    logger.info("Unexpected: " + str(unexpected))
    model = model.to('cuda')
    model.eval()

    # 3. Evaluation.
    eval_pipeline(model, data_list)


if __name__ == '__main__':
    main()