from omegaconf import ListConfig
import torch
from datasets.dataset_train import DatasetTrainAllLabels
from datasets.dataset_val import DatasetVal



def build_dataset(cfg, image_set='train'):

    if image_set == 'train' and cfg.DATASETS.DATASETS_AND_RATIOS:
        if isinstance(cfg.DATASETS.DATASETS_AND_RATIOS, ListConfig):
            dataset_names = cfg.DATASETS.DATASETS_AND_RATIOS
        else:
            raise ValueError(f"Unknown dataset type: {type(cfg.DATASETS.DATASETS_AND_RATIOS)}")
        dataset_list = [DatasetTrainAllLabels(cfg, ds, is_train=True) for ds in dataset_names]
        train_ds = torch.utils.data.ConcatDataset(dataset_list)
        return train_ds

    elif image_set == 'val':
        dataset_list = []
        if cfg.DATASETS.VAL_DATASETS is not None:
            dataset_names = cfg.DATASETS.VAL_DATASETS.split('_')
            dataset_list = [DatasetVal(cfg, ds, is_train=False) for ds in dataset_names]

        return dataset_list
    else:
        raise ValueError(f"Unknown image set: {image_set}")