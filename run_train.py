# -*- coding: utf-8 -*-
import os
import random
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import hydra
import numpy as np
import torch
from datasets import build_dataset
from datasets.moyo_datasets import EvalMoyoDataset
from engine import evaluate_hmr2, evaluate_moyo, train_one_epoch
from models.ema import EmaModel
from models.skelvit import build_model
from run_hsmr_test import get_data
import util.misc as utils
from util.pylogger import get_pylogger 
from torch.utils.data import DistributedSampler, DataLoader

@hydra.main(version_base='1.2', config_path="config", config_name="train.yaml")
def main(cfg):

    utils.init_distributed_mode(cfg)
    
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    logger = get_pylogger(__name__)

    if utils.get_rank() == 0:
        writter = SummaryWriter(output_dir / 'tensorboards')
        writter_ema = SummaryWriter(output_dir / 'tensorboards/ema')
    else:
        writter = None
        writter_ema = None
        
    # os.local_rank 
    device = torch.device(f'cuda:{cfg.gpu}')
    # fix the seed for reproducibility, different per gpu
    seed = cfg.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    model, criterion = build_model(cfg)
    model.to(device)

    model_without_ddp = model
    if cfg.distributed:
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], find_unused_parameters=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.gpu], gradient_as_bucket_view=True)
        model_without_ddp = model.module

    ema_model = EmaModel(cfg, model_without_ddp)

    n_parameters = sum(p.numel() for p in model_without_ddp.get_trainable_parameters() if p.requires_grad) / 1_000_000
    logger.info(f'number of params: {n_parameters} M')
    
    # train: concat_dataset, val: first dataset of dataset_list
    dataset_train = build_dataset(cfg=cfg, image_set='train')
    # dataset_val = build_dataset(cfg=cfg, image_set='val')[0]

    if cfg.distributed:
        sampler_train = DistributedSampler(dataset_train)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, cfg.trainer.batch_size, drop_last=True)

    # xuele-optim, use pin_memory to speedup cpu to gpu.
    data_loader_train = DataLoader(dataset_train, 
                                    batch_sampler=batch_sampler_train, 
                                    num_workers=cfg.general.num_workers,
                                    pin_memory=True,
                                    persistent_workers=True if cfg.general.num_workers > 0 else False,
                                    prefetch_factor=2)
    # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=cfg.general.num_workers)
    # data_loader_val = DataLoader(dataset_val, cfg.trainer.test_batch_size, sampler=sampler_val, drop_last=False, num_workers=cfg.general.num_workers)
   
    # load evaluation data     
    ds_conf = cfg.eval_list.datasets
    # 2. Load data.
    data_list = []
    ds_list = ds_conf.split('_')
    for ds_name in ds_list:
        logger.info(f'Will test on {ds_name}')
        data = get_data(ds_name, cfg)
        data_list.append(data)

    # 3. evalaute moyo 
    moyo_name = cfg.trainer.moyo_name
    moyo_name = moyo_name + '.npz' 
    moyo_name = Path(moyo_name)
    npz_file = Path(cfg.paths.data_inputs) / 'skel-evaluation-labels' / moyo_name 
    moyo_dataset = EvalMoyoDataset(cfg, npz_fn=npz_file, ignore_img=False)
    moyo_data_loader = DataLoader(moyo_dataset, cfg.trainer.test_batch_size, drop_last=False, num_workers=cfg.general.num_workers)

    param_dicts = [
        {
            "params": [p for p in model_without_ddp.get_trainable_parameters() if p.requires_grad]
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.trainer.max_lr, weight_decay=cfg.trainer.weight_decay)

    # only warmup and constant after 
    warmup_steps = cfg.trainer.warmup_epochs * len(data_loader_train)
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            return 1.0
        
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info("Start training")
    start_epoch = 0
    best_pve = float('inf')
    for epoch in range(start_epoch, cfg.trainer.total_epochs):
        
        if cfg.distributed:
            sampler_train.set_epoch(epoch)
        
        train_one_epoch(
            cfg=cfg, model=model, ema_model=ema_model, 
            criterion=criterion, data_loader=data_loader_train, optimizer=optimizer, 
            lr_scheduler=lr_scheduler, device=device, writter=writter,epoch=epoch,

        )
        
     
        evaluate_hmr2(
            cfg=cfg, model=model_without_ddp, data_list=data_list
        )
        # evaluate moyo_hard
        mpjpe, pampjpe, pve, _= evaluate_moyo(cfg, model_without_ddp, moyo_data_loader, device, writter, epoch, ema=False)
        
        # evaluate ema Model
        evaluate_hmr2(
            cfg=cfg, model=ema_model.model, data_list=data_list
        )
        # evaluate moyo_hard
        ema_mpjpe, ema_pampjpe, ema_pve, _ =  evaluate_moyo(cfg, ema_model.model, moyo_data_loader, device, writter_ema, epoch, ema=True)

        
        if cfg.output_dir:
            if ema_pve < best_pve:
                best_pve = ema_pve
                ckpt_path = output_dir / 'checkpoints/best.pth'
                logger.info(f'BEST epoch {epoch}: MPJPE->{mpjpe:.4f} PAMPJPE->{pampjpe:.4f} PVE->{pve:.4f}')
                logger.info(f'BEST EMA epoch {epoch}:  MPJPE->{ema_mpjpe:.4f} PAMPJPE->{ema_pampjpe:.4f} PVE->{ema_pve:.4f}')
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'ema_model': ema_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'cfg': cfg,
                }, ckpt_path)

        
if __name__ == '__main__':
    main()
