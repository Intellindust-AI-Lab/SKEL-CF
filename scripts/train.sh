export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=8
torchrun  run_train.py \
            trainer.batch_size=64 \
            exp_name=1127-train-test \
            trainer.max_lr=1e-5 \
            trainer.min_lr=1e-5 \
            trainer.total_epochs=30 \
            general.num_workers=10 \
            logger.logger_interval=500 \
            trainer.ema_decaybase=0.9995 \
            hub.skel_head.transformer_decoder.context_dim=1280 \
            trainer.ema_tau=6000 \
            policy.img_patch_size=256 \
            trainer.lamda=1. \
            misc.aux_loss_weight=0.1 
                                        