# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
export HYDRA_FULL_ERROR=1
torchrun --nproc_per_node=1  run_test.py \
                            general.num_workers=12 \
                            hub.skel_head.transformer_decoder.context_dim=1280 \
                            policy.img_patch_size=256 \
                            trainer.test_batch_size=256 \
                            exp_name='' \
                            trainer.ckpt_path='' 
                                        


                                        