# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
PYTHONPATH=. torchrun   run_hsmr_test.py \
                        trainer.test_batch_size=64 \
                        general.num_workers=12 \
                        hub.skel_head.transformer_decoder.context_dim=1280 \
                        policy.img_patch_size=256 \
                        trainer.ckpt_path=''\
                        exp_name=''



                                        