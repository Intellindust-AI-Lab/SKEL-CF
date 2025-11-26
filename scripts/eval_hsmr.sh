# export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12345  run_hsmr_test.py \
                                        trainer.test_batch_size=64 \
                                        exp_name=1113-final-version-evaluation-hsmr-all\
                                        general.num_workers=12 \
                                        hub.skel_head.transformer_decoder.context_dim=1280 \
                                        policy.img_patch_size=256 \
                                        trainer.ckpt_path=/public_hw/home/cit_yuzeng/skelvit/data_outputs/exp/1105_aux_loss_base_0.1/checkpoints/best.pth



                                        