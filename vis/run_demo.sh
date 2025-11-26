CUDA_VISIBLE_DEVICES=8 PYTHONPATH=. python vis_cam/run_demo.py \
                  trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                  hub.skel_head.transformer_decoder.context_dim=1280 \
                  misc.image_folder=/data1032/ld_data/skelvit/data_inputs/teaser-nb\
                  misc.output_folder=/data1032/ld_data/skelvit/data_outputs/teaser-nb-white-bg \
                  trainer.backbone=vitpose_h











    