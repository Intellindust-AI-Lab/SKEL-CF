# b1
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/badminton01.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/badminton01-rendered.mp4 \
                misc.num_keep=2 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280 

# b2
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/badminton02.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/badminton02-rendered.mp4 \
                misc.num_keep=2 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280
# b3
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/badminton03.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/badminton03-rendered.mp4 \
                misc.num_keep=2 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280

# g1
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/gymnastics01.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/gymnastics01-rendered.mp4 \
                misc.num_keep=1 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280

# g2
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/gymnastics02.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/gymnastics02-rendered.mp4 \
                misc.num_keep=1 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280
# g3
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/gymnastics03.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/gymnastics03-rendered.mp4 \
                misc.num_keep=1 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280
# ping 1
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/pingpang01.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/pingpang01-rendered.mp4 \
                misc.num_keep=2 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280
# ping 2
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/pingpang02.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/pingpang02-rendered.mp4 \
                misc.num_keep=2 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280
# ping 3
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/pingpang03.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/pingpang03-rendered.mp4 \
                misc.num_keep=2 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280

# skate 1
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/skate01.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/skate01-rendered.mp4 \
                misc.num_keep=1 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280

# skate 2
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/skate02.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/skate02-rendered.mp4 \
                misc.num_keep=1 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280
# skate 3
CUDA_VISIBLE_DEVICES=9 PYTHONPATH=. python vis_cam/run_video.py \
                misc.video_path=/data1032/ld_data/skelvit/data_inputs/videos-demo/skate03.mp4 \
                misc.to_video=/data1032/ld_data/skelvit/data_outputs/videos-demo/skate03-rendered.mp4 \
                misc.num_keep=1 \
                misc.pattern="*_concat.jpg" \
                trainer.ckpt_path=/data1032/ld_data/skelvit/data_inputs/1110-aux0.1.pth \
                trainer.backbone=vitpose_h \
                hub.skel_head.transformer_decoder.context_dim=1280








    









    








    








    









    









    









    








    








    







    









    









    