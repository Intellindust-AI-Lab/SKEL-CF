from pathlib import Path
import hydra
from vis.mesh_estimator import HumanMeshEstimator
from vis.video_precessor import VideoPrecessor





@hydra.main(version_base='1.2', config_path="../config", config_name="vis.yaml")
def main(cfg):
    processor = VideoPrecessor(cfg)

    # extrat frames 
    frames_folder, fps = processor.extract_frames(
        video_path=cfg.misc.video_path,
    )
    
    estimator = HumanMeshEstimator(cfg=cfg)
    video_path = Path(cfg.misc.video_path)

    estimator.run_on_images(
        frames_folder, 
        frames_folder.parent / f"{video_path.stem}_rendered"
    )
    # frames to video
    processor.frames_to_video(
        frames_dir=frames_folder.parent / f"{video_path.stem}_rendered",
        output_video_path=cfg.misc.to_video,
        fps=fps,
        pattern=cfg.misc.pattern
    )


if __name__ == "__main__":
    main()