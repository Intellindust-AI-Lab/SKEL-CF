import os
from pathlib import Path
from pydoc import render_doc
from typing import List, Optional, Union 
from tqdm import tqdm
import cv2

from vis.video_utils import load_video, save_video 

class VideoPrecessor:

    def __init__(self, cfg):
        self.cfg = cfg 
    

    def extract_frames(
        self,
        video_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        frame_prefix: str = 'frame',
        image_format: str = 'jpg',
        start_frame: int = 0,
        end_frame: int = None,
        step: int = 1
    ) -> Path:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"The vide file is not found: {video_path}")
        
        # confir the directory 
        if output_dir is None:
            output_dir = video_path.parent / f"{video_path.stem}_frames"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading video in {video_path} and saving frames to {output_dir}")

        frames, video_meta = load_video(video_path)
        
        print(f"video info: {video_meta['L']} frames, {video_meta['w']}x{video_meta['h']}, {video_meta['fps']} FPS")
        
        total_frames = len(frames)
        start_frame = start_frame if start_frame is not None else 0
        end_frame = end_frame if end_frame is not None else total_frames
        end_frame = min(end_frame, total_frames)

        # extract frames 
        extracted_files = []
        frames_indices = range(start_frame, end_frame, step)

        print(f"Strat extract frames (from {start_frame} till {end_frame} frame, Step size {step})...")
        for idx in tqdm(frames_indices, desc="Extracting frames"):
            frame = frames[idx]

            # save image 
            frame_filename = f"{frame_prefix}_{idx:06d}.{image_format}"
            frame_path = output_dir / frame_filename
            
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            extracted_files.append(frame_path)
        
        print(f"✅ successfully extracted {len(extracted_files)} frames to {output_dir}")
        return output_dir

    
    def extract_and_render(
        self,
        video_path : Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        renderer: Optional[object] = None,
        **kwargs
    ) -> List[Path]:
        
        extracted_files = self.extract_frames(video_path, output_dir, **kwargs)

        assert renderer is not None, "Renderer is required"
        print("Rendering frames...")
        rendered_files = []

        for img_path in tqdm(extracted_files, desc="Rendering frames"):
            
            renderer.process_image(
                
            )
            rendered_files.append(img_path)
        

        return rendered_files
    
    def frames_to_video(
        self, 
        frames_dir: Union[str, Path],
        output_video_path: Union[str, Path],
        fps: int = 30,
        pattern: str = "*_concat.jpg"
    ):
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"The frames directory is not found: {frames_dir}")
        
        frames_file = sorted(frames_dir.glob(pattern))

        if len(frames_file) == 0:
            raise FileNotFoundError(f"No frames found in {frames_dir} with pattern {pattern}")
        
        print(f"Loadding {len(frames_file)} images")

        frames = []
        
        for frame_file in tqdm(frames_file, desc="Loadding frames"):
            img = cv2.imread(str(frame_file))

            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        if len(frames) == 0:
            raise ValueError(f"No valid frames found in {frames_dir} with pattern {pattern}")
        
        print(f"Successfully loaded {len(frames)} frames")
        save_video(frames, output_video_path, fps=fps)
        print(f"✅ successfully converted {len(frames)} frames to video: {output_video_path}")

        return output_video_path
