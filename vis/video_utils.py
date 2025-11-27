
import cv2
import imageio
import numpy as np
from typing import Union, Tuple, List
from pathlib import Path
from tqdm import tqdm

def load_video(video_path: Union[str, Path]):
    if isinstance(video_path, str):
        video_path = Path(video_path)
    
    assert video_path.exists(), f'Video not found: {video_path}'
    
    if video_path.is_dir():
        print(f'Found {video_path} is a directory. Checking for video files...')
        

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
        video_files = []
        for ext in video_extensions:
            video_files.extend(video_path.glob(f'*{ext}'))
            video_files.extend(video_path.glob(f'*{ext.upper()}'))
        
        if video_files:
            print(f'Found {len(video_files)} video files in directory.')

            video_file = sorted(video_files)[0]
            print(f'Processing first video: {video_file}')
            reader = imageio.get_reader(video_file, format='FFMPEG')
            frames = []
            for frame in tqdm(reader, total=reader.count_frames()):
                frames.append(frame)
            fps = reader.get_meta_data()['fps']
        else:

            print(f'No video files found. Treating as image folder.')
            imgs_path = sorted(list(video_path.glob('*.*')))
            imgs_path = [p for p in imgs_path if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
            frames = []
            for img_path in tqdm(imgs_path):
                frames.append(imageio.imread(img_path))
            fps = 30  # default fps
    else:
        print(f'Found {video_path} is a file. It will be regarded as a video file.')
        reader = imageio.get_reader(video_path, format='FFMPEG')
        frames = []
        for frame in tqdm(reader, total=reader.count_frames()):
            frames.append(frame)
        fps = reader.get_meta_data()['fps']
    
    if len(frames) == 0:
        raise ValueError(f'No frames found in {video_path}. Please check if the directory contains valid video files or images.')
    
    frames = np.stack(frames, axis=0)  # (L, H, W, 3)
    meta = {
        'fps': fps,
        'w': frames.shape[2],
        'h': frames.shape[1],
        'L': frames.shape[0],
    }
    
    return frames, meta


def get_video_files(video_path: Union[str, Path]) -> List[Path]:

    if isinstance(video_path, str):
        video_path = Path(video_path)
    
    if not video_path.is_dir():
        return [video_path] if video_path.exists() else []
    
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_path.glob(f'*{ext}'))
        video_files.extend(video_path.glob(f'*{ext.upper()}'))
    
    return sorted(video_files)


def save_video(
    frames: Union[np.ndarray, List[np.ndarray]],
    output_path: Union[str, Path],
    fps: float = 30,
    resize_ratio: Union[float, None] = None,
    quality: Union[int, None] = None,
):

    if isinstance(frames, list):
        frames = np.stack(frames, axis=0)
    assert frames.ndim == 4, f'Invalid frames shape: {frames.shape}'
    
    if resize_ratio is not None:
        frames = flex_resize_video(frames, ratio=resize_ratio)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    writer = imageio.get_writer(output_path, fps=fps, quality=quality)
    output_seq_name = str(output_path).split('/')[-1]
    for frame in tqdm(frames, desc=f'Saving {output_seq_name}'):
        writer.append_data(frame)
    writer.close()


def flex_resize_video(
    frames: np.ndarray,
    tgt_wh: Union[Tuple[int, int], None] = None,
    ratio: Union[float, None] = None,
    kp_mod: int = 1,
):

    assert tgt_wh is not None or ratio is not None, 'At least one of tgt_wh and ratio must be set.'
    if tgt_wh is not None:
        assert len(tgt_wh) == 2, 'tgt_wh must be a tuple of 2 elements.'
        assert tgt_wh[0] > 0 or tgt_wh[1] > 0, 'At least one of width and height must be positive.'
    if ratio is not None:
        assert ratio > 0, 'ratio must be positive.'
    assert len(frames.shape) == 4, 'frames must have 4 dimensions.'
    
    def align_size(val: float):

        return int(round(val / kp_mod) * kp_mod)
    

    orig_h, orig_w = frames.shape[1], frames.shape[2]
    tgt_wh = (int(orig_w * ratio), int(orig_h * ratio)) if tgt_wh is None else tgt_wh
    tgt_w, tgt_h = tgt_wh
    tgt_w = align_size(orig_w * tgt_h / orig_h) if tgt_w == -1 else align_size(tgt_w)
    tgt_h = align_size(orig_h * tgt_w / orig_w) if tgt_h == -1 else align_size(tgt_h)
    

    resized_frames = np.stack([cv2.resize(frame, (tgt_w, tgt_h)) for frame in frames])
    
    return resized_frames


def flex_resize_img(
    img: np.ndarray,
    tgt_wh: Union[Tuple[int, int], None] = None,
    ratio: Union[float, None] = None,
    kp_mod: int = 1,
):

    assert len(img.shape) == 3, 'img must have 3 dimensions.'
    return flex_resize_video(img[None], tgt_wh, ratio, kp_mod)[0]


def splice_img(
    img_grids: Union[List[np.ndarray], np.ndarray],
    grid_ids: Union[List[int], np.ndarray],
):

    if isinstance(img_grids, list):
        img_grids = np.stack(img_grids)
    if isinstance(grid_ids, list):
        grid_ids = np.array(grid_ids)
    
    assert len(img_grids.shape) == 4, 'img_grids must be in shape (K, H, W, 3).'
    return splice_video(img_grids[:, None], grid_ids)[0]


def splice_video(
    video_grids: Union[List[np.ndarray], np.ndarray],
    grid_ids: Union[List[int], np.ndarray],
):

    if isinstance(video_grids, list):
        video_grids = np.stack(video_grids)
    if isinstance(grid_ids, list):
        grid_ids = np.array(grid_ids)
    
    assert len(video_grids.shape) == 5, 'video_grids must be in shape (K, L, H, W, 3).'
    assert len(grid_ids.shape) == 2, 'grid_ids must be a 2D matrix.'

    K, L, H, W, C = video_grids.shape
    Y, X = grid_ids.shape

    spliced_video = np.zeros((L, H*Y, W*X, C), dtype=np.uint8)
    for x in range(X):
        for y in range(Y):
            grid_id = grid_ids[y, x]
            if grid_id == -1:
                continue
            spliced_video[:, y*H:(y+1)*H, x*W:(x+1)*W, :] = video_grids[grid_id]
    
    return spliced_video