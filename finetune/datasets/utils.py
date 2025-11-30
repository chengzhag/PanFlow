import logging
from pathlib import Path
from typing import List, Tuple, Dict
import random
import json
import numpy as np
from einops import rearrange, repeat

import cv2
import torch
from torchvision.transforms.functional import resize


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip



##########  loaders  ##########


def load_prompts(prompt_path: Path) -> List[str]:
    with open(prompt_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> List[Path]:
    with open(video_path, "r", encoding="utf-8") as file:
        return [
            video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images(image_path: Path) -> List[Path]:
    with open(image_path, "r", encoding="utf-8") as file:
        return [
            image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images_from_videos(videos_path: List[Path]) -> List[Path]:
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        frame_path = first_frames_dir / f"{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with same name as video
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths


##########  samplers  ##########


def sample_clip(
    clip: Dict,
    rng: random.Random,
    sample_n_frames: int,
    stride: Tuple[int, int],
) -> Dict:
    clip_begin, clip_end = clip["frames"]
    clip_end += 1
    total_frames = clip_end - clip_begin
    stride_min, stride_max = stride
    stride_max = min(stride_max, total_frames // sample_n_frames)
    stride = rng.randint(stride_min, stride_max)
    sample_frames = sample_n_frames * stride - stride + 1
    sample_begin = rng.randint(clip_begin, max(clip_begin, clip_end - sample_frames))
    sample_end = sample_begin + sample_frames - 1
    return {
        "frames": (sample_begin, sample_end),
        "stride": stride,
    }


##########  preprocessors  ##########


def preprocess_image_with_resize(
    image_path: Path | str,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single image.

    Args:
        image_path: Path to the image file.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] where:
            C = number of channels (3 for RGB)
            H = height
            W = width
    """
    if isinstance(image_path, str):
        image_path = Path(image_path)
    image = cv2.imread(image_path.as_posix())
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height))
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).contiguous()
    return image


def preprocess_video_with_resize(
    video_path: Path | str,
    max_num_frames: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    if isinstance(video_path, str):
        video_path = Path(video_path)
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(uri=video_path.as_posix(), width=width, height=height)
        video_num_frames = len(video_reader)
        if video_num_frames < max_num_frames:
            # Get all frames first
            frames = video_reader.get_batch(list(range(video_num_frames)))
            # Repeat the last frame until we reach max_num_frames
            last_frame = frames[-1:]
            num_repeats = max_num_frames - video_num_frames
            repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
            frames = torch.cat([frames, repeated_frames], dim=0)
        else:
            indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
            frames = video_reader.get_batch(indices)
            return frames


def preprocess_video_with_sampling(
    video_path: Path | str,
    frame_range: Tuple[int, int],
    frames: int,
    height: int = -1,
    width: int = -1,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video_path: Path to the video file.
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(
            str(video_path),
            width=width,
            height=height,
            # ctx=decord.cpu(0),
            num_threads=1,
        )
        sample_begin, sample_end = frame_range
        frame_indices = np.linspace(
            sample_begin, sample_end, frames, dtype=int
        )
        frame_range = video_reader.get_batch(frame_indices)
        return frame_range


def preprocess_pose_with_sampling(
    pose_path: Path | str,
    frame_range: Tuple[int, int],
    frames: int,
) -> torch.Tensor:
    c2w = np.load(pose_path)
    sample_begin, sample_end = frame_range
    frame_indices = np.linspace(
        sample_begin, sample_end, frames, dtype=int
    )
    c2w = c2w[frame_indices]
    c2w_4x4 = np.eye(4, dtype=np.float32)
    c2w = np.hstack((c2w, repeat(c2w_4x4[-1], "n -> f 1 n", f=c2w.shape[0])))
    w2c0 = np.linalg.inv(c2w[0])
    c2w = w2c0 @ c2w
    c2w = torch.from_numpy(c2w)
    return c2w
