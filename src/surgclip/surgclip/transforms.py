from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_MEAN = (0.5, 0.5, 0.5)
_STD = (0.5, 0.5, 0.5)


def get_preprocess(image_res: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(
            (image_res, image_res),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.Lambda(lambda img: torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0)),
        transforms.Normalize(_MEAN, _STD),
    ])


class VideoPreprocessor:
    """
    Loads, samples, and preprocesses a video clip into a (1, T, C, H, W) tensor.

    Accepts either:
      - A frame path (str or Path): loads neighboring frames from the same directory.
      - A list of PIL images or numpy frames: uses them directly.

    Args:
        num_frames:  Number of frames in the output clip.
        image_res:   Spatial resolution to resize each frame to.
        sample_rate: Stride when reading frames from disk (only used with frame_path).
        mode:        Window selection when given a frame_path.
                       "centered" — window centered on the anchor frame (offline).
                       "online"   — anchor frame is the last in the window.
        sample_type: How to sub-sample if the loaded window exceeds num_frames.
                       "uniform" — evenly spaced across the window.
                       "first"   — take the first num_frames.
                       "middle"  — take the central num_frames.

    Example — from a frame path:
        proc  = VideoPreprocessor(num_frames=8, mode="online")
        video = proc("frames/frame_0050.jpg").to(device)  # (1, 8, 3, 224, 224)

    Example — from a list of PIL images:
        proc  = VideoPreprocessor(num_frames=8)
        video = proc([Image.open(f) for f in frame_paths]).to(device)
    """

    def __init__(
        self,
        num_frames: int = 16,
        image_res: int = 224,
        sample_rate: int = 1,
        mode: str = "centered",
        sample_type: str = "uniform",
    ):
        self.num_frames = num_frames
        self.sample_rate = sample_rate
        self.mode = mode
        self.sample_type = sample_type
        self._transform = get_preprocess(image_res)

    def __call__(
        self,
        video: Union[str, Path, list, np.ndarray],
        add_batch_dim: bool = True,
    ) -> torch.Tensor:
        frames = self._load(video)
        frames = self._sample(frames)
        clip = torch.stack([self._transform(f) for f in frames], dim=0)  # (T, C, H, W)
        return clip.unsqueeze(0) if add_batch_dim else clip

    def _load(self, video) -> list:
        # frame path: load neighbors from disk 
        if isinstance(video, (str, Path)):
            frame_path = Path(video)
            all_frames = sorted(
                f for f in frame_path.parent.glob("*")
                if f.suffix.lower() in _IMAGE_EXTENSIONS
            )
            all_frames = all_frames[::self.sample_rate]

            anchor_idx = next((i for i, f in enumerate(all_frames) if f.name == frame_path.name), None)
            if anchor_idx is None:
                raise ValueError(
                    f"{frame_path.name} not found in directory after applying sample_rate={self.sample_rate}."
                )

            if self.mode == "centered":
                half = self.num_frames // 2
                start = max(0, anchor_idx - half)
                end = start + self.num_frames
                if end > len(all_frames):
                    end = len(all_frames)
                    start = max(0, end - self.num_frames)
            elif self.mode == "online":
                end = anchor_idx + 1
                start = max(0, end - self.num_frames)
            else:
                raise ValueError(f"Unknown mode '{self.mode}'. Choose: 'centered' or 'online'.")

            return [Image.open(f).convert("RGB") for f in all_frames[start:end]]

        # list of PIL images or numpy frames
        if isinstance(video, list):
            if not video:
                raise ValueError("Empty frame list.")
            if isinstance(video[0], np.ndarray):
                return [Image.fromarray(f).convert("RGB") for f in video]
            if isinstance(video[0], Image.Image):
                return [f.convert("RGB") for f in video]
            raise TypeError(f"Unsupported list element type: {type(video[0])}")

        # (T, H, W, C)
        if isinstance(video, np.ndarray):
            if video.ndim != 4:
                raise ValueError(f"numpy video must be (T, H, W, C), got {video.shape}")
            return [Image.fromarray(video[i]).convert("RGB") for i in range(video.shape[0])]

        raise TypeError(f"Unsupported input type: {type(video)}")

    def _sample(self, frames: list) -> list:
        "Takes the loaded list of frames and ensures the output is exactly num_frames long"
        T, N = len(frames), self.num_frames
        if T <= N:
            # Pad by repeating last frame if clip is too short
            return frames + [frames[-1]] * (N - T)
        if self.sample_type == "uniform":
            indices = np.linspace(0, T - 1, N, dtype=int)
        elif self.sample_type == "first":
            indices = np.arange(N)
        elif self.sample_type == "middle":
            start = (T - N) // 2
            indices = np.arange(start, start + N)
        else:
            raise ValueError(f"Unknown sample_type '{self.sample_type}'. Choose: uniform, first, middle.")
        return [frames[i] for i in indices]