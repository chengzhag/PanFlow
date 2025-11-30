from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Optional, Literal
from functools import cache, cached_property
from abc import abstractmethod
import yaml

import torch
from accelerate.logging import get_logger
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from einops import rearrange, einsum
import random
import json
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
from equilib import equi2equi
from diffusers.utils.export_utils import export_to_video
import inspect

from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.schemas import DataArgs
from ..utils.erp_utils import transformation_to_flow, equilib_rotation
from thirdparty.PanoFlowAPI.utils.pano_vis import better_flow_to_image
from thirdparty.PanoFlowAPI.utils.pano_vis import flow_to_arrows

import rp
import finetune.utils.noise_warp_loop as nw
from thirdparty.go_with_the_flow.cut_and_drag_inference import get_downtemp_noise

from .utils import (
    sample_clip,
    preprocess_video_with_sampling,
    preprocess_pose_with_sampling,
)


if TYPE_CHECKING:
    from finetune.trainer import Trainer

# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip


logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseDataset(Dataset):
    flow_size = (512, 1024)
    video_transform = transforms.Compose([
        transforms.Lambda(lambda x: rearrange(x, "... h w c -> ... c h w")),
        transforms.Lambda(lambda x: x.float().contiguous()),
        transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
    ])
    inverse_video_transform = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1.0) / 2.0 * 255.0),
        transforms.Lambda(lambda x: x.round().clamp(0, 255).to(torch.uint8)),
        transforms.Lambda(lambda x: rearrange(x, "... c h w -> ... h w c")),
    ])
    flow_in_transforms = transforms.Compose([
        transforms.Lambda(lambda x: (x + 1.0) / 2.0 * 255.0),
        transforms.Resize(flow_size),
    ])

    def __init__(
        self,
        args: DataArgs,
        split: str,
        device: torch.device | None = torch.device("cuda"),
        trainer: Optional["Trainer"] = None,
        cache: Literal[
            "r",  # read cache if exists, return None if not exists
            "rw",  # read cache if exists, write cache if not exists
            "w",  # return None if cache exists, write cache if not exists
            "f",  # skip cache, force recalculate
            "fw",  # force recalculate and write cache
        ] = "rw",
        keys=None,
        warp_flow: Literal["flow", "rot_flow", "trans_flow", "trans_rot_flow"] = "flow",
    ) -> None:
        super().__init__()
        self.args = args
        self.split = split
        self.load_split()
        self.stride = (args.stride_min, args.stride_max if split == "train" else args.test_stride_max)
        self.trainer = trainer
        self.device = device
        self.encode_video = trainer.encode_video if trainer else None
        self.encode_text = trainer.encode_text if trainer else None
        self.resize_flow = 2**2
        self.latent_downsample = 8

        # filter chosen clips
        if args.clips is not None:
            self.dataset = [
                d for d in self.dataset
                if f"{d['video_id']}/{d['clip_name']}" in args.clips
            ]

        # go-with-the-flow
        self.downscale_factor = self.resize_flow * self.latent_downsample
        self.warp_flow = warp_flow

        # pose
        if args.derotation in ("yes", "random"):
            assert self.warp_flow == "flow"

        # logger
        try:
            logger.debug("Initializing Dataset")
        except RuntimeError:
            from accelerate.state import PartialState
            PartialState()

        # scan invalid clips and sample
        invalid_clips_path = args.data_root / "cache" / f"invalid_{split}.json"
        if invalid_clips_path.exists():
            with open(invalid_clips_path, "r") as f:
                invalid_clips = json.load(f)
        else:
            logger.info("Scanning invalid clips...")
            self.cache = "w"
            self.keys = ["_sample_meta"]
            invalid_clips = []
            for i in tqdm(range(len(self)), desc="Scan invalid clips"):
                try:
                    self[i]
                except Exception as e:
                    logger.warning(f"Invalid clip {i}: {e}")
                    invalid_clips.append(i)
            with open(invalid_clips_path, "w") as f:
                json.dump(invalid_clips, f, indent=4)
            logger.info(f"Saved invalid clips to {invalid_clips_path}")
        self.dataset = [clip for i, clip in enumerate(self.dataset) if i not in invalid_clips]

        self.cache = cache
        if keys is None:
            self.keys = ["derotation", "prompt_embedding", "image", "video", "encoded_video", "flow", "noise", "pose", "rot_flow"]
        else:
            self.keys = keys

        self.init_demo()

    @abstractmethod
    def load_split(self):
        ...

    def init_demo(self):
        if self.args.demo_name is None:
            return

        self.demo_name = f"{self.args.data_root.name}-{self.args.demo_name}"
        with open(Path("demo") / f"{self.demo_name}.yaml", "r") as f:
            self.demo_config = yaml.safe_load(f)

        self.data_index = {
            f"{d['video_id']}/{d['clip_name']}": i
            for i, d in enumerate(self.dataset)
        }

        self.demo_list = []
        self.src_keys = self.keys.copy()
        self.src_keys.remove("image")
        self.tgt_keys = ["image"]
        for demo in self.demo_config:
            motion_from = demo["motion_from"]
            motion_to = demo.get("motion_to", None)
            if isinstance(motion_from, str) and isinstance(motion_to, list):
                motion_from_name = motion_from.replace("/", "-")
                for tgt in motion_to:
                    self.demo_list.append({
                        "motion_from": self.data_index[motion_from],
                        "motion_to": self.data_index[tgt],
                        "demo_folder": f"motion_from-{motion_from_name}",
                        "output_name": f"motion_to-{tgt}".replace("/", "-"),
                    })
            elif isinstance(motion_from, list) and isinstance(motion_to, str):
                motion_to_name = motion_to.replace("/", "-")
                for src in motion_from:
                    self.demo_list.append({
                        "motion_from": self.data_index[src],
                        "motion_to": self.data_index[motion_to],
                        "demo_folder": f"motion_to-{motion_to_name}",
                        "output_name": f"motion_from-{src}".replace("/", "-"),
                    })
            elif isinstance(motion_from, str) and motion_to is None:
                motion_from_name = motion_from.replace("/", "-")
                motion_to = Path("demo") / self.demo_name / motion_from
                motion_to = motion_to.glob("*.png")
                for img_path in motion_to:
                    self.demo_list.append({
                        "motion_from": self.data_index[motion_from],
                        "motion_to": img_path,
                        "demo_folder": f"motion_from-{motion_from_name}",
                        "output_name": img_path.stem,
                    })

    def __len__(self) -> int:
        return len(self.demo_list) if hasattr(self, "demo_list") else len(self.dataset)

    @cached_property
    def flow_estimater(self):
        from thirdparty.PanoFlowAPI.apis.PanoRaft import PanoRAFTAPI
        flow_estimater = PanoRAFTAPI(
            device=self.device, model_path=self.args.flow_estimater_ckpt
        )
        return flow_estimater

    @classmethod
    def save_image(cls, image: torch.Tensor | np.ndarray, path: str | Path) -> None:
        if isinstance(image, torch.Tensor):
            image = cls.inverse_video_transform(image).cpu().numpy()
        image = Image.fromarray(image)
        image.save(path)

    @classmethod
    def save_video(cls, video: torch.Tensor | np.ndarray, path: str | Path, fps: int = 16) -> None:
        if isinstance(video, torch.Tensor):
            video = cls.inverse_video_transform(video).cpu().numpy()
        video_save = [Image.fromarray(frame) for frame in video]
        export_to_video(video_save, path, fps=fps)

    @classmethod
    def visualize_flow(
        cls,
        flow: torch.Tensor,
        path: str | Path | None = None,
        fps: int = 16,
        method: Literal["arrows", "colormap"] = "colormap",
        max_flow=25,
        canvas: torch.Tensor = (0, 0, 0),
        invert: bool = False,
        bgr: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        flow = rearrange(flow, "n c h w -> n h w c")
        flow = flow.cpu().numpy()
        if method == "arrows":
            if canvas is not None and isinstance(canvas, torch.Tensor):
                canvas = cls.inverse_video_transform(canvas).cpu().numpy()
            flow = flow_to_arrows(flow, canvas=canvas, **kwargs)
        elif method == "colormap":
            flow = better_flow_to_image(flow, max_flow=max_flow, **kwargs)
            if invert:
                flow = 255 - flow
            if bgr:
                flow = flow[..., ::-1]  # Convert RGB to BGR
        if path is not None:
            flow_save = [Image.fromarray(frame) for frame in flow]
            export_to_video(flow_save, path, fps=fps)
        return flow

    @classmethod
    def visualize_noise(
        cls,
        noise: torch.Tensor,
        path: str | Path | None = None,
        fps: int = 4,
        scale: float = 5.,
        resize: int = 8,
    ):
        noise = noise[1:, :3]
        noise = rearrange(noise, "n c h w -> n h w c")
        noise = noise / scale + 0.5
        noise = noise.cpu().numpy()
        noise = rp.resize_images(noise, size=resize, interp='nearest')
        noise = (noise * 255).round().clip(0, 255).astype(np.uint8)
        if path is not None:
            noise_save = [Image.fromarray(frame) for frame in noise]
            export_to_video(noise_save, path, fps=fps)
        return noise

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if not hasattr(self, "demo_list"):
            return self.get_data(index)

        demo_info = self.demo_list[index]
        src_data = self.get_data(demo_info["motion_from"], self.src_keys)
        motion_to = demo_info["motion_to"]

        if isinstance(motion_to, int):
            tgt_data = self.get_data(motion_to, self.tgt_keys)
            data = tgt_data | src_data
            data["caption"] = tgt_data["caption"]
            data["clip_name"] = tgt_data["video_id"]  # for compatibility with test_dataset
        elif isinstance(motion_to, Path):
            data = src_data.copy()
            image = Image.open(motion_to).convert("RGB")
            image = torch.from_numpy(np.array(image))
            image = self.video_transform(image)
            data["image"] = image

            with open(motion_to.with_suffix(".txt"), "r") as f:
                caption = f.read().strip()
            data["caption"] = self.args.caption_prefix + caption
            data["clip_name"] = motion_to.stem  # for compatibility with test_dataset
        else:
            raise ValueError(f"Unsupported motion_to type: {type(motion_to)}")

        data["demo_name"] = self.demo_name
        data["demo_folder"] = demo_info["demo_folder"]
        data["output_name"] = demo_info["output_name"]
        data["motion_from"] = "-".join((src_data["video_id"], src_data["clip_name"]))

        return data

    def get_data(
        self,
        index: int,
        keys: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if keys is None:
            keys = self.keys

        try:
            logger.debug(f"Loading clip {index}")
        except RuntimeError:
            from accelerate.state import PartialState
            PartialState()

        clip = self.dataset[index]
        data = {
            "video_id": clip["video_id"],
            "clip_id": clip["clip_id"],
            "clip_name": clip["clip_name"],
            "caption": self.args.caption_prefix + clip.get("caption", ""),
        }
        seed = self.args.seed + index if self.args.seed is not None else None
        dataset = self

        return_if_not_exists = self.cache == "r"
        return_if_exists = self.cache == "w"
        read_if_exists = self.cache in ("r", "rw")
        calc_if_not_exists = self.cache in ("rw", "fw", "w")
        force_calc = self.cache in ("f", "fw")
        write_if_not_exists = self.cache in ("rw", "w")
        force_write = self.cache == "fw"

        class Cache:
            @cache
            def __new__(
                cls,
                *args: Any,
                **kwargs: Any,
            ) -> Any:
                inst = super().__new__(cls)
                return inst(*args, **kwargs)

            @cached_property
            @abstractmethod
            def _cache_path(self) -> Path:
                ...

            @abstractmethod
            def _read(self) -> Any:
                ...

            @abstractmethod
            def _calc(self) -> Any:
                ...

            @abstractmethod
            def _write(self, value):
                ...

            def _post(self, value) -> Any:
                return value

            @cached_property
            def name(self) -> str:
                return type(self).__name__

            def __call__(
                self,
                force_read: bool = True,
                cache: bool = False,
                post: bool = False,
            ) -> Any:
                exist = self._cache_path.exists()

                if not force_read and not cache and (
                    (return_if_exists and exist) or (return_if_not_exists and not exist)
                ):
                    return

                fix_cache = False
                if not cache and (read_if_exists or force_read) and exist:
                    try:
                        value = self._read()
                        logger.debug(f"Loaded {self.name} from {self._cache_path}", main_process_only=False)
                    except Exception as e:
                        fix_cache = True
                        logger.warning(f"Failed to read {self.name} from {self._cache_path}: {e}")

                if fix_cache or ((calc_if_not_exists or cache) and not exist) or force_calc:
                    value = self._calc()

                if fix_cache or ((write_if_not_exists or cache) and not exist) or force_write:
                    self._write(value)
                    logger.info(f"Saved {self.name} to {self._cache_path}", main_process_only=False)

                if cache:
                    return self._cache_path
                if post:
                    return self._post(value)
                return value

        @cache
        def cache_dir():
            cache_dir = self.args.data_root / "cache" / clip["video_id"] / clip["clip_name"]
            cache_dir.mkdir(parents=True, exist_ok=True)
            return cache_dir

        class _sample_meta(Cache):
            @cached_property
            def _cache_path(self):
                stride = f"{dataset.stride[0]}_{dataset.stride[1]}"
                return cache_dir() / f"sample_meta-{dataset.args.frames}-{stride}.json"

            def _read(self):
                with open(self._cache_path, "r") as f:
                    meta = json.load(f)
                return meta

            def _calc(self):
                return sample_clip(
                    clip,
                    random.Random(seed),
                    dataset.args.frames,
                    dataset.stride,
                )

            def _write(self, value):
                with open(self._cache_path, "w") as f:
                    json.dump(value, f, indent=4)

        class prompt_embedding(Cache):
            @cached_property
            def _cache_path(self):
                return cache_dir() / "prompt_embeddings.safetensors"

            def _read(self):
                return load_file(self._cache_path)["prompt_embedding"]

            def _calc(self):
                prompt_embedding = dataset.encode_text(data["caption"])
                prompt_embedding = prompt_embedding.to("cpu")
                # [1, seq_len, hidden_size] -> [seq_len, hidden_size]
                return prompt_embedding[0]

            def _write(self, value):
                save_file({"prompt_embedding": value}, self._cache_path)

        @cache
        def _video_cache_suffix():
            resolution = "x".join((str(self.args.frames), str(self.args.height), str(self.args.width)))
            frames = _sample_meta()["frames"]
            frame_range = f"{frames[0]}_{frames[1]}"
            stride = _sample_meta()["stride"]
            return f"{resolution}-{stride}-{frame_range}"

        @cache
        def _random_rotate():
            if self.split == "train" and self.args.random_rotate:
                return int(random.random() * self.args.width)
            return 0

        @cache
        def derotation():
            if dataset.args.derotation == "random":
                return torch.tensor(index % 2 == 0)
            return torch.tensor(dataset.args.derotation == "yes")

        class video(Cache):
            @cached_property
            def _cache_path(self):
                path = cache_dir() / f"video-{_video_cache_suffix()}"
                if derotation():
                    path = path.with_name(path.stem + "-derotation")
                path = path.with_suffix(".mp4")
                return path

            def _read(self):
                with decord.bridge.use_torch():
                    video = decord.VideoReader(str(self._cache_path), num_threads=1)
                    video = video.get_batch(range(len(video)))
                video = dataset.video_transform(video)
                return video

            def _calc(self):
                if derotation():
                    height, width = -1, -1
                else:
                    height, width = dataset.args.height, dataset.args.width
                video = preprocess_video_with_sampling(
                    video_path = clip["video_path"],
                    frame_range = _sample_meta()["frames"],
                    frames = dataset.args.frames,
                    height = height,
                    width = width,
                )
                video = dataset.video_transform(video)
                if derotation():
                    rotation = equilib_rotation(derotation_R())
                    video = equi2equi(
                        video,
                        rotation,
                        height=dataset.args.height,
                        width=dataset.args.width,
                    )
                return video

            def _write(self, value):
                dataset.save_video(value, self._cache_path)

            def _post(self, value):
                if dataset.args.first_n_frames is not None:
                    value = value[:dataset.args.first_n_frames]
                return value

        class image(Cache):
            @cached_property
            def _cache_path(self):
                return cache_dir() / f"image-{_video_cache_suffix()}.png"

            def _read(self):
                image = Image.open(self._cache_path).convert("RGB")
                image = torch.from_numpy(np.array(image))
                image = dataset.video_transform(image)
                return image

            def _calc(self):
                return video()[0]

            def _write(self, value):
                dataset.save_image(value, self._cache_path)

            def _post(self, value):
                image = torch.roll(value, shifts=_random_rotate() * 8, dims=-1)
                return image

        class encoded_video(Cache):
            @cached_property
            def _cache_path(self):
                assert dataset.trainer is not None, "trainer must be set to use encoded_video"
                encoded_video_path = cache_dir() / f"video_latent-{dataset.trainer.args.model_name}-{_video_cache_suffix()}"
                if derotation():
                    encoded_video_path = encoded_video_path.with_name(encoded_video_path.stem + "-derotation")
                if dataset.args.circular_padding > 0:
                    encoded_video_path = encoded_video_path.with_name(encoded_video_path.stem + f"-circpad_{dataset.args.circular_padding}")
                encoded_video_path = encoded_video_path.with_suffix(".safetensors")
                return encoded_video_path

            def _read(self):
                return load_file(self._cache_path)["encoded_video"]

            def _calc(self):
                assert dataset.args.first_n_frames is None, "encoded_video does not support first_n_frames"

                # Current shape of frames: [F, C, H, W]
                frames = video().to(dataset.device)

                # Convert to [B, C, F, H, W]
                frames = frames.unsqueeze(0)
                frames = frames.permute(0, 2, 1, 3, 4).contiguous()
                encoded_video = dataset.encode_video(frames)

                # [1, C, F, H, W] -> [C, F, H, W]
                encoded_video = encoded_video[0]
                encoded_video = encoded_video.to("cpu")
                return encoded_video

            def _write(self, value):
                save_file({"encoded_video": value}, self._cache_path)

            def _post(self, value):
                latent = torch.roll(value, shifts=_random_rotate(), dims=-1)
                return latent

        @cache
        def flow():
            flow_in = self.flow_in_transforms(video()).to(self.device)
            flows = self.flow_estimater.chunk_estimate_flow_cfe(flow_in)
            flows = rearrange(flows, "n h w c -> n c h w")
            flows = F.interpolate(
                flows, size=(self.args.height, self.args.width), mode="bilinear", align_corners=False
            )
            flows[:, 0] = flows[:, 0] * (self.args.width / self.flow_size[1])
            flows[:, 1] = flows[:, 1] * (self.args.height / self.flow_size[0])
            return flows

        class noise(Cache):
            @cached_property
            def _cache_path(self):
                noise_path = cache_dir() / f"noise-{_video_cache_suffix()}"
                if derotation():
                    noise_path = noise_path.with_name(noise_path.stem + "-derotation")
                if dataset.args.loop_x_noise_warp:
                    noise_path = noise_path.with_name(noise_path.stem + "-loop_x")
                if dataset.args.loop_y_noise_warp:
                    noise_path = noise_path.with_name(noise_path.stem + "-loop_y")
                if dataset.warp_flow != "flow":
                    noise_path = noise_path.with_name(noise_path.stem + f"-{dataset.warp_flow}")
                noise_path = noise_path.with_suffix(".safetensors")
                return noise_path

            def _read(self):
                return load_file(self._cache_path)["noise"]

            def _calc(self):
                assert dataset.args.first_n_frames is None, "noise does not support first_n_frames"

                warper = nw.NoiseWarper(
                    c=16,
                    h=dataset.resize_flow * dataset.args.height,
                    w=dataset.resize_flow * dataset.args.width,
                    device=dataset.device,
                    warp_kwargs=dict(
                        loop_x=dataset.args.loop_x_noise_warp,
                        loop_y=dataset.args.loop_y_noise_warp,
                    ),
                )

                def downscale_noise(noise):
                    down_noise = rp.torch_resize_image(noise, 1 / dataset.downscale_factor, interp='area')
                    down_noise = down_noise * dataset.downscale_factor
                    return down_noise

                noise = warper.noise
                noises = [downscale_noise(noise).cpu()]
                flow_fn = {fn.__name__: fn for fn in (
                    flow, rot_flow, trans_flow, trans_rot_flow
                )}
                for f in flow_fn[dataset.warp_flow]():
                    noise = warper(f[0], f[1]).noise
                    noises.append(downscale_noise(noise).cpu())
                noises = torch.stack(noises, dim=0)
                noises = get_downtemp_noise(noises)
                noises = noises.to(torch.float16)

                return noises

            def _write(self, value):
                save_file({"noise": value}, str(self._cache_path))

            def _post(self, value):
                noise = torch.roll(value, shifts=_random_rotate(), dims=-1)
                return noise

        @cache
        def _pose():
            pose_path = (self.args.data_root / "slam_pose" / clip["video_id"] / clip["clip_name"]).with_suffix(".npy")
            clip_start = clip["frames"][0]
            sample_frames = _sample_meta()["frames"]
            sample_frames = (sample_frames[0] - clip_start, sample_frames[1] - clip_start)
            pose = preprocess_pose_with_sampling(
                pose_path = pose_path,
                frame_range = sample_frames,
                frames = self.args.frames,
            )
            return pose

        @cache
        def pose(derotation=None, scale_normalization=None):
            if derotation is None:
                derotation = self.args.derotation
            if scale_normalization is None:
                scale_normalization = self.args.scale_normalization

            pose = _pose()
            pose = pose.clone()

            if derotation:
                pose[:, :3, :3] = torch.eye(3)
            if scale_normalization:
                pose[:, :3, 3] /= _average_depth()
            return pose

        def derotation_R():
            rotation = _pose()[:, :3, :3]
            return rotation[0] @ rotation.transpose(-1, -2)

        @cache
        def rot_flow():
            M = _pose().clone()
            M[:, :3, 3] = 0
            M = M[1:].inverse() @ M[:-1]
            return transformation_to_flow(M, (self.args.height, self.args.width))

        @cache
        def absolute_rot_flow():
            M = _pose().clone()
            M[:, :3, 3] = 0
            return transformation_to_flow(M, (self.args.height, self.args.width))

        @cache
        def derotated_flow():
            return flow() - rot_flow()

        @cache
        def trans_flow(scale_normalization=True):
            M = pose(derotation=True, scale_normalization=scale_normalization)
            M = M[1:].inverse() @ M[:-1]
            return transformation_to_flow(M, (self.args.height, self.args.width))

        @cache
        def trans_rot_flow():
            M = pose(derotation=False, scale_normalization=True)
            M = M[1:].inverse() @ M[:-1]
            return transformation_to_flow(M, (self.args.height, self.args.width))

        @cache
        def _flow_depth():
            df = derotated_flow()
            tf = trans_flow(scale_normalization=False)
            dot = einsum(df, tf, "n c h w, n c h w -> n h w")
            tf_norm = tf.norm(dim=1)
            depth = tf_norm ** 2 / dot.clamp_min(1e-6)
            df_norm = df.norm(dim=1)
            cos = dot / (tf_norm * df_norm).clamp_min(1e-6)
            cos = cos.clamp(-1, 1)
            degree = torch.rad2deg(torch.acos(cos))
            depth[degree > 45] = float("nan")
            return depth

        class _average_depth(Cache):
            @cached_property
            def _cache_path(self):
                return cache_dir() / f"average_depth-{_video_cache_suffix()}.txt"

            def _read(self):
                with open(self._cache_path, "r") as f:
                    average_depth = f.read()
                return float(average_depth)

            def _calc(self):
                depth = _flow_depth()
                depth = rearrange(depth, "n h w -> n (h w)")
                middle_plane = torch.nanquantile(depth, 0.5, dim=-1)
                middle_plane = torch.nanquantile(middle_plane, 0.5)
                average_depth = 1. if middle_plane.isnan() else middle_plane.item()
                return average_depth

            def _write(self, value):
                with open(self._cache_path, "w") as f:
                    f.write(str(value))

        for key in keys:
            kwargs = {}
            fcn = key

            if key.endswith("_cache"):
                kwargs["cache"] = True
                fcn = fcn.removesuffix("_cache")

            if inspect.isclass(locals()[fcn]) and issubclass(locals()[fcn], Cache):
                kwargs["force_read"] = False
                kwargs["post"] = True

            data[key] = eval(fcn)(**kwargs)

        return data


class TestDataArgs(DataArgs):
    num_test_samples: Optional[int] = 3


def test_dataset(
    cls: type[BaseDataset],
    args: TestDataArgs,
    keys = [
        "video", "image", "flow", "noise", "pose", "derotation_R",
        "rot_flow", "derotated_flow", "trans_flow", "trans_rot_flow",
        "_average_depth"
    ],
):
    dataset = cls(
        args,
        split="test",
        keys=keys,
        cache="rw",
        # warp_flow="rot_flow",
    )

    average_depths = []
    dataset = torch.utils.data.Subset(
        dataset,
        range(min(args.num_test_samples, len(dataset))) if args.num_test_samples is not None else len(dataset)
    )
    for data in tqdm(dataset):
        output_path = Path("debug") / "panflow_dataset" / f"{data['video_id']}-{data['clip_name']}"
        output_path.mkdir(parents=True, exist_ok=True)

        if "pose" in data:
            c2w = data.pop("pose")

        if "video" in data:
            video = data.pop("video")
            if "derotation_R" in data:
                derotation_R = data.pop("derotation_R")
                rotation = equilib_rotation(derotation_R.transpose(-1, -2))
                video = equi2equi(video, rotation)
            clip_path = output_path / "video.mp4"
            cls.save_video(video, clip_path)
            tqdm.write(f"Saved video to {clip_path}")

        if "image" in data:
            image = data.pop("image")
            image_path = output_path / "image.png"
            cls.save_image(image, image_path)
            tqdm.write(f"Saved image to {image_path}")

        for key in keys:
            if "flow" not in key:
                continue
            flow = data.pop(key)
            clip_path = output_path / f"{key}.mp4"
            cls.visualize_flow(
                flow=flow,
                path=clip_path,
                method="arrows",
                canvas=video[:-1],
            )
            tqdm.write(f"Saved flow to {clip_path}")

        if "noise" in data:
            noise = data.pop("noise")
            clip_path = output_path / "noise.mp4"
            cls.visualize_noise(
                noise=noise,
                path=clip_path,
            )
            tqdm.write(f"Saved noise to {clip_path}")

        if "_average_depth" in data:
            average_depth = data.pop("_average_depth")
            average_depths.append(average_depth)

        meta_path = output_path / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(data, f, indent=4)
        tqdm.write(f"Saved meta to {meta_path}")

    print("Average depth:", average_depths)
