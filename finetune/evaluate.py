import sys
from pathlib import Path
from collections import defaultdict
import importlib
from tqdm.auto import tqdm
import os
import torch
import torch.multiprocessing as mp
from torch import Tensor
from functools import cache
import numpy as np
import json
import decord
from diffusers.utils.export_utils import export_to_video
from PIL import Image
from typing import Any, List, Literal, Tuple, Optional
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from finetune.schemas import Args
from finetune.models.utils import get_model_cls
from finetune.datasets import *
from thirdparty.fvd.fvd import (
    load_i3d_pretrained,
    get_fvd_logits,
)

from torchmetrics import MeanSquaredError
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.fid import FrechetInceptionDistance, _compute_fid
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal import CLIPScore
from torchmetrics import MeanMetric, Metric


class EvaluateArgs(Args):
    test_steps: List[str] = ["inference", "image", "overall", "visualize"]
    test_device: Literal["cuda", "cpu"] = "cuda"
    test_res_path: Optional[Path] = None
    target_model: Literal["PanFlow", "360DVD", "DynamicScaler", "HoloTime", "MotionClone", "GoWithTheFlow"] = "PanFlow"
    evaluate_gt: bool = False
    test_name: str = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    first_n_frames: Optional[int] = None

    # image
    test_chunk_size: int = 8


def get_path(args):
    if args.evaluate_gt:
        test_res_path = None
        test_dir = args.data_root / "evaluate"
    else:
        if args.test_res_path is not None:
            test_res_path = args.test_res_path
            test_dir = test_res_path.parent
        elif args.exp_dir is not None:
            exp_name = args.data_root.name
            if args.test_suffix is not None:
                exp_name += f"-{args.test_suffix}"
            test_dir = Path(args.exp_dir) / exp_name
            test_res_path = test_dir / "test_res"
        else:
            run_id = os.environ.get('WANDB_RUN_ID', None)
            assert run_id is not None, "WANDB_RUN_ID environment variable must be set."
            exp_name = args.data_root.name
            if args.test_suffix is not None:
                exp_name += f"-{args.test_suffix}"
            test_dir = Path(args.output_dir) / run_id / exp_name
            test_res_path = test_dir/ "test_res"
    print(f"Experiment directory: {test_dir}")
    print(f"Test results path: {test_res_path}")
    print(f"Evaluation name: {args.test_name}")
    return test_dir, test_res_path


@cache
def scan_results(test_res_path, target_model, check_length=None):
    print(f"Scanning test results in {test_res_path}")

    if target_model == "DynamicScaler":
        video_list = list(test_res_path.glob("*/shift_windows.mp4"))
        folder_list = defaultdict(list)
        for video in video_list:
            video_name = video.parent.name

            # 0712_08-40-55-100071_Clip-001_s2333333
            # 0709_23-57-45-tmaQLgUMlis_Clip-009_s2333333
            # remove the timestamp and suffix
            video_name = "-".join(video_name.split("-")[3:])
            video_name = video_name.removesuffix("_s2333333")
            video_name = video_name.split("_Clip-")
            video_id = video_name[0]
            clip_id = video_name[1]

            folder = Path(video_id) / f"Clip-{clip_id}"
            folder_list[str(folder)].append(video)
    elif target_model == "HoloTime":
        video_list = list(test_res_path.glob("*/*.mp4"))
        folder_list = defaultdict(list)
        for video in video_list:
            video_name = video.parent.name

            # scene_tmaQLgUMlis_Clip-009
            # scene_100001_Clip-001
            # remove the scene prefix
            video_name = video_name.removeprefix("scene_")
            video_name = video_name.split("_Clip-")
            video_id = video_name[0]
            clip_id = video_name[1]

            folder = Path(video_id) / f"Clip-{clip_id}"
            folder_list[str(folder)].append(video)
    elif target_model == "MotionClone":
        video_list = list(test_res_path.glob("*.mp4"))
        folder_list = defaultdict(list)
        for video in video_list:
            video_name = video.stem

            # tmaQLgUMlis_Clip-009
            # 100001_Clip-001
            video_name = video_name.split("_Clip-")
            video_id = video_name[0]
            clip_id = video_name[1]

            folder = Path(video_id) / f"Clip-{clip_id}"
            folder_list[str(folder)].append(video)
    else:
        video_list = list(test_res_path.glob("*/*/*.mp4"))
        folder_list = defaultdict(list)
        for video in video_list:
            folder = video.parent.relative_to(test_res_path)
            folder_list[str(folder)].append(video)

    print(f"Found {len(folder_list)} folders with {len(video_list)} videos")
    if check_length is not None:
        assert len(folder_list) >= check_length, \
            f"Number of generated videos ({len(folder_list)}) is less than the number of samples in the dataset ({check_length})."

    return folder_list


def inference(args):
    print("Running inference...")
    first_n_frames = args.first_n_frames
    args.first_n_frames = None  # Remove this to avoid issues with the model
    trainer_cls = get_model_cls(args.model_name, args.training_type)
    trainer = trainer_cls(args)
    trainer.test()
    args.first_n_frames = first_n_frames  # Restore the original value


def collate_fn(samples):
    data = samples[0]
    return data


def prepare_dataloader(
    args,
    keys,
    cache="rw",
):
    mp.set_start_method('spawn', force=True)
    dataset = globals()[args.test_dataset](
        args,
        split="test",
        keys=keys,
        cache=cache,
    )
    if args.num_test_samples is not None:
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.num_test_samples, len(dataset)))
        )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collate_fn,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
    )
    return dataloader


class EndContinuity(MeanMetric):
    higher_is_better = False

    def update(self, preds):
        error = torch.abs(preds[..., -1] - preds[..., 0]).mean()
        super().update(error, preds.shape[0])


class EndPointError(MeanMetric):
    higher_is_better = False

    def __init__(
        self,
        flow_estimater_ckpt: Optional[str] = None,
        chunk_size: int = 1,
    ):
        super().__init__()
        from thirdparty.PanoFlowAPI.apis.PanoRaft import PanoRAFTAPI
        self.flow_estimater = PanoRAFTAPI(model_path=flow_estimater_ckpt)
        self.chunk_size = chunk_size

    def update(self, preds, targets):
        flows = []
        for frames in (preds, targets):
            flow_in = BaseDataset.flow_in_transforms(frames * 2. - 1.)
            flow = self.flow_estimater.chunk_estimate_flow_cfe(flow_in, self.chunk_size)
            flows.append(flow)
        error = ((flows[0] - flows[1])**2).sum(dim=-1).sqrt().mean()
        super().update(error, preds.shape[0])


class FrechetVideoDistance(Metric):
    higher_is_better: bool = False
    full_state_update: bool = False

    def __init__(
        self,
        crop_center: bool = True,
        batch_size: int = 10,
    ):
        super().__init__()
        self.crop_center = crop_center
        self.batch_size = batch_size
        self.i3d = load_i3d_pretrained()

        num_features = 400
        mx_num_feats = (num_features, num_features)
        self.add_state("real_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("real_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
        self.add_state("real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

        self.add_state("fake_features_sum", torch.zeros(num_features).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_cov_sum", torch.zeros(mx_num_feats).double(), dist_reduce_fx="sum")
        self.add_state("fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum")

    def update(self, videos: Tensor, real: bool) -> None:
        features = get_fvd_logits(videos, self.i3d, self.device, bs=self.batch_size, crop_center=self.crop_center)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += videos.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += videos.shape[0]

    def compute(self) -> Tensor:
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)


def read_pred_video(video_path, frames, height, width):
    with decord.bridge.use_torch():
        video_reader = decord.VideoReader(
            str(video_path),
            # ctx=decord.cpu(0),
            # num_threads=1,
            height=height,
            width=width,
        )
        frames = len(video_reader) if frames is None else frames
        video = video_reader.get_batch(list(range(frames)))
    return video


def image(args):
    print("Running image evaluation...")
    test_dir, test_res_path = get_path(args)

    dataloader = prepare_dataloader(
        args,
        keys=["video"] + (["video_cache"] if args.evaluate_gt else []),
    )

    folder_list = scan_results(
        test_res_path,
        target_model=args.target_model,
        check_length=len(dataloader.dataset)
    ) if not args.evaluate_gt else None

    image_metrics = {
        "lpips": LearnedPerceptualImagePatchSimilarity(
            net_type="vgg",
            normalize=True,
        ),
        "psnr": PeakSignalNoiseRatio(
            data_range=1.,
            dim=(1, 2, 3)
        ),
        "ssim": StructuralSimilarityIndexMeasure(
            data_range=1.,
        ),
        "mse": MeanSquaredError(),  # Go-with-the-Flow
        "cs_text": CLIPScore(
            model_name_or_path="zer0int/LongCLIP-L-Diffusers",
        ),  # Go-with-the-Flow
        "cs_image": CLIPScore(),  # Go-with-the-Flow
        "end_continuity": EndContinuity(),  # PanoWan
        "flow_epe": EndPointError(
            flow_estimater_ckpt=args.flow_estimater_ckpt,
            chunk_size=args.test_chunk_size,
        ),  # PanoFlow
    }
    image_metrics = {k: v.to(args.test_device) for k, v in image_metrics.items()}
    data_metrics = {
        "fvd_center": FrechetVideoDistance(),
        "fvd": FrechetVideoDistance(
            crop_center=False,
        ),  # Go-with-the-Flow, PanoDiT, PanoWan
        "fid": FrechetInceptionDistance(
            normalize=True,
        ),  # Go-with-the-Flow
        "is": InceptionScore(
            normalize=True,
        ),  # PanoDiT
    }
    data_metrics = {k: v.to(args.test_device) for k, v in data_metrics.items()}

    eval_results = defaultdict(list)
    for data in tqdm(dataloader, desc="Evaluating videos"):
        folder = Path(data["video_id"]) / data["clip_name"]
        if args.evaluate_gt:
            video_list = [str(data["video_cache"])]
        else:
            video_list = folder_list[str(folder)]
        gt_video = data["video"]
        gt_video = gt_video / 2. + 0.5
        gt_video_cuda = gt_video.to(args.test_device)

        for video_path in video_list:
            video = read_pred_video(video_path, args.first_n_frames, args.height, args.width)
            video = video.float().permute(0, 3, 1, 2) / 255.0
            video = video.to(args.test_device)

            for metric_name, metric in image_metrics.items():
                metric_value = 0.
                with torch.no_grad():
                    if metric_name == "cs_text":
                        for pred in video.split(args.test_chunk_size, dim=0):
                            pred = pred * 255.0
                            metric_value += metric(
                                pred,
                                [data["caption"]] * len(pred)
                            ).cpu().item() * len(pred)
                    elif metric_name == "cs_image":
                        for pred, gt in zip(
                            video[:-1].split(args.test_chunk_size, dim=0),
                            video[1:].split(args.test_chunk_size, dim=0),
                        ):
                            pred, gt = pred * 255.0, gt * 255.0
                            metric_value += metric(pred, gt).cpu().item() * len(pred)
                    elif metric_name == "end_continuity":
                        for pred in video.split(args.test_chunk_size, dim=0):
                            metric_value += metric(pred).cpu().item() * len(pred)
                    elif metric_name == "flow_epe":
                        metric_value += metric(video, gt_video_cuda).cpu().item() * len(video)
                    else:
                        for pred, gt in zip(
                            video.split(args.test_chunk_size, dim=0),
                            gt_video_cuda.split(args.test_chunk_size, dim=0),
                        ):
                            metric_value += metric(
                                pred.contiguous(),
                                gt
                            ).cpu().item() * len(pred)
                metric_value /= len(video)
                eval_results[metric_name].append({
                    "video_path": str(video_path),
                    "video_results": metric_value,
                })

            for metric_name, metric in data_metrics.items():
                with torch.no_grad():
                    for pred, gt in zip(
                        video.split(args.test_chunk_size, dim=0),
                        gt_video_cuda.split(args.test_chunk_size, dim=0),
                    ):
                        if metric_name in "fid":
                            metric.update(pred, real=False)
                            metric.update(gt, real=True)
                        elif metric_name == "is":
                            metric.update(pred)
                    if metric_name in ("fvd", "fvd_center"):
                        metric.update(video.unsqueeze(0), real=False)
                        metric.update(gt_video_cuda.unsqueeze(0), real=True)

    for metric_name, metric in data_metrics.items():
        if metric_name in ("fid", "fvd", "fvd_center"):
            eval_results[metric_name] = metric.compute().item()
        elif metric_name == "is":
            eval_results[metric_name], eval_results[f"{metric_name}_std"] = metric.compute()
            eval_results[metric_name] = eval_results[metric_name].cpu().item()
            eval_results[f"{metric_name}_std"] = eval_results[f"{metric_name}_std"].cpu().item()

    for key, values in eval_results.items():
        if isinstance(values, list):
            results = [v["video_results"] for v in values]
            results = float(np.mean(results))
            eval_results[key] = [results, values]
        else:
            eval_results[key] = [values]

    output_path = test_dir / "image_metrics"
    output_path.mkdir(parents=True, exist_ok=True)
    output_path = output_path / f"{args.test_name}_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    print(f"Evaluation results saved to {output_path}")


def overall(args):
    test_dir, test_res_path = get_path(args)
    overall_res = {}
    for key in ["image_metrics"]:
        eval_res_path = test_dir / key / f"{args.test_name}_eval_results.json"
        if not eval_res_path.exists():
            print(f"Evaluation results for {key} not found at {eval_res_path}. Skipping.")
            continue
        with open(eval_res_path, "r") as f:
            eval_res = json.load(f)
        for metric, values in eval_res.items():
            overall_res[f"{key}/{metric}"] = values[0]
    overall_res_path = test_dir / "overall" / f"{args.test_name}.json"
    overall_res_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overall_res_path, "w") as f:
        json.dump(overall_res, f, indent=4)
    print(f"Overall evaluation results saved to {overall_res_path}")
    print(json.dumps(overall_res, indent=4))


def visualize(args):
    print("Running visualization...")
    test_dir, test_res_path = get_path(args)

    dataloader = prepare_dataloader(
        args,
        keys=["video", "video_cache"],
    )

    folder_list = scan_results(
        test_res_path,
        target_model=args.target_model,
        check_length=len(dataloader.dataset)
    )

    for data in tqdm(dataloader, desc="Visualizing videos"):
        folder = Path(data["video_id"]) / data["clip_name"]
        video_list = folder_list[str(folder)]
        gt_video = data["video"]
        gt_video = BaseDataset.inverse_video_transform(gt_video)
        gt_name = data["video_cache"].name

        for video_path in video_list:
            video = read_pred_video(video_path, args.first_n_frames, args.height, args.width)
            vis = np.concatenate([gt_video, video], axis=-2)
            vis = [Image.fromarray(frame) for frame in vis]
            output_path = test_dir / "visualize" / folder / f"{video_path.stem}-{gt_name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(vis, output_path, fps=len(vis) // 2)
            tqdm.write(f"Saved visualization to {output_path}")


def main():
    args = EvaluateArgs.parse_args()

    if args.evaluate_gt:
        print("Running evaluation with ground truth videos.")
        if "inference" in args.test_steps:
            print("Inference step is not applicable for ground truth evaluation. Skipping.")
            args.test_steps.remove("inference")
        if "visualize" in args.test_steps:
            print("Visualization step is not applicable for ground truth evaluation. Skipping.")
            args.test_steps.remove("visualize")
    
    if args.target_model not in ("PanFlow", "GoWithTheFlow"):
        if "inference" in args.test_steps:
            print(f"{args.target_model} does not support inference step. Skipping.")
            args.test_steps.remove("inference")
        args.caption_prefix = ""
    elif args.target_model == "GoWithTheFlow":
        args.output_dir = Path("outputs")
        args.exp_dir = args.output_dir / "GoWithTheFlow"
        args.circular_padding = 0
        args.latent_rotation = False
        args.noise_alpha = 0.5

        args.caption_prefix = ""
        args.loop_x_noise_warp = False
        args.loop_y_noise_warp = False

    for key in args.test_steps:
        globals()[key](args)


if __name__ == "__main__":
    main()
