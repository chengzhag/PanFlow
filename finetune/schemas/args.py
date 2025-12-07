import logging
from pathlib import Path
from typing import Any, List, Literal, Tuple, Optional
import tyro

from pydantic import BaseModel, ValidationInfo, field_validator


class TyroArgs(BaseModel):
    @classmethod
    def parse_args(cls):
        args = tyro.cli(cls)
        return args


class BaseArgs(TyroArgs):
    ########## Output ##########
    output_dir: Path = Path("logs")
    exp_dir: Optional[Path] = None

    ########## Model ##########
    model_type: Literal["i2v", "t2v"] = "i2v"
    training_type: Literal["lora", "sft"] = "lora"
    model_path: Path = Path("THUDM/CogVideoX-5B-I2V")
    model_name: str = "cogvideox-i2v"
    fuse_lora_checkpoint: Path | None = Path("checkpoints/I2V5B_final_i38800_nearest_lora_weights.safetensors")
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    skip_model_load: bool = False

    ######## Inference ##########
    enable_slicing: bool = True
    enable_tiling: bool = True
    num_inference_steps: int = 30
    guidance_scale: float = 6.
    noise_alpha: float = 0.0
    fps: int = 16
    seed: int | None = 42
    latent_rotation: bool = True
    latent_rotation_degree: float = 40.0

    ########## Lora ##########
    rank: int = 128
    lora_alpha: int = 64
    target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]

    @field_validator("mixed_precision")
    def validate_mixed_precision(cls, v: str, info: ValidationInfo) -> str:
        if v == "fp16" and "cogvideox-2b" not in str(info.data.get("model_path", "")).lower():
            logging.warning(
                "All CogVideoX models except cogvideox-2b were trained with bfloat16. "
                "Using fp16 precision may lead to training instability."
            )
        return v


class DataArgs(TyroArgs):
    train_dataset: Literal["PanFlowDataset", "Web360Dataset"] = "PanFlowDataset"
    test_dataset: Literal["PanFlowDataset", "Web360Dataset"] = "PanFlowDataset"
    data_root: Path = Path("data/PanFlow")
    caption_prefix: str = "A 360 panorama video, "
    seed: int | None = 42
    num_workers: int = 4
    clips: Optional[list[str]] = None

    # video parameters
    stride_min: int = 1
    stride_max: int = 3
    test_stride_max: int = 1
    frames: int = 49
    height: int = 480
    width: int = 720
    first_n_frames: Optional[int] = None
    num_test_samples: Optional[int] = None
    test_start_index: int = 0

    # optical flow
    flow_estimater_ckpt: Optional[Path] = Path("checkpoints/PanoFlow(RAFT)-wo-CFE.pth")

    # go-with-the-flow
    loop_x_noise_warp: bool = True
    loop_y_noise_warp: bool = True

    # pose
    derotation: Literal["yes", "no", "random"] = "no"
    scale_normalization: bool = True

    # circular_padding
    circular_padding: int = 1

    # augmentation
    random_rotate: bool = True

    # demo
    demo_name: Optional[str] = None

    @field_validator("frames")
    def validate_frames(cls, v: int) -> int:
        if (v - 1) % 8 != 0:
            raise ValueError("Number of frames - 1 must be a multiple of 8")
        return v

    @field_validator("height")
    def validate_height(cls, v: int, info: ValidationInfo) -> int:
        model_name = info.data.get("model_name", "")
        if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"] and v != 480:
            raise ValueError("For cogvideox-5b models, height must be 480")
        return v

    @field_validator("width")
    def validate_width(cls, v: int, info: ValidationInfo) -> int:
        model_name = info.data.get("model_name", "")
        if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"] and v != 720:
            raise ValueError("For cogvideox-5b models, width must be 720")
        return v


class TestArgs(BaseModel):
    num_videos_per_prompt: int = 1
    test_suffix: Optional[str] = None
    rotate_180: bool = False


class Args(DataArgs, TestArgs, BaseArgs):
    ########## Output ##########
    report_to: Literal["tensorboard", "wandb", "all"] | None = "wandb"
    tracker_name: str = "panflow"
    wandb_entity: str = "panflow_team"
    backup_dir: Path | None = None
    upload_ckpt_to_wandb: bool = True

    ########## Data ###########
    num_train_samples: int | None = None
    num_validation_samples_per_gpu: int | None = 1

    ########## Training #########
    resume_from_checkpoint: Path | None = None
    finetune_from_checkpoint: Path | None = None

    train_epochs: int = 20
    train_steps: int | None = None
    checkpointing_steps: int = 500
    checkpointing_limit: int = 2

    batch_size: int = 1
    gradient_accumulation_steps: int = 1

    #### deprecated args: video_resolution_buckets
    # if use bucket for training, should not be None
    # Note1: At least one frame rate in the bucket must be less than or equal to the frame rate of any video in the dataset
    # Note2:  For cogvideox, cogvideox1.5
    #   The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #   The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # video_resolution_buckets: List[Tuple[int, int, int]] | None = None

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    pin_memory: bool = True

    gradient_checkpointing: bool = True
    nccl_timeout: int = 1800

    ########## Validation ##########
    do_validation: bool = True
    validation_steps: int | None = 500  # if set, should be a multiple of checkpointing_steps

    #### deprecated args: gen_video_resolution
    # 1. If set do_validation, should not be None
    # 2. Suggest selecting the bucket from `video_resolution_buckets` that is closest to the resolution you have chosen for fine-tuning
    #        or the resolution recommended by the model
    # 3. Note:  For cogvideox, cogvideox1.5
    #        The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #        The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # gen_video_resolution: Tuple[int, int, int] | None  # shape: (frames, height, width)

    @field_validator("validation_steps")
    def validate_validation_steps(cls, v: int | None, info: ValidationInfo) -> int | None:
        values = info.data
        if values.get("do_validation"):
            if v is None:
                raise ValueError("validation_steps must be specified when do_validation is True")
            if values.get("checkpointing_steps") and v % values["checkpointing_steps"] != 0:
                raise ValueError("validation_steps must be a multiple of checkpointing_steps")
        return v
