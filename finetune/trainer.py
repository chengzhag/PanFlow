import hashlib
import json
import logging
import math
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
from abc import abstractmethod

import diffusers
import torch
import transformers
import wandb
from accelerate.accelerator import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
    gather_object,
    set_seed,
    broadcast_object_list,
)
from diffusers.optimization import get_scheduler
from diffusers.pipelines import DiffusionPipeline
from diffusers.utils.export_utils import export_to_video
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.datasets import *
from finetune.schemas import Args, Components, State
from finetune.utils import (
    cast_training_params,
    free_memory,
    get_intermediate_ckpt_path,
    get_latest_ckpt_path_to_resume_from,
    get_latest_ckpt_path,
    get_memory_statistics,
    get_optimizer,
    unload_model,
    unwrap_model,
)
import rp
rp.r._pip_import_autoyes = True  # Automatically install missing packages
rp.git_import("CommonSource")
import rp.git.CommonSource.noise_warp as nw


logger = get_logger(LOG_NAME, LOG_LEVEL)

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,  # FP16 is Only Support for CogVideoX-2B
    "bf16": torch.bfloat16,
}


class Trainer:
    # If set, should be a list of components to unload (refer to `Components``)
    UNLOAD_LIST: List[str] = None

    def __init__(self, args: Args) -> None:
        self.args = args
        self.state = State(
            weight_dtype=self.__get_training_dtype(),
            train_frames=self.args.frames,
            train_height=self.args.height,
            train_width=self.args.width,
        )

        self.components: Components = self.load_components()
        self.accelerator: Accelerator = None
        self.dataset: Dataset = None
        self.data_loader: DataLoader = None

        self.optimizer = None
        self.lr_scheduler = None

        self._init_distributed()
        self._init_logging()
        self._init_directories()

        self.state.using_deepspeed = self.accelerator.state.deepspeed_plugin is not None

    def _init_distributed(self):
        logging_dir = Path(self.args.output_dir, "logs")
        project_config = ProjectConfiguration(
            project_dir=self.args.output_dir, logging_dir=logging_dir
        )
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_process_group_kwargs = InitProcessGroupKwargs(
            backend="nccl", timeout=timedelta(seconds=self.args.nccl_timeout)
        )
        mixed_precision = "no" if torch.backends.mps.is_available() else self.args.mixed_precision
        report_to = None if self.args.report_to.lower() == "none" else self.args.report_to

        accelerator = Accelerator(
            project_config=project_config,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with=report_to,
            kwargs_handlers=[ddp_kwargs, init_process_group_kwargs],
        )

        # Disable AMP for MPS.
        if torch.backends.mps.is_available():
            accelerator.native_amp = False

        self.accelerator = accelerator

        if self.args.seed is not None:
            set_seed(self.args.seed)

    def _init_logging(self) -> None:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=LOG_LEVEL,
        )
        if self.accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        logger.info("Initialized Trainer")
        logger.info(f"Accelerator state: \n{self.accelerator.state}", main_process_only=False)

    def _init_directories(self) -> None:
        if self.args.exp_dir is None:
            run_id_list = [None]
            if self.accelerator.is_main_process:
                run_id_list[0] = os.environ.get('WANDB_RUN_ID', wandb.util.generate_id())
                os.environ['WANDB_RUN_ID'] = run_id_list[0]
            broadcast_object_list(run_id_list)
            run_id = run_id_list[0]

            self.exp_dir = Path(self.args.output_dir) / run_id
            if self.accelerator.is_main_process:
                self.exp_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Output directory: {self.exp_dir}")
        else:
            self.exp_dir = Path(self.args.exp_dir)

        if self.args.backup_dir is not None:
            self.backup_dir = Path(self.args.backup_dir) / run_id
            if self.accelerator.is_main_process:
                logger.info(f"Backup directory: {self.backup_dir}")
        else:
            self.backup_dir = None

    def check_setting(self) -> None:
        # Check for unload_list
        if self.UNLOAD_LIST is None:
            logger.warning(
                "\033[91mNo unload_list specified for this Trainer. All components will be loaded to GPU during training.\033[0m"
            )
        else:
            for name in self.UNLOAD_LIST:
                if name not in self.components.model_fields:
                    raise ValueError(f"Invalid component name in unload_list: {name}")

    def prepare_models(self) -> None:
        logger.info("Initializing models")

        if self.components.vae is not None:
            if self.args.enable_slicing:
                self.components.vae.enable_slicing()
            if self.args.enable_tiling:
                self.components.vae.enable_tiling()

        self.state.transformer_config = self.components.transformer.config

    def prepare_dataset(self) -> None:
        logger.info("Initializing training dataset and dataloader")

        # Prepare VAE and text encoder for encoding
        self.components.vae.requires_grad_(False)
        self.components.text_encoder.requires_grad_(False)
        self.components.vae = self.components.vae.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )
        self.components.text_encoder = self.components.text_encoder.to(
            self.accelerator.device, dtype=self.state.weight_dtype
        )

        self.dataset, self.data_loader = self.prepare_dataloader(
            globals()[self.args.train_dataset],
            split="train",
            cache="r",
            collate_fn=self.collate_fn,
            num_workers=self.args.num_workers,
            num_samples=self.args.num_train_samples,
        )

        unload_model(self.components.vae)
        unload_model(self.components.text_encoder)
        free_memory()

    def prepare_test_dataset(self) -> None:
        logger.info("Initializing test dataset and dataloader")

        self.test_dataset, self.test_dataloader = self.prepare_dataloader(
            globals()[self.args.test_dataset],
            split="test",
            cache="r",
            collate_fn=self.validation_collate_fn,
            num_workers=1,
            num_samples=self.args.num_test_samples,
            start_index=self.args.test_start_index,
        )

    def prepare_trainable_parameters(self):
        logger.info("Initializing trainable parameters")

        # For mixed precision training we cast all non-trainable weights to half-precision
        # as these weights are only used for inference, keeping weights in full precision is not required.
        weight_dtype = self.state.weight_dtype

        if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
            # due to pytorch#99272, MPS does not yet support bfloat16.
            raise ValueError(
                "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
            )

        # For LoRA, we freeze all the parameters
        # For SFT, we train all the parameters in transformer model
        for attr_name, component in vars(self.components).items():
            if hasattr(component, "requires_grad_"):
                if self.args.training_type == "sft" and attr_name == "transformer":
                    component.requires_grad_(True)
                else:
                    component.requires_grad_(False)

        if self.args.training_type == "lora":
            transformer_lora_config = LoraConfig(
                r=self.args.rank,
                lora_alpha=self.args.lora_alpha,
                init_lora_weights=True,
                target_modules=self.args.target_modules,
            )
            self.components.transformer.add_adapter(transformer_lora_config)
            self.__prepare_saving_loading_hooks(transformer_lora_config)

        # Load components needed for training to GPU (except transformer), and cast them to the specified data type
        ignore_list = ["transformer"] + self.UNLOAD_LIST
        self.__move_components_to_device(dtype=weight_dtype, ignore_list=ignore_list)

        if self.args.gradient_checkpointing:
            self.components.transformer.enable_gradient_checkpointing()

    def prepare_optimizer(self) -> None:
        logger.info("Initializing optimizer and lr scheduler")

        # Make sure the trainable params are in float32
        cast_training_params([self.components.transformer], dtype=torch.float32)

        # For LoRA, we only want to train the LoRA weights
        # For SFT, we want to train all the parameters
        trainable_parameters = list(
            filter(lambda p: p.requires_grad, self.components.transformer.parameters())
        )
        transformer_parameters_with_lr = {
            "params": trainable_parameters,
            "lr": self.args.learning_rate,
        }
        params_to_optimize = [transformer_parameters_with_lr]
        self.state.num_trainable_parameters = sum(p.numel() for p in trainable_parameters)

        use_deepspeed_opt = (
            self.accelerator.state.deepspeed_plugin is not None
            and "optimizer" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        optimizer = get_optimizer(
            params_to_optimize=params_to_optimize,
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.learning_rate,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_deepspeed=use_deepspeed_opt,
        )

        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.args.train_steps is None:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        use_deepspeed_lr_scheduler = (
            self.accelerator.state.deepspeed_plugin is not None
            and "scheduler" in self.accelerator.state.deepspeed_plugin.deepspeed_config
        )
        total_training_steps = self.args.train_steps * self.accelerator.num_processes
        num_warmup_steps = self.args.lr_warmup_steps * self.accelerator.num_processes

        if use_deepspeed_lr_scheduler:
            from accelerate.utils import DummyScheduler

            lr_scheduler = DummyScheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                total_num_steps=total_training_steps,
                num_warmup_steps=num_warmup_steps,
            )
        else:
            lr_scheduler = get_scheduler(
                name=self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_training_steps,
                num_cycles=self.args.lr_num_cycles,
                power=self.args.lr_power,
            )

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def prepare_for_training(self) -> None:
        self.components.transformer, self.optimizer, self.lr_scheduler = (
            self.accelerator.prepare(
                self.components.transformer, self.optimizer, self.lr_scheduler
            )
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(
            len(self.data_loader) / self.args.gradient_accumulation_steps
        )
        if self.state.overwrote_max_train_steps:
            self.args.train_steps = self.args.train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        self.args.train_epochs = math.ceil(self.args.train_steps / num_update_steps_per_epoch)
        self.state.num_update_steps_per_epoch = num_update_steps_per_epoch

    @property
    @abstractmethod
    def train_data_keys():
        ...

    @property
    @abstractmethod
    def validation_data_keys(self):
        ...

    def prepare_dataloader(
        self,
        dataset_cls,
        split,
        collate_fn,
        num_workers,
        cache="r",
        num_samples=None,
        drop_last=False,
        precompute_cache=True,
        batch_size=None,
        shuffle=None,
        start_index=0,
    ):
        if precompute_cache:
            logger.info("Precomputing cache for dataset")
            _, dataloader = self.prepare_dataloader(
                dataset_cls,
                split=split,
                cache="rw",
                collate_fn=lambda x: None,
                num_workers=0,
                num_samples=num_samples,
                drop_last=drop_last,
                precompute_cache=False,
                batch_size=1,
                shuffle=False,
                start_index=start_index,
            )

            for _ in tqdm(
                dataloader,
                desc="Precomputing",
                disable=not self.accelerator.is_local_main_process,
            ):
                ...
            self.accelerator.wait_for_everyone()

        keys = self.train_data_keys if split == "train" else self.validation_data_keys
        if batch_size is None:
            batch_size = 1 if split == "test" else self.args.batch_size
        shuffle = (False if split == "test" else True) if shuffle is None else shuffle
        device_placement = False if split == "test" else True

        dataset = dataset_cls(
            self.args,
            split=split,
            device=self.accelerator.device,
            trainer=self,
            keys=keys,
            cache=cache,
        )
        if start_index > 0:
            dataset = torch.utils.data.Subset(
                dataset, range(start_index, len(dataset))
            )
        if num_samples is not None and len(dataset) > num_samples:
            dataset = torch.utils.data.Subset(
                dataset, range(num_samples)
            )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=self.args.pin_memory,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        dataloader = self.accelerator.prepare_data_loader(
            dataloader,
            device_placement=device_placement,
        )

        return dataset, dataloader

    def prepare_validation_dataset(self):
        logger.info("Initializing validation dataset and dataloader")

        if self.args.num_validation_samples_per_gpu is not None:
            num_samples = self.args.num_validation_samples_per_gpu * self.accelerator.num_processes
        else:
            num_samples = None

        self.validation_dataset, self.validation_dataloader = self.prepare_dataloader(
            globals()[self.args.test_dataset],
            split="test",
            collate_fn=self.validation_collate_fn,
            num_workers=1,
            num_samples=num_samples,
            drop_last=True,
        )

    def prepare_for_testing(self) -> None:
        self.components.transformer = self.accelerator.prepare(self.components.transformer)

    def prepare_trackers(self) -> None:
        logger.info("Initializing trackers")

        tracker_name = self.args.tracker_name
        self.accelerator.init_trackers(
            tracker_name,
            config=self.args.model_dump(),
            init_kwargs={
                "wandb": {
                    "entity": self.args.wandb_entity,
                    "dir": str(self.exp_dir),
                    "resume": "allow",
                    "settings": wandb.Settings(
                        code_dir=".",
                    )
                }
            }
        )

    def train(self) -> None:
        logger.info("Starting training")

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.total_batch_size_count = (
            self.args.batch_size
            * self.accelerator.num_processes
            * self.args.gradient_accumulation_steps
        )
        info = {
            "trainable parameters": self.state.num_trainable_parameters,
            "total samples": len(self.dataset),
            "train epochs": self.args.train_epochs,
            "train steps": self.args.train_steps,
            "batches per device": self.args.batch_size,
            "total batches observed per epoch": len(self.data_loader),
            "train batch size total count": self.state.total_batch_size_count,
            "gradient accumulation steps": self.args.gradient_accumulation_steps,
        }
        logger.info(f"Training configuration: {json.dumps(info, indent=4)}")

        global_step = 0
        first_epoch = 0
        initial_global_step = 0

        # Potentially load in the weights and states from a previous save
        if self.args.resume_from_checkpoint is None:
            self.args.resume_from_checkpoint = get_latest_ckpt_path(self.exp_dir)

        if self.args.resume_from_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {self.args.resume_from_checkpoint}")
            (
                resume_from_checkpoint_path,
                initial_global_step,
                global_step,
                first_epoch,
            ) = get_latest_ckpt_path_to_resume_from(
                resume_from_checkpoint=self.args.resume_from_checkpoint,
                num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
            )
            logger.info(
                f"Initial global step: {initial_global_step}, "
                f"global step: {global_step}, "
                f"first epoch: {first_epoch}"
            )
        elif self.args.finetune_from_checkpoint is not None:
            logger.info(f"Fine-tuning from checkpoint: {self.args.finetune_from_checkpoint}")
            resume_from_checkpoint_path = get_latest_ckpt_path_to_resume_from(
                resume_from_checkpoint=self.args.finetune_from_checkpoint,
                num_update_steps_per_epoch=self.state.num_update_steps_per_epoch,
            )[0]
        else:
            resume_from_checkpoint_path = None

        if resume_from_checkpoint_path is not None:
            self.accelerator.load_state(resume_from_checkpoint_path)

        progress_bar = tqdm(
            range(0, self.args.train_steps),
            initial=initial_global_step,
            desc="Training steps",
            disable=not self.accelerator.is_local_main_process,
        )

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator

        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        free_memory()
        for epoch in range(first_epoch, self.args.train_epochs):
            logger.debug(f"Starting epoch ({epoch + 1}/{self.args.train_epochs})")

            self.components.transformer.train()
            models_to_accumulate = [self.components.transformer]

            for step, batch in enumerate(self.data_loader):
                logger.debug(f"Starting step {step + 1}")
                logs = {}

                with accelerator.accumulate(models_to_accumulate):
                    # These weighting schemes use a uniform timestep sampling and instead post-weight the loss
                    loss = self.compute_loss(batch)
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        if accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = self.components.transformer.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = accelerator.clip_grad_norm_(
                                self.components.transformer.parameters(), self.args.max_grad_norm
                            )
                            if torch.is_tensor(grad_norm):
                                grad_norm = grad_norm.item()

                        logs["grad_norm"] = grad_norm

                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    self.__maybe_save_checkpoint(global_step)

                logs["loss"] = loss.detach().item()
                logs["lr"] = self.lr_scheduler.get_last_lr()[0]
                progress_bar.set_postfix(logs)

                # Maybe run validation
                should_run_validation = (
                    self.args.do_validation and global_step % self.args.validation_steps == 0
                )
                if should_run_validation:
                    del loss
                    free_memory()
                    self.validate(global_step)

                accelerator.log(logs, step=global_step)

                if global_step >= self.args.train_steps:
                    break

            memory_statistics = get_memory_statistics()
            logger.info(
                f"Memory after epoch {epoch + 1}: {json.dumps(memory_statistics, indent=4)}"
            )

        accelerator.wait_for_everyone()
        self.__maybe_save_checkpoint(global_step, must_save=True, backup=True, upload=True)
        if self.args.do_validation:
            free_memory()
            self.validate(global_step)

        del self.components
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")

        accelerator.end_training()

    def validate(self, step: int) -> None:
        logger.info("Starting validation")

        accelerator = self.accelerator
        num_validation_samples = len(self.validation_dataset)

        if num_validation_samples == 0:
            logger.warning("No validation samples found. Skipping validation.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before validation start: {json.dumps(memory_statistics, indent=4)}")

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()

        if self.state.using_deepspeed:
            # Can't using model_cpu_offload in deepspeed,
            # so we need to move all components in pipe to device
            # pipe.to(self.accelerator.device, dtype=self.state.weight_dtype)
            self.__move_components_to_device(
                dtype=self.state.weight_dtype, ignore_list=["transformer"]
            )
        else:
            # if not using deepspeed, use model_cpu_offload to further reduce memory usage
            # Or use pipe.enable_sequential_cpu_offload() to further reduce memory usage
            pipe.enable_model_cpu_offload(device=self.accelerator.device)

            # Convert all model weights to training dtype
            # Note, this will change LoRA weights in self.components.transformer to training dtype, rather than keep them in fp32
            pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        all_processes_artifacts = []
        for i, data in enumerate(self.validation_dataloader):
            i_sample = i * self.accelerator.num_processes + accelerator.process_index
            logger.info(
                f"Validating sample {i_sample + 1}/{num_validation_samples} on process {accelerator.process_index}. Prompt: {data['caption']}",
                main_process_only=False,
            )
            validation_artifacts = self.validation_step(data, pipe)

            if (
                self.state.using_deepspeed
                and self.accelerator.deepspeed_plugin.zero_stage == 3
                and not accelerator.is_main_process
            ):
                continue

            artifacts = {
                "image": {"type": "image", "value": data["image"]},
                "video": {"type": "video", "value": data["video"]},
                "prompt": {"type": "text", "value": data["caption"]},
            }
            for j, (artifact_type, artifact_value) in enumerate(validation_artifacts):
                artifacts.update(
                    {f"artifact_{j}": {"type": artifact_type, "value": artifact_value}}
                )
            logger.debug(
                f"Validation artifacts on process {accelerator.process_index}: {list(artifacts.keys())}",
                main_process_only=False,
            )

            for key, value in list(artifacts.items()):
                artifact_type = value["type"]
                artifact_value = value["value"]
                if artifact_type not in ["image", "video", "text"] or artifact_value is None:
                    continue

                filename = f"validation-{step}-{i_sample}-{key}-{data['video_id']}-{data['clip_id']}"
                validation_path = self.exp_dir / "validation_res"
                validation_path.mkdir(parents=True, exist_ok=True)
                filename = validation_path / filename

                if artifact_type == "image":
                    file_path = filename.with_suffix(".png")
                    logger.debug(f"Saving image to {file_path}", main_process_only=False)
                    artifact_value.save(file_path)
                    artifact_value = wandb.Image(str(file_path), caption=data["caption"])
                elif artifact_type == "video":
                    file_path = filename.with_suffix(".mp4")
                    logger.debug(f"Saving video to {file_path}", main_process_only=False)
                    export_to_video(artifact_value, file_path, fps=self.args.fps)
                    artifact_value = wandb.Video(str(file_path))
                elif artifact_type == "text":
                    file_path = filename.with_suffix(".txt")
                    logger.debug(f"Saving text to {file_path}", main_process_only=False)
                    with open(file_path, "w") as f:
                        f.write(artifact_value)

                all_processes_artifacts.append(artifact_value)

        accelerator.wait_for_everyone()
        all_artifacts = gather_object(all_processes_artifacts)

        if accelerator.is_main_process:
            tracker_key = "validation"
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    image_artifacts = [
                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Image)
                    ]
                    video_artifacts = [
                        artifact for artifact in all_artifacts if isinstance(artifact, wandb.Video)
                    ]
                    tracker.log(
                        {
                            tracker_key: {"images": image_artifacts, "videos": video_artifacts},
                        },
                        step=step,
                    )

        ##########  Clean up  ##########
        if self.state.using_deepspeed:
            del pipe
            # Unload models except those needed for training
            self.__move_components_to_cpu(unload_list=self.UNLOAD_LIST)
        else:
            pipe.remove_all_hooks()
            del pipe
            # Load models except those not needed for training
            self.__move_components_to_device(
                dtype=self.state.weight_dtype, ignore_list=self.UNLOAD_LIST
            )
            self.components.transformer.to(self.accelerator.device, dtype=self.state.weight_dtype)

            # Change trainable weights back to fp32 to keep with dtype after prepare the model
            cast_training_params([self.components.transformer], dtype=torch.float32)

        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)

        torch.set_grad_enabled(True)
        self.components.transformer.train()

    def fit(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_dataset()
        if self.args.do_validation:
            self.prepare_validation_dataset()
        self.prepare_trainable_parameters()
        self.prepare_optimizer()
        self.prepare_for_training()
        self.prepare_trackers()
        self.train()

    def test(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_test_dataset()
        self.prepare_trainable_parameters()
        self.prepare_for_testing()
        self.run_test()

    def run_test(self):
        logger.info("Starting testing")

        accelerator = self.accelerator
        generator = torch.Generator(device=accelerator.device)
        if self.args.seed is not None:
            generator = generator.manual_seed(self.args.seed)
        self.state.generator = generator
        num_test_samples = len(self.test_dataset)

        if num_test_samples == 0:
            logger.warning("No test samples found. Skipping test.")
            return

        self.components.transformer.eval()
        torch.set_grad_enabled(False)

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before test start: {json.dumps(memory_statistics, indent=4)}")

        if self.args.resume_from_checkpoint is None:
            self.args.resume_from_checkpoint = get_latest_ckpt_path(self.exp_dir)
        if self.args.resume_from_checkpoint is not None:
            logger.info(
                f"Resuming from checkpoint: {self.args.resume_from_checkpoint}",
                main_process_only=False,
            )
            self.accelerator.load_state(self.args.resume_from_checkpoint)

        #####  Initialize pipeline  #####
        pipe = self.initialize_pipeline()
        pipe.enable_model_cpu_offload(device=self.accelerator.device)
        pipe = pipe.to(dtype=self.state.weight_dtype)

        #################################

        exp_name = self.args.data_root.name
        if self.args.test_suffix is not None:
            exp_name += f"-{self.args.test_suffix}"
        test_res_path = self.exp_dir / exp_name / "test_res"

        for i, data in enumerate(self.test_dataloader):
            i_sample = i * self.accelerator.num_processes + accelerator.process_index
            logger.info(
                f"Testing sample {i_sample + 1}/{num_test_samples} on process {accelerator.process_index}. Prompt: {data['caption']}",
                main_process_only=False,
            )
            clip_dir = test_res_path / data["video_id"] / data["clip_name"]
            clip_dir.mkdir(parents=True, exist_ok=True)
            for j in range(self.args.num_videos_per_prompt):
                test_artifacts = self.validation_step(data, pipe)
                file_path = clip_dir / f"{j}.mp4"
                export_to_video(test_artifacts[0][1], file_path, fps=self.args.fps)
                logger.info(f"Saved test video to {file_path}")

        ##########  Clean up  ##########
        pipe.remove_all_hooks()
        del pipe
        free_memory()
        accelerator.wait_for_everyone()
        ################################

        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after validation end: {json.dumps(memory_statistics, indent=4)}")
        torch.cuda.reset_peak_memory_stats(accelerator.device)


    def collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError

    def validation_collate_fn(self, examples: List[Dict[str, Any]]):
        raise NotImplementedError

    def load_components(self) -> Components:
        raise NotImplementedError

    def initialize_pipeline(self) -> DiffusionPipeline:
        raise NotImplementedError

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W], where B = 1
        # shape of output video: [B, C', F', H', W'], where B = 1
        raise NotImplementedError

    def encode_text(self, text: str) -> torch.Tensor:
        # shape of output text: [batch size, sequence length, embedding dimension]
        raise NotImplementedError

    def compute_loss(self, batch) -> torch.Tensor:
        raise NotImplementedError

    def validation_step(self) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        raise NotImplementedError

    def __get_training_dtype(self) -> torch.dtype:
        if self.args.mixed_precision == "no":
            return _DTYPE_MAP["fp32"]
        elif self.args.mixed_precision == "fp16":
            return _DTYPE_MAP["fp16"]
        elif self.args.mixed_precision == "bf16":
            return _DTYPE_MAP["bf16"]
        else:
            raise ValueError(f"Invalid mixed precision: {self.args.mixed_precision}")

    def __move_components_to_device(self, dtype, ignore_list: List[str] = []):
        ignore_list = set(ignore_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name not in ignore_list:
                    setattr(
                        self.components, name, component.to(self.accelerator.device, dtype=dtype)
                    )

    def __move_components_to_cpu(self, unload_list: List[str] = []):
        unload_list = set(unload_list)
        components = self.components.model_dump()
        for name, component in components.items():
            if not isinstance(component, type) and hasattr(component, "to"):
                if name in unload_list:
                    setattr(self.components, name, component.to("cpu"))

    def __prepare_saving_loading_hooks(self, transformer_lora_config):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if self.accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        model = unwrap_model(self.accelerator, model)
                        transformer_lora_layers_to_save = get_peft_model_state_dict(model)
                    else:
                        raise ValueError(f"Unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                self.components.pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_lora_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            if not self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()
                    if isinstance(
                        unwrap_model(self.accelerator, model),
                        type(unwrap_model(self.accelerator, self.components.transformer)),
                    ):
                        transformer_ = unwrap_model(self.accelerator, model)
                    else:
                        raise ValueError(
                            f"Unexpected save model: {unwrap_model(self.accelerator, model).__class__}"
                        )
            else:
                transformer_ = unwrap_model(
                    self.accelerator, self.components.transformer
                ).__class__.from_pretrained(self.args.model_path, subfolder="transformer")
                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = self.components.pipeline_cls.lora_state_dict(
                input_dir, weight_name="pytorch_lora_weights.safetensors"
            )
            transformer_state_dict = {
                f'{k.replace("transformer.", "")}': v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.")
            }
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )

        self.accelerator.register_save_state_pre_hook(save_model_hook)
        self.accelerator.register_load_state_pre_hook(load_model_hook)

    def __maybe_save_checkpoint(
        self,
        global_step: int,
        must_save: bool = False,
        backup: bool = False,
        upload: bool = False,
    ):
        if (
            self.accelerator.distributed_type == DistributedType.DEEPSPEED
            or self.accelerator.is_main_process
        ):
            if must_save or global_step % self.args.checkpointing_steps == 0:
                # for training
                save_path = get_intermediate_ckpt_path(
                    checkpointing_limit=self.args.checkpointing_limit,
                    step=global_step,
                    output_dir=self.exp_dir,
                )
                self.accelerator.save_state(save_path, safe_serialization=True)

                if backup and self.backup_dir is not None:
                    self.backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = self.backup_dir / f"checkpoint-{global_step}"
                    self.accelerator.save_state(backup_path, safe_serialization=True)

                if (
                    self.args.upload_ckpt_to_wandb \
                    and upload \
                    and any([tracker.name == "wandb" for tracker in self.accelerator.trackers])
                ):
                    art = wandb.Artifact("checkpoint", type="model")
                    art.add_dir(save_path, name="checkpoint")
                    wandb.log_artifact(art)
