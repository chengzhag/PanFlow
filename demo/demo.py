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
from accelerate.logging import get_logger

sys.path.append(str(Path(__file__).parent.parent))
from finetune.constants import LOG_LEVEL, LOG_NAME
from finetune.schemas import Args
from finetune.datasets import *
from finetune.models.cogvideox_i2v.lora_trainer import CogVideoXI2VLoraTrainer
from finetune.utils import (
    free_memory,
    get_latest_ckpt_path,
)

logger = get_logger(LOG_NAME, LOG_LEVEL)


class DemoTrainer(CogVideoXI2VLoraTrainer):
    def demo(self):
        self.check_setting()
        self.prepare_models()
        self.prepare_demo_dataset()
        self.prepare_trainable_parameters()
        self.prepare_for_testing()
        self.run_demo()

    def prepare_demo_dataset(self) -> None:
        logger.info("Initializing demo dataset and dataloader")

        self.test_dataset, self.test_dataloader = self.prepare_dataloader(
            globals()[self.args.test_dataset],
            split="test",
            cache="r",
            collate_fn=self.validation_collate_fn,
            num_workers=1,
            num_samples=self.args.num_test_samples,
        )

    def run_demo(self):
        logger.info("Starting demo")

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

        for i, data in enumerate(self.test_dataloader):
            i_sample = i * self.accelerator.num_processes + accelerator.process_index
            logger.info(
                f"Testing sample {i_sample + 1}/{num_test_samples} on process {accelerator.process_index}. Prompt: {data['caption']}",
                main_process_only=False,
            )
            demo_dir = self.exp_dir / data["demo_name"] / data["demo_folder"] / data["output_name"]
            demo_dir.parent.mkdir(parents=True, exist_ok=True)

            test_artifacts = self.validation_step(data, pipe)
            file_path = demo_dir.with_suffix(".mp4")
            export_to_video(test_artifacts[0][1], file_path, fps=self.args.fps)
            logger.info(f"Saved output video to {file_path}")

            file_path = demo_dir.with_suffix(".txt")
            with open(file_path, "w") as f:
                f.write(data["caption"])
            logger.info(f"Saved prompt to {file_path}")

            file_path = demo_dir.with_suffix(".png")
            data["image"].save(file_path)

            file_path = demo_dir.parent / f"{data['motion_from']}.mp4"
            export_to_video(data["video"], file_path, fps=self.args.fps)
            logger.info(f"Saved source video to {file_path}")

        ##########  Clean up  ##########
        pipe.remove_all_hooks()
        del pipe
        free_memory()
        accelerator.wait_for_everyone()
        ################################


def main():
    args = Args.parse_args()
    trainer = DemoTrainer(args)
    trainer.demo()


if __name__ == "__main__":
    main()
