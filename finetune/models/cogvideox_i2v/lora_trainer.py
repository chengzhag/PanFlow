from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXTransformer3DModel,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from numpy import dtype
from transformers import AutoTokenizer, T5EncoderModel, T5Config
from typing_extensions import override
from einops import rearrange
from pathlib import Path

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
from finetune.datasets import BaseDataset
import rp
rp.r._pip_import_autoyes = True  # Automatically install missing packages
rp.git_import("CommonSource")
import rp.git.CommonSource.noise_warp as nw

from ..utils import register
from finetune.utils.erp_utils import equilib_rotation
from equilib import equi2equi
from finetune.pipelines.pipeline_cogvideox_panorama_image2video import (
    CogVideoXPanoramaImageToVideoPipeline,
)
from finetune.pipelines.autoencoder_kl_cogvideox_panorama import (
    AutoencoderKLCogVideoXPanorama,
)


class CogVideoXI2VLoraTrainer(Trainer):
    UNLOAD_LIST = ["text_encoder"]

    @property
    def train_data_keys(self):
        train_keys = ["encoded_video", "prompt_embedding", "image", "noise"]
        if self.args.train_dataset == "PanFlowDataset":
            train_keys.append("derotation")
        return train_keys

    @property
    def validation_data_keys(self):
        validation_keys = ["image", "video", "noise"]
        if self.args.test_dataset == "PanFlowDataset":
            validation_keys.append("derotation_R")
        return validation_keys

    @override
    def load_components(self) -> Dict[str, Any]:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPanoramaImageToVideoPipeline

        func_name = "from_config" if self.args.skip_model_load else "from_pretrained"

        components.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            subfolder="tokenizer",
            low_cpu_mem_usage=True,
        )

        if self.args.skip_model_load:
            cfg = T5Config.from_pretrained(
                model_path,
                subfolder="text_encoder",
            )
            components.text_encoder = T5EncoderModel(cfg)
        else:
            components.text_encoder = T5EncoderModel.from_pretrained(
                model_path,
                subfolder="text_encoder",
                low_cpu_mem_usage=True,
            )

        components.vae = getattr(AutoencoderKLCogVideoXPanorama, func_name)(
            model_path,
            subfolder="vae",
            low_cpu_mem_usage=True,
        )
        components.vae.circular_padding = self.args.circular_padding

        components.scheduler = getattr(CogVideoXDDIMScheduler, func_name)(
            model_path,
            subfolder="scheduler",
            low_cpu_mem_usage=True,
        )

        # Fuse LoRA weights into the model
        if self.args.fuse_lora_checkpoint:
            # TODO: Automatically download the LoRA weights if not present
            fuse_transformer_path = (
                self.args.fuse_lora_checkpoint.parent / 
                f"{self.args.fuse_lora_checkpoint.stem}-fused_transformer"
            )
            if not fuse_transformer_path.exists():
                print(
                    f"Fusing LoRA weights into the model: {self.args.fuse_lora_checkpoint}"
                )
                pipe = CogVideoXPanoramaImageToVideoPipeline(
                    tokenizer=components.tokenizer,
                    text_encoder=components.text_encoder,
                    vae=components.vae,
                    transformer=CogVideoXTransformer3DModel.from_pretrained(
                        model_path,
                        subfolder="transformer",
                        torch_dtype=self.state.weight_dtype,
                        low_cpu_mem_usage=True,
                    ),
                    scheduler=components.scheduler,
                )
                pipe.load_lora_weights(
                    str(self.args.fuse_lora_checkpoint),
                    adapter_name="lora",
                )
                pipe.fuse_lora()
                pipe.unload_lora_weights()
                pipe.transformer.save_pretrained(
                    fuse_transformer_path,
                    safe_serialization=True,
                )
                model_path = fuse_transformer_path
                print("LoRA weights fused into the model.")

            print(f"Loading fused model from {fuse_transformer_path}")
            components.transformer = getattr(CogVideoXTransformer3DModel, func_name)(
                fuse_transformer_path,
                low_cpu_mem_usage=True,
            )
            print("Fused model loaded.")
        else:
            components.transformer = getattr(CogVideoXTransformer3DModel, func_name)(
                model_path,
                subfolder="transformer",
                low_cpu_mem_usage=True,
            )

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPanoramaImageToVideoPipeline:
        pipe = CogVideoXPanoramaImageToVideoPipeline(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(
            prompt_token_ids.to(self.accelerator.device)
        )[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {}
        for key in self.train_data_keys:
            value = []
            for sample in samples:
                value.append(sample[key])
            ret[key] = torch.stack(value)
        return ret

    @override
    def validation_collate_fn(
        self, samples: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        assert len(samples) == 1
        data = samples[0]

        # Convert image tensor (C, H, W) to PIL images
        image = data["image"]
        image = BaseDataset.inverse_video_transform(image).cpu().numpy()
        image = Image.fromarray(image)
        data["image"] = image

        # Convert video tensor (F, C, H, W) to list of PIL images
        video = data["video"]
        video = BaseDataset.inverse_video_transform(video).cpu().numpy()
        video = [Image.fromarray(frame) for frame in video]
        data["video"] = video

        return data

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_video"]
        images = batch["image"]
        noise = batch["noise"]

        # Shape of prompt_embedding: [B, seq_len, hidden_size]
        # Shape of latent: [B, C, F, H, W]
        # Shape of images: [B, C, H, W]

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]  # Get first frame [B, C, 1, H, W]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Add frame dimension to images [B,C,H,W] -> [B,C,F,H,W]
        images = images.unsqueeze(2)
        # Add noise to images
        image_noise_sigma = torch.normal(
            mean=-3.0, std=0.5, size=(1,), device=self.accelerator.device
        )
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype=images.dtype)
        noisy_images = (
            images + torch.randn_like(images) * image_noise_sigma[:, None, None, None, None]
        )
        image_latent_dist = self.components.vae.encode(
            noisy_images.to(dtype=self.components.vae.dtype)
        ).latent_dist
        image_latents = image_latent_dist.sample() * self.components.vae.config.scaling_factor

        # Sample a random timestep for each sample
        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        )
        timesteps = timesteps.long()

        # from [B, C, F, H, W] to [B, F, C, H, W]
        latent = latent.permute(0, 2, 1, 3, 4)
        image_latents = image_latents.permute(0, 2, 1, 3, 4)
        assert (latent.shape[0], *latent.shape[2:]) == (
            image_latents.shape[0],
            *image_latents.shape[2:],
        )

        # Padding image_latents to the same frame number as latent
        padding_shape = (latent.shape[0], latent.shape[1] - 1, *latent.shape[2:])
        latent_padding = image_latents.new_zeros(padding_shape)
        image_latents = torch.cat([image_latents, latent_padding], dim=1)

        # Add noise to latent
        noise_alpha = torch.rand(batch_size, device=self.accelerator.device)
        if self.args.derotation == "random":
            noise_alpha = noise_alpha ** 0.5
            noise_alpha = (batch["derotation"].float() - noise_alpha).abs()
        noise_alpha = rearrange(noise_alpha, "b -> b 1 1 1 1")
        noise = nw.mix_new_noise(noise, noise_alpha)
        latent_noisy = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Concatenate latent and image_latents in the channel dimension
        latent_img_noisy = torch.cat([latent_noisy, image_latents], dim=2)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Predict noise, For CogVideoX1.5 Only.
        ofs_emb = (
            None
            if self.state.transformer_config.ofs_embed_dim is None
            else latent.new_full((1,), fill_value=2.0)
        )
        predicted_noise = self.components.transformer(
            hidden_states=latent_img_noisy,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            ofs=ofs_emb,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )[0]

        # Denoise
        latent_pred = self.components.scheduler.get_velocity(
            predicted_noise, latent_noisy, timesteps
        )

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return loss

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPanoramaImageToVideoPipeline
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
        """
        prompt, image, noise = eval_data["caption"], eval_data["image"], eval_data["noise"]

        noise = noise.to(self.state.weight_dtype).to(self.accelerator.device)
        noise = noise.unsqueeze(0)
        noise = nw.mix_new_noise(noise, self.args.noise_alpha)

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            image=image,
            latents=noise,
            generator=self.state.generator,
            num_inference_steps=self.args.num_inference_steps,
            guidance_scale=self.args.guidance_scale,
            output_type="pt",
            latent_rotation=self.args.latent_rotation,
            latent_rotation_degree=self.args.latent_rotation_degree,
        ).frames[0]
        video_generate = video_generate.cpu().to(torch.float32)

        if self.args.derotation == "yes" and "derotation_R" in eval_data:
            derotation_R = eval_data["derotation_R"].cpu()
            rotation = equilib_rotation(derotation_R.transpose(-1, -2))
            video_generate = equi2equi(video_generate, rotation)

        if self.args.rotate_180:
            video_generate = torch.roll(video_generate, shifts=video_generate.shape[-1] // 2, dims=-1)

        video_generate = pipe.video_processor.postprocess(
            image=video_generate,
            output_type="pil",
            do_denormalize=[False] * len(video_generate),
        )

        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-i2v", "lora", CogVideoXI2VLoraTrainer)
