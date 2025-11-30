
from typing import Dict, Optional, Tuple, Union

import torch

from diffusers.utils import logging
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers import AutoencoderKLCogVideoX


class AutoencoderKLCogVideoXPanorama(AutoencoderKLCogVideoX):
    circular_padding = 1

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.circular_padding > 0:
            x = torch.cat([
                x[..., -self.circular_padding * 8:],
                x,
                x[..., :self.circular_padding * 8]
            ], dim=-1)

        enc = super()._encode(x)

        if self.circular_padding > 0:
            enc = enc[..., self.circular_padding:-self.circular_padding]
        return enc

    def _decode(
        self, z: torch.Tensor, return_dict: bool = True,
    ) -> Union[DecoderOutput, torch.Tensor]:
        if self.circular_padding > 0:
            z = torch.cat([
                z[..., -self.circular_padding:],
                z,
                z[..., :self.circular_padding]
            ], dim=-1)

        dec = super()._decode(z)

        if isinstance(dec, DecoderOutput):
            dec = dec.sample

        if self.circular_padding > 0:
            dec = dec[..., self.circular_padding * 8:-self.circular_padding * 8]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)
        
