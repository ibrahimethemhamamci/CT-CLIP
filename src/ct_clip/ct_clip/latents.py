from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator
from transformers import BertTokenizer

from ct_clip import CTCLIP

T = TypeVar("T")


def combine(instance: T | None = None, instances: list[T] | None = None) -> list[T]:
    instances = instances or []
    if instance:
        instances.append(instance)
    return instances


@dataclass
class Latents:
    texts: list[npt.NDArray]
    images: list[npt.NDArray]


class CTClipLatents:
    def __init__(
        self,
        tokenizer: BertTokenizer,
        ctclip: CTCLIP,
        path_to_pretrained_model: str,
    ):
        self.tokenizer = tokenizer
        self.ctclip = ctclip

        self.accelerator = Accelerator()
        self.ctclip.load(path_to_pretrained_model, self.accelerator.device)

    def generate_latents(
        self,
        text: str = None,
        texts: list[str] = None,
        image: torch.Tensor = None,
        images: list[torch.Tensor] = None,
        max_text_length: int = 512,
    ) -> Latents:
        texts = combine(text, texts)
        images = combine(image, images)

        text_latents = [
            self.generate_text_latents(text, max_text_length) for text in texts
        ]
        image_latents = [self.generate_image_latents(image) for image in images]

        return Latents(text_latents, image_latents)

    def generate_text_latents(self, text: str, max_length: int = 512) -> npt.NDArray:
        text_tokens = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        text_latents: torch.Tensor
        text_latents, _ = self.ctclip(text=text_tokens, return_latents=True)
        return text_latents.detach().numpy()

    def generate_image_latents(self, image: torch.Tensor) -> npt.NDArray:
        _, image_latents = self.ctclip(image=image, return_latents=True)
        return image_latents.detach().numpy()
