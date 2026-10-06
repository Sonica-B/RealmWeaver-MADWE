"""Style vectors: DINOv2 CLS embeddings, L2-normalised (384-d for `facebook/dinov2-small`)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from realmweaver.config import settings


class DinoEmbedder:
    """Callable `image -> style vector`. Runs on `settings().device` unless `device` is given; CPU works too."""

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        device: str | None = None,
        model: Any | None = None,
        processor: Any | None = None,
    ) -> None:
        """`model` and `processor` are a transformers `AutoModel` / `AutoImageProcessor` pair; either one not
        given is loaded from `model_id`."""
        self.device = torch.device(device or settings().device)
        if processor is None or model is None:
            from transformers import AutoImageProcessor, AutoModel

            processor = AutoImageProcessor.from_pretrained(model_id) if processor is None else processor
            model = AutoModel.from_pretrained(model_id) if model is None else model
        self._processor = processor
        self._model = model.to(self.device).eval()
        self.dim = int(self._model.config.hidden_size)

    @torch.inference_mode()
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """`img` is uint8 HxWx3 or HxWx4 (alpha dropped). Returns float32 `[dim]` with unit L2 norm."""
        inputs = self._processor(images=img[..., :3], return_tensors="pt").to(self.device)
        cls = self._model(**inputs).last_hidden_state[:, 0].float()
        return torch.nn.functional.normalize(cls, dim=-1)[0].cpu().numpy()
