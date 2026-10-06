"""Style vectors: DINOv2 CLS embeddings, L2-normalised (384-d for `facebook/dinov2-small`)."""

from __future__ import annotations

import numpy as np
import torch

from realmweaver.config import settings


class DinoEmbedder:
    """Callable `image -> style vector`. Runs on `settings().device` unless `device` is given; CPU works too."""

    def __init__(self, model_id: str = "facebook/dinov2-small", device: str | None = None) -> None:
        from transformers import AutoImageProcessor, AutoModel

        self.device = torch.device(device or settings().device)
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = int(self._model.config.hidden_size)

    @torch.inference_mode()
    def __call__(self, img: np.ndarray) -> np.ndarray:
        """`img` is uint8 HxWx3 or HxWx4 (alpha dropped). Returns float32 `[dim]` with unit L2 norm."""
        inputs = self._processor(images=img[..., :3], return_tensors="pt").to(self.device)
        cls = self._model(**inputs).last_hidden_state[:, 0].float()
        return torch.nn.functional.normalize(cls, dim=-1)[0].cpu().numpy()
