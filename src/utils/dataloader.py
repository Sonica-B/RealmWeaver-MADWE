"""
Data loading utilities for MADWE project 
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm


class GameAssetDataset(Dataset):
    """Dataset for game assets with Python 3.13 improvements"""

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        asset_types: List[str] | None = None,
        biomes: List[str] | None = None,
        transform: Optional[transforms.Compose] = None,
    ):

        self.data_dir = Path(data_dir)
        self.split = split
        self.asset_types = asset_types or ["textures", "sprites"]
        self.biomes = biomes or [
            "forest",
            "desert",
            "snow",
            "volcanic",
            "underwater",
            "sky",
        ]
        self.transform = transform or self._default_transform()

        self.samples: List[Dict[str, str]] = []
        self._load_samples()

    def _default_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(
                    (512, 512), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _load_samples(self) -> None:
        """Load all image paths"""
        for asset_type in self.asset_types:
            for biome in self.biomes:
                asset_path = self.data_dir / self.split / asset_type / biome
                if asset_path.exists():
                    for img_path in asset_path.glob("*.png"):
                        self.samples.append(
                            {
                                "path": str(img_path),
                                "asset_type": asset_type,
                                "biome": biome,
                            }
                        )

        print(f"Loaded {len(self.samples)} samples for {self.split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "asset_type": sample["asset_type"],
            "biome": sample["biome"],
            "path": sample["path"],
        }


class BiomeDataset(Dataset):
    """Dataset for biome-specific training with Python 3.13 enhancements"""

    def __init__(
        self,
        data_dir: Path | str,
        biome: str,
        tokenizer,
        size: int = 512,
        random_flip: bool = True,
    ):

        self.data_dir = Path(data_dir)
        self.biome = biome
        self.tokenizer = tokenizer
        self.size = size
        self.random_flip = random_flip

        self.images: List[Path] = []
        self._load_images()
        self.prompts = self._get_biome_prompts()

    def _load_images(self) -> None:
        """Load images for biome"""
        for asset_type in ["textures", "sprites"]:
            biome_path = self.data_dir / asset_type / self.biome
            if biome_path.exists():
                self.images.extend(list(biome_path.glob("*.png")))

        print(f"Loaded {len(self.images)} images for {self.biome} biome")

    def _get_biome_prompts(self) -> List[str]:
        """Get biome-specific prompts"""
        prompts_map = {
            "forest": [
                "lush forest texture",
                "green woodland asset",
                "mystical forest tile",
            ],
            "desert": [
                "sandy desert texture",
                "arid wasteland asset",
                "golden dunes tile",
            ],
            "snow": ["frozen tundra texture", "icy winter asset", "snow-covered tile"],
            "volcanic": [
                "molten lava texture",
                "volcanic ash asset",
                "fiery magma tile",
            ],
            "underwater": ["deep ocean texture", "coral reef asset", "aquatic tile"],
            "sky": ["cloudy sky texture", "celestial asset", "heavenly tile"],
        }
        return prompts_map.get(self.biome, ["game texture"])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        # Load image
        image = Image.open(self.images[idx]).convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)

        # Random flip
        if self.random_flip and np.random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        # Convert to tensor
        image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        # Get prompt
        prompt = np.random.choice(self.prompts)
        prompt = f"{prompt}, high quality, detailed, game ready"

        # Tokenize
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "pixel_values": image_tensor,
            "input_ids": text_inputs.input_ids.squeeze(),
            "prompt": prompt,
        }


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    pin_memory: bool | None = None,
) -> Dict[str, DataLoader]:
    """Create dataloaders for all splits with Python 3.13 improvements"""

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    dataloaders = {}

    for split in ["train", "val", "test"]:
        dataset = GameAssetDataset(data_dir=Path(data_dir), split=split)

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,  # 0 for Windows compatibility
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

    return dataloaders
