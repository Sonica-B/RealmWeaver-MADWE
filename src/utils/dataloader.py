"""
Data loading utilities for MADWE project - fully dynamic
Handles directory structure: raw/[asset_type]/[categories]/
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from collections import defaultdict


class DynamicGameAssetDataset(Dataset):
    """Fully dynamic dataset that discovers structure from directory"""

    def __init__(
        self,
        data_dir: Path | str,
        split: str = "train",
        asset_types: List[str] | None = None,
        filters: Dict[str, Any] | None = None,
        transform: Optional[transforms.Compose] = None,
        load_metadata: bool = True,
        discover_structure: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or self._default_transform()
        self.filters = filters or {}

        # Discover asset types if not provided
        if asset_types is None and discover_structure:
            asset_types = self._discover_asset_types()
        self.asset_types = asset_types or ["textures", "sprites", "gameplay"]

        # Load metadata if available
        self.metadata = {}
        if load_metadata:
            self._load_metadata()

        # Discover directory structure
        self.structure = {}
        if discover_structure:
            self.structure = self._discover_structure()

        # Load samples
        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    def _discover_asset_types(self) -> List[str]:
        """Discover available asset types from directory"""
        base_path = self.data_dir / self.split
        if not base_path.exists():
            base_path = self.data_dir / "raw"

        asset_types = []
        if base_path.exists():
            asset_types = [d.name for d in base_path.iterdir() if d.is_dir()]

        return sorted(asset_types)

    def _discover_structure(self) -> Dict[str, Dict[str, Set[str]]]:
        """Discover the complete directory structure dynamically"""
        structure = defaultdict(lambda: defaultdict(set))

        for asset_type in self.asset_types:
            asset_path = self.data_dir / self.split / asset_type
            if not asset_path.exists():
                asset_path = self.data_dir / "raw" / asset_type

            if asset_path.exists():
                # Discover all subdirectories and their attributes
                for subdir in asset_path.iterdir():
                    if subdir.is_dir():
                        # Parse directory name for attributes
                        dir_parts = subdir.name.split("_")

                        # Group parts into key-value pairs
                        i = 0
                        while i < len(dir_parts) - 1:
                            key = dir_parts[i]
                            value = dir_parts[i + 1]
                            structure[asset_type][key].add(value)
                            i += 2

        # Convert sets to lists
        return {
            asset_type: {k: sorted(list(v)) for k, v in categories.items()}
            for asset_type, categories in structure.items()
        }

    def _load_metadata(self) -> None:
        """Load metadata from JSON file"""
        metadata_paths = [
            self.data_dir / "metadata" / "generated_assets.json",
            self.data_dir / "metadata" / "comprehensive_assets.json",
        ]

        for metadata_path in metadata_paths:
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    data = json.load(f)
                    # Create lookup by filename
                    for asset in data.get("assets", []):
                        self.metadata[asset["filename"]] = asset
                break

    def _default_transform(self) -> transforms.Compose:
        """Default transformation pipeline"""
        return transforms.Compose(
            [
                transforms.Resize(
                    (512, 512), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def _parse_directory_name(self, dir_name: str) -> Dict[str, str]:
        """Parse directory name into attributes"""
        attributes = {}
        parts = dir_name.split("_")

        # Parse key-value pairs
        i = 0
        while i < len(parts) - 1:
            key = parts[i]
            value = parts[i + 1]
            attributes[key] = value
            i += 2

        return attributes

    def _matches_filters(self, attributes: Dict[str, str]) -> bool:
        """Check if attributes match the provided filters"""
        for key, values in self.filters.items():
            if key in attributes:
                if isinstance(values, list):
                    if attributes[key] not in values:
                        return False
                elif attributes[key] != values:
                    return False
        return True

    def _load_samples(self) -> None:
        """Load all samples matching filters"""
        for asset_type in self.asset_types:
            asset_path = self.data_dir / self.split / asset_type
            if not asset_path.exists():
                asset_path = self.data_dir / "raw" / asset_type

            if not asset_path.exists():
                continue

            # Iterate through category directories
            for category_dir in asset_path.iterdir():
                if not category_dir.is_dir():
                    continue

                # Parse attributes from directory name
                attributes = self._parse_directory_name(category_dir.name)

                # Check filters
                if not self._matches_filters(attributes):
                    continue

                # Load all images in this directory
                for img_path in category_dir.glob("*.png"):
                    sample = {
                        "path": str(img_path),
                        "filename": img_path.name,
                        "asset_type": asset_type,
                        "category_dir": category_dir.name,
                        "attributes": attributes,
                    }

                    # Add metadata if available
                    if img_path.name in self.metadata:
                        sample.update(self.metadata[img_path.name])

                    self.samples.append(sample)

        print(f"Loaded {len(self.samples)} samples for {self.split}")
        self._print_statistics()

    def _print_statistics(self) -> None:
        """Print dataset statistics"""
        stats = defaultdict(lambda: defaultdict(int))

        for sample in self.samples:
            stats["asset_types"][sample["asset_type"]] += 1
            for key, value in sample.get("attributes", {}).items():
                stats[f"attribute_{key}"][value] += 1

        print("Dataset Statistics:")
        for category, counts in stats.items():
            print(f"  {category}: {dict(counts)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Prepare output
        output = {
            "image": image,
            "asset_type": sample["asset_type"],
            "path": sample["path"],
            "attributes": sample.get("attributes", {}),
        }

        # Add all dynamic attributes
        for key, value in sample.get("attributes", {}).items():
            output[key] = value

        # Add prompt if available
        if "prompt" in sample:
            output["prompt"] = sample["prompt"]

        return output


class DynamicConditionalDataset(Dataset):
    """Dataset for conditional generation with dynamic categories"""

    def __init__(
        self,
        data_dir: Path | str,
        asset_type: str,
        condition: Dict[str, str],  # e.g., {"genre": "platformer", "style": "pixel"}
        tokenizer,
        size: int = 512,
        discover_prompts: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.asset_type = asset_type
        self.condition = condition
        self.tokenizer = tokenizer
        self.size = size

        self.samples: List[Dict[str, Any]] = []
        self._load_conditional_samples()

        # Discover prompt patterns if requested
        self.prompt_templates = []
        if discover_prompts:
            self._discover_prompt_templates()

    def _load_conditional_samples(self) -> None:
        """Load samples matching conditions"""
        asset_path = self.data_dir / "raw" / self.asset_type

        if not asset_path.exists():
            print(f"Warning: {asset_path} does not exist")
            return

        # Find matching directories
        for category_dir in asset_path.iterdir():
            if not category_dir.is_dir():
                continue

            # Check if directory matches conditions
            dir_name = category_dir.name
            matches = True

            for key, value in self.condition.items():
                if f"{key}_{value}" not in dir_name:
                    matches = False
                    break

            if matches:
                # Load all images
                for img_path in category_dir.glob("*.png"):
                    self.samples.append(
                        {
                            "path": str(img_path),
                            "category": category_dir.name,
                            "asset_type": self.asset_type,
                        }
                    )

        print(
            f"Loaded {len(self.samples)} samples for {self.asset_type} with conditions {self.condition}"
        )

    def _discover_prompt_templates(self) -> None:
        """Discover common prompt patterns from metadata"""
        metadata_path = self.data_dir / "metadata" / "generated_assets.json"

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                data = json.load(f)

                # Extract prompts for this asset type
                prompts = [
                    asset["prompt"]
                    for asset in data.get("assets", [])
                    if asset.get("asset_type") == self.asset_type
                ]

                # Store unique patterns (simplified)
                self.prompt_templates = list(set(prompts[:10]))  # Sample

    def _generate_prompt(self, sample: Dict[str, Any]) -> str:
        """Generate prompt for sample"""
        if self.prompt_templates:
            # Use discovered template
            template = random.choice(self.prompt_templates)
            # Customize with current sample info
            prompt = template
        else:
            # Generate from scratch
            parts = [
                f"{self.asset_type} asset",
                f"category: {sample['category']}",
                "high quality, game ready",
                "professional game art",
            ]
            prompt = ", ".join(parts)

        return prompt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load and preprocess image
        image = Image.open(sample["path"]).convert("RGB")
        image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)

        # Convert to tensor
        image_array = np.array(image).astype(np.float32) / 127.5 - 1.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)

        # Generate prompt
        prompt = self._generate_prompt(sample)

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
            "metadata": sample,
        }


def create_dynamic_dataloaders(
    data_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    asset_types: List[str] | None = None,
    filters: Dict[str, Any] | None = None,
    discover_structure: bool = True,
) -> Dict[str, DataLoader]:
    """Create dataloaders with dynamic discovery"""

    dataloaders = {}

    for split in ["train", "val", "test"]:
        # Try both split-based and raw directory
        split_path = Path(data_dir) / split
        if not split_path.exists():
            if split == "train":
                # Use raw directory for training if processed not available
                dataset = DynamicGameAssetDataset(
                    data_dir=data_dir,
                    split="raw",
                    asset_types=asset_types,
                    filters=filters,
                    discover_structure=discover_structure,
                )
            else:
                continue
        else:
            dataset = DynamicGameAssetDataset(
                data_dir=data_dir,
                split=split,
                asset_types=asset_types,
                filters=filters,
                discover_structure=discover_structure,
            )

        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return dataloaders


def discover_dataset_structure(data_dir: str | Path) -> Dict[str, Any]:
    """Discover and return the complete dataset structure"""
    dataset = DynamicGameAssetDataset(
        data_dir=data_dir, split="raw", discover_structure=True
    )

    return {
        "asset_types": dataset.asset_types,
        "structure": dataset.structure,
        "total_samples": len(dataset.samples),
    }
