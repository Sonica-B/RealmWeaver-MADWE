"""
Enhanced data preprocessing for MADWE project
Includes advanced augmentation, quality validation, and metadata generation
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import hashlib
from collections import defaultdict
import cv2
from datetime import datetime
import json


def to_serializable(obj):
    """Convert objects to something JSON serializable."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, defaultdict):
        return dict(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class EnhancedDataPreprocessor:
    """Advanced data preprocessing with quality checks and augmentation"""

    def __init__(
        self,
        data_dir: str = "data",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
    ):
        self.data_dir = Path(data_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Quality thresholds
        self.min_resolution = (256, 256)
        self.max_resolution = (2048, 2048)
        self.min_file_size = 1024  # 1KB minimum
        self.max_file_size = 10 * 1024 * 1024  # 10MB maximum

        # Statistics tracking
        self.stats = {
            "total_files": 0,
            "valid_files": 0,
            "rejected_files": 0,
            "rejection_reasons": defaultdict(int),
            "biome_distribution": defaultdict(int),
            "asset_type_distribution": defaultdict(int),
            "resolution_distribution": defaultdict(int),
            "file_sizes": [],
        }

    def validate_image(self, image_path: Path) -> Tuple[bool, str]:
        """Validate image quality and properties"""
        try:
            # Check file size
            file_size = image_path.stat().st_size
            if file_size < self.min_file_size:
                return False, "file_too_small"
            if file_size > self.max_file_size:
                return False, "file_too_large"

            # Open and validate image
            with Image.open(image_path) as img:
                # Check format
                if img.format not in ["PNG", "JPEG", "JPG"]:
                    return False, "invalid_format"

                # Check resolution
                width, height = img.size
                if width < self.min_resolution[0] or height < self.min_resolution[1]:
                    return False, "resolution_too_low"
                if width > self.max_resolution[0] or height > self.max_resolution[1]:
                    return False, "resolution_too_high"

                # Check color mode
                if img.mode not in ["RGB", "RGBA"]:
                    return False, "invalid_color_mode"

                # Check for corruption
                img.verify()

            # Additional OpenCV checks
            cv_img = cv2.imread(str(image_path))
            if cv_img is None:
                return False, "opencv_read_failed"

            # Check for blank/solid color images
            if len(np.unique(cv_img)) < 10:
                return False, "insufficient_detail"

            # Check for proper contrast
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            contrast = gray.std()
            if contrast < 10:
                return False, "low_contrast"

            return True, "valid"

        except Exception as e:
            return False, f"validation_error: {str(e)}"

    def compute_image_hash(self, image_path: Path) -> str:
        """Compute perceptual hash for duplicate detection"""
        img = cv2.imread(str(image_path))
        img = cv2.resize(img, (8, 8))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Compute DCT
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:8, :8]

        # Compute average
        avg = (dct_low.sum() - dct_low[0, 0]) / 63

        # Compute hash
        hash_str = ""
        for i in range(8):
            for j in range(8):
                if i == 0 and j == 0:
                    continue
                hash_str += "1" if dct_low[i, j] > avg else "0"

        return hash_str

    def analyze_texture_properties(self, image_path: Path) -> Dict[str, Any]:
        """Analyze texture-specific properties"""
        img = cv2.imread(str(image_path))
        properties = {}

        # Check if tileable (seamless)
        height, width = img.shape[:2]
        edge_size = 10

        # Compare edges for seamless tiling
        left_edge = img[:, :edge_size]
        right_edge = img[:, -edge_size:]
        top_edge = img[:edge_size, :]
        bottom_edge = img[-edge_size:, :]

        h_diff = np.mean(np.abs(left_edge - right_edge))
        v_diff = np.mean(np.abs(top_edge - bottom_edge))

        properties["seamless_score"] = 1.0 - (h_diff + v_diff) / 510.0
        properties["is_seamless"] = properties["seamless_score"] > 0.9

        # Analyze pattern complexity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Frequency analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        # High frequency content indicates detail
        center = magnitude_spectrum.shape[0] // 2
        high_freq = magnitude_spectrum[
            center - 10 : center + 10, center - 10 : center + 10
        ]
        properties["detail_level"] = np.mean(high_freq)

        # Pattern regularity (autocorrelation)
        autocorr = cv2.matchTemplate(gray, gray, cv2.TM_CCORR_NORMED)
        properties["pattern_regularity"] = np.std(autocorr)

        return properties

    def augment_dataset(
        self, image_path: Path, output_dir: Path, num_augmentations: int = 3
    ) -> List[Path]:
        """Generate augmented versions of images"""
        augmented_paths = []
        img = Image.open(image_path)
        base_name = image_path.stem

        augmentation_configs = [
            {"name": "rot90", "transform": lambda x: x.rotate(90)},
            {"name": "rot180", "transform": lambda x: x.rotate(180)},
            {"name": "rot270", "transform": lambda x: x.rotate(270)},
            {
                "name": "flip_h",
                "transform": lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
            },
            {
                "name": "flip_v",
                "transform": lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
            },
            {
                "name": "bright_up",
                "transform": lambda x: Image.eval(x, lambda p: min(255, int(p * 1.2))),
            },
            {
                "name": "bright_down",
                "transform": lambda x: Image.eval(x, lambda p: int(p * 0.8)),
            },
        ]

        # Select random augmentations
        selected = np.random.choice(
            augmentation_configs, num_augmentations, replace=False
        )

        for aug_config in selected:
            aug_img = aug_config["transform"](img)
            aug_name = f"{base_name}_{aug_config['name']}.png"
            aug_path = output_dir / aug_name
            aug_img.save(aug_path)
            augmented_paths.append(aug_path)

        return augmented_paths

    def create_data_splits(self) -> Dict[str, Any]:
        """Create train/val/test splits with enhanced metadata"""
        raw_dir = self.data_dir / "raw"
        processed_dir = self.data_dir / "processed"

        # Collect all valid images with metadata
        all_files = []
        duplicate_hashes = set()
        seen_hashes = {}

        print("Scanning and validating images...")
        for asset_type in ["textures", "sprites"]:
            asset_dir = raw_dir / asset_type
            if not asset_dir.exists():
                continue

            for biome_dir in asset_dir.iterdir():
                if not biome_dir.is_dir():
                    continue

                biome = biome_dir.name

                for img_path in tqdm(
                    biome_dir.glob("*.png"), desc=f"{asset_type}/{biome}"
                ):
                    self.stats["total_files"] += 1

                    # Validate image
                    is_valid, reason = self.validate_image(img_path)
                    if not is_valid:
                        self.stats["rejected_files"] += 1
                        self.stats["rejection_reasons"][reason] += 1
                        continue

                    # Check for duplicates
                    img_hash = self.compute_image_hash(img_path)
                    if img_hash in seen_hashes:
                        duplicate_hashes.add(img_hash)
                        self.stats["rejection_reasons"]["duplicate"] += 1
                        continue
                    seen_hashes[img_hash] = img_path

                    # Analyze properties
                    properties = {}
                    if asset_type == "textures":
                        properties = self.analyze_texture_properties(img_path)

                    # Get file stats
                    file_stats = img_path.stat()
                    with Image.open(img_path) as img:
                        width, height = img.size
                        mode = img.mode

                    file_info = {
                        "path": img_path,
                        "asset_type": asset_type,
                        "biome": biome,
                        "filename": img_path.name,
                        "file_size": file_stats.st_size,
                        "width": width,
                        "height": height,
                        "mode": mode,
                        "hash": img_hash,
                        "properties": properties,
                    }

                    all_files.append(file_info)
                    self.stats["valid_files"] += 1
                    self.stats["biome_distribution"][biome] += 1
                    self.stats["asset_type_distribution"][asset_type] += 1
                    self.stats["resolution_distribution"][f"{width}x{height}"] += 1
                    self.stats["file_sizes"].append(file_stats.st_size)

        print(
            f"\nValidation complete: {self.stats['valid_files']}/{self.stats['total_files']} files valid"
        )

        # Stratified split by biome and asset type
        print("\nCreating stratified splits...")
        train_files = []
        val_files = []
        test_files = []

        # Group files by stratification key
        stratified_groups = defaultdict(list)
        for file_info in all_files:
            key = f"{file_info['asset_type']}_{file_info['biome']}"
            stratified_groups[key].append(file_info)

        # Split each group
        for key, group_files in stratified_groups.items():
            if len(group_files) < 3:
                # Too few samples, put all in training
                train_files.extend(group_files)
                continue

            # First split: separate test set
            temp_files, test_group = train_test_split(
                group_files, test_size=self.test_ratio, random_state=42
            )

            # Second split: separate train and val
            val_size = self.val_ratio / (self.train_ratio + self.val_ratio)
            train_group, val_group = train_test_split(
                temp_files, test_size=val_size, random_state=42
            )

            train_files.extend(train_group)
            val_files.extend(val_group)
            test_files.extend(test_group)

        # Create directories and copy files
        splits = {"train": train_files, "val": val_files, "test": test_files}

        metadata = {
            "created_at": datetime.now().isoformat(),
            "stats": dict(self.stats),
            "splits": {},
        }

        for split_name, files in splits.items():
            print(f"\nProcessing {split_name} split ({len(files)} files)...")
            split_dir = processed_dir / split_name
            split_metadata = []

            for file_info in tqdm(files, desc=f"Copying {split_name}"):
                # Create directory structure
                dest_dir = split_dir / file_info["asset_type"] / file_info["biome"]
                dest_dir.mkdir(parents=True, exist_ok=True)

                # Copy file
                src_path = file_info["path"]
                dest_path = dest_dir / file_info["filename"]
                shutil.copy2(src_path, dest_path)

                # Add metadata
                meta_entry = {
                    "filename": file_info["filename"],
                    "path": str(dest_path.relative_to(processed_dir)),
                    "asset_type": file_info["asset_type"],
                    "biome": file_info["biome"],
                    "resolution": [file_info["width"], file_info["height"]],
                    "file_size": file_info["file_size"],
                    "color_mode": file_info["mode"],
                    "properties": file_info["properties"],
                }
                split_metadata.append(meta_entry)

                # Generate augmentations for training set
                if split_name == "train" and file_info["asset_type"] == "textures":
                    aug_paths = self.augment_dataset(
                        src_path, dest_dir, num_augmentations=2
                    )
                    for aug_path in aug_paths:
                        aug_meta = meta_entry.copy()
                        aug_meta["filename"] = aug_path.name
                        aug_meta["path"] = str(aug_path.relative_to(processed_dir))
                        aug_meta["is_augmented"] = True
                        split_metadata.append(aug_meta)

            metadata["splits"][split_name] = {
                "num_files": len(split_metadata),
                "files": split_metadata,
            }

        # Save comprehensive metadata
        metadata_path = processed_dir / "metadata" / "preprocessing_metadata.json"
        metadata_path.parent.mkdir(exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=to_serializable)

        # Save split lists for easy loading
        for split_name in ["train", "val", "test"]:
            split_list_path = processed_dir / f"{split_name}_list.txt"
            with open(split_list_path, "w") as f:
                for file_meta in metadata["splits"][split_name]["files"]:
                    f.write(f"{file_meta['path']}\n")

        # Generate quality report
        self.generate_quality_report(processed_dir / "metadata" / "quality_report.txt")

        return metadata

    def generate_quality_report(self, output_path: Path):
        """Generate detailed quality analysis report"""
        with open(output_path, "w") as f:
            f.write("MADWE Data Quality Report\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total files scanned: {self.stats['total_files']}\n")
            f.write(f"Valid files: {self.stats['valid_files']}\n")
            f.write(f"Rejected files: {self.stats['rejected_files']}\n\n")

            f.write("Rejection Reasons:\n")
            for reason, count in sorted(
                self.stats["rejection_reasons"].items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                f.write(f"  - {reason}: {count}\n")

            f.write("\nBiome Distribution:\n")
            for biome, count in sorted(self.stats["biome_distribution"].items()):
                f.write(f"  - {biome}: {count}\n")

            f.write("\nAsset Type Distribution:\n")
            for asset_type, count in sorted(
                self.stats["asset_type_distribution"].items()
            ):
                f.write(f"  - {asset_type}: {count}\n")

            f.write("\nResolution Distribution:\n")
            for res, count in sorted(
                self.stats["resolution_distribution"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]:
                f.write(f"  - {res}: {count}\n")

            if self.stats["file_sizes"]:
                f.write("\nFile Size Statistics:\n")
                sizes = np.array(self.stats["file_sizes"])
                f.write(f"  - Mean: {sizes.mean() / 1024:.1f} KB\n")
                f.write(f"  - Median: {np.median(sizes) / 1024:.1f} KB\n")
                f.write(f"  - Min: {sizes.min() / 1024:.1f} KB\n")
                f.write(f"  - Max: {sizes.max() / 1024:.1f} KB\n")


def main():
    """Main preprocessing function"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced data preprocessing for MADWE"
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")

    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError(f"Train/val/test ratios must sum to 1.0, got {total_ratio}")

    preprocessor = EnhancedDataPreprocessor(
        data_dir=args.data_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    metadata = preprocessor.create_data_splits()

    print("\nPreprocessing complete!")
    print(f"Train: {metadata['splits']['train']['num_files']} files")
    print(f"Val: {metadata['splits']['val']['num_files']} files")
    print(f"Test: {metadata['splits']['test']['num_files']} files")
    print(f"\nMetadata saved to: {args.data_dir}/processed/metadata/")


if __name__ == "__main__":
    main()
