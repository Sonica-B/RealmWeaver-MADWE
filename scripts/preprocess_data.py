"""
Preprocess raw assets into train/val/test splits - Python 3.13.5 compatible
"""

import shutil
from pathlib import Path
import random
from tqdm import tqdm
from typing import Dict, List, Tuple
import json
from datetime import datetime


def split_data(
    source_dir: Path | str,
    dest_dir: Path | str,
    splits: Dict[str, float] | None = None,
    seed: int = 42,
) -> Dict[str, int]:
    """Split data into train/val/test with stratified sampling"""

    if splits is None:
        splits = {"train": 0.8, "val": 0.1, "test": 0.1}

    # Validate splits
    total = sum(splits.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Splits must sum to 1.0, got {total}")

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    asset_types = ["textures", "sprites"]
    biomes = ["forest", "desert", "snow", "volcanic", "underwater", "sky"]

    random.seed(seed)

    split_counts = {split: 0 for split in splits.keys()}
    split_metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "splits": splits,
        "files": {split: [] for split in splits.keys()},
    }

    for asset_type in asset_types:
        for biome in biomes:
            biome_path = source_dir / asset_type / biome
            if not biome_path.exists():
                print(f"Warning: {biome_path} does not exist, skipping...")
                continue

            images = list(biome_path.glob("*.png"))
            if not images:
                continue

            random.shuffle(images)

            n_total = len(images)
            n_train = int(splits["train"] * n_total)
            n_val = int(splits["val"] * n_total)

            # Create split directories
            for split in splits.keys():
                split_path = dest_dir / split / asset_type / biome
                split_path.mkdir(parents=True, exist_ok=True)

            # Copy files to splits with progress bar
            for i, img_path in enumerate(
                tqdm(images, desc=f"{asset_type}/{biome}", leave=False)
            ):
                if i < n_train:
                    split = "train"
                elif i < n_train + n_val:
                    split = "val"
                else:
                    split = "test"

                dest_path = dest_dir / split / asset_type / biome / img_path.name
                shutil.copy2(img_path, dest_path)

                split_counts[split] += 1
                split_metadata["files"][split].append(
                    {
                        "path": str(dest_path.relative_to(dest_dir)),
                        "asset_type": asset_type,
                        "biome": biome,
                        "original": str(img_path.relative_to(source_dir.parent)),
                    }
                )

    # Save split metadata
    metadata_dir = dest_dir.parent / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    metadata_path = metadata_dir / "split_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(split_metadata, f, indent=2)

    print(f"\nSplit metadata saved to {metadata_path}")

    return split_counts


def verify_splits(processed_dir: Path | str) -> None:
    """Verify the data splits are correct"""
    processed_dir = Path(processed_dir)

    print("\nVerifying data splits...")

    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        if not split_dir.exists():
            print(f"Warning: {split} directory does not exist")
            continue

        total_files = 0
        biome_counts = {}

        for asset_type in ["textures", "sprites"]:
            for biome_path in (split_dir / asset_type).glob("*"):
                if biome_path.is_dir():
                    biome = biome_path.name
                    count = len(list(biome_path.glob("*.png")))
                    total_files += count

                    key = f"{asset_type}/{biome}"
                    biome_counts[key] = count

        print(f"\n{split.upper()} split:")
        print(f"  Total files: {total_files}")
        print("  Distribution:")
        for key, count in sorted(biome_counts.items()):
            print(f"    {key}: {count}")


def create_file_lists(processed_dir: Path | str) -> None:
    """Create text files listing all images in each split"""
    processed_dir = Path(processed_dir)

    print("\nCreating file lists...")

    for split in ["train", "val", "test"]:
        split_dir = processed_dir / split
        if not split_dir.exists():
            continue

        file_list = []

        for img_path in split_dir.rglob("*.png"):
            relative_path = img_path.relative_to(split_dir)
            file_list.append(str(relative_path))

        # Save file list
        list_path = processed_dir.parent / f"{split}_files.txt"
        with open(list_path, "w") as f:
            f.write("\n".join(sorted(file_list)))

        print(f"  {split}: {len(file_list)} files -> {list_path}")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data for MADWE")
    parser.add_argument(
        "--source-dir",
        type=str,
        default="data/raw",
        help="Source directory with raw assets",
    )
    parser.add_argument(
        "--dest-dir",
        type=str,
        default="data/processed",
        help="Destination directory for processed data",
    )
    parser.add_argument(
        "--train-split", type=float, default=0.8, help="Training split ratio"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--test-split", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--verify", action="store_true", help="Verify splits after creation"
    )
    parser.add_argument(
        "--create-lists", action="store_true", help="Create file lists for each split"
    )

    args = parser.parse_args()

    # Create splits dictionary
    splits = {"train": args.train_split, "val": args.val_split, "test": args.test_split}

    # Perform splitting
    print(f"Splitting data from {args.source_dir} to {args.dest_dir}")
    print(f"Splits: {splits}")

    split_counts = split_data(
        args.source_dir, args.dest_dir, splits=splits, seed=args.seed
    )

    # Print results
    print("\nData splitting complete!")
    print("Files per split:")
    for split, count in split_counts.items():
        percentage = count / sum(split_counts.values()) * 100
        print(f"  {split}: {count} files ({percentage:.1f}%)")

    # Verify if requested
    if args.verify:
        verify_splits(args.dest_dir)

    # Create file lists if requested
    if args.create_lists:
        create_file_lists(args.dest_dir)


if __name__ == "__main__":
    main()
