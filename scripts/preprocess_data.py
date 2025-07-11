"""
Preprocess raw assets into train/val/test splits - Python 3.11 compatible
Works with dynamic directory structure
"""

import shutil
from pathlib import Path
import random
from tqdm import tqdm
from typing import Dict, List
import json
from datetime import datetime
from collections import defaultdict


def split_dynamic_data(
    source_dir: Path | str,
    dest_dir: Path | str,
    splits: Dict[str, float] | None = None,
    seed: int = 42,
) -> Dict[str, int]:
    """Split dynamically structured data into train/val/test"""

    if splits is None:
        splits = {"train": 0.8, "val": 0.1, "test": 0.1}

    # Validate splits
    total = sum(splits.values())
    if abs(total - 1.0) > 0.001:
        raise ValueError(f"Splits must sum to 1.0, got {total}")

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    random.seed(seed)

    split_counts = {split: 0 for split in splits.keys()}
    split_metadata = {
        "created_at": datetime.now().isoformat(),
        "seed": seed,
        "splits": splits,
        "structure": {},
        "files": {split: [] for split in splits.keys()},
    }

    # Discover asset types
    asset_types = [d.name for d in source_dir.iterdir() if d.is_dir()]

    for asset_type in asset_types:
        asset_path = source_dir / asset_type
        if not asset_path.exists():
            continue

        # Process each category directory
        for category_dir in asset_path.iterdir():
            if not category_dir.is_dir():
                continue

            images = list(category_dir.glob("*.png"))
            if not images:
                continue

            random.shuffle(images)

            n_total = len(images)
            n_train = int(splits["train"] * n_total)
            n_val = int(splits["val"] * n_total)

            # Create split directories maintaining structure
            for split in splits.keys():
                split_path = dest_dir / split / asset_type / category_dir.name
                split_path.mkdir(parents=True, exist_ok=True)

            # Copy files to splits
            for i, img_path in enumerate(
                tqdm(images, desc=f"{asset_type}/{category_dir.name}", leave=False)
            ):
                if i < n_train:
                    split = "train"
                elif i < n_train + n_val:
                    split = "val"
                else:
                    split = "test"

                dest_path = (
                    dest_dir / split / asset_type / category_dir.name / img_path.name
                )
                shutil.copy2(img_path, dest_path)

                split_counts[split] += 1
                split_metadata["files"][split].append(
                    {
                        "path": str(dest_path.relative_to(dest_dir)),
                        "asset_type": asset_type,
                        "category": category_dir.name,
                        "original": str(img_path.relative_to(source_dir.parent)),
                    }
                )

    # Store discovered structure
    split_metadata["structure"] = discover_structure(source_dir)

    # Save split metadata
    metadata_dir = dest_dir.parent / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    metadata_path = metadata_dir / "split_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(split_metadata, f, indent=2)

    print(f"\nSplit metadata saved to {metadata_path}")

    return split_counts


def discover_structure(data_dir: Path) -> Dict[str, List[str]]:
    """Discover the data structure"""
    structure = defaultdict(list)

    for asset_type_dir in data_dir.iterdir():
        if asset_type_dir.is_dir():
            categories = [d.name for d in asset_type_dir.iterdir() if d.is_dir()]
            structure[asset_type_dir.name] = categories

    return dict(structure)


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
        asset_counts = defaultdict(lambda: defaultdict(int))

        for asset_type_dir in split_dir.iterdir():
            if not asset_type_dir.is_dir():
                continue

            for category_dir in asset_type_dir.iterdir():
                if category_dir.is_dir():
                    count = len(list(category_dir.glob("*.png")))
                    total_files += count
                    asset_counts[asset_type_dir.name][category_dir.name] = count

        print(f"\n{split.upper()} split:")
        print(f"  Total files: {total_files}")
        print("  Distribution:")
        for asset_type, categories in sorted(asset_counts.items()):
            print(f"    {asset_type}:")
            for category, count in sorted(categories.items()):
                print(f"      {category}: {count}")


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
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--verify", action="store_true", help="Verify splits after creation"
    )

    args = parser.parse_args()

    # Create splits dictionary
    splits = {"train": args.train_split, "val": args.val_split, "test": args.test_split}

    # Perform splitting
    print(f"Splitting data from {args.source_dir} to {args.dest_dir}")
    print(f"Splits: {splits}")

    split_counts = split_dynamic_data(
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


if __name__ == "__main__":
    main()
