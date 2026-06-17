#!/usr/bin/env python3
"""
Merge the existing hidden surface dataset with the new wall-crop examples
into a single combined dataset directory using symlinks.

Usage:
  python merge_hidden_surface_datasets.py
"""
import json
import os
from pathlib import Path
import numpy as np

ORIGINAL = Path("repo/scannet/completion/hidden_surface_scannet_train")
WALLCROP  = Path("repo/scannet/completion/hidden_surface_wall_crop")
COMBINED  = Path("repo/scannet/completion/hidden_surface_combined")
CATEGORY  = "scannet_remove_hidden_surface"


def read_lst(path: Path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def write_lst(path: Path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(ids) + "\n")


def main():
    cat_out = COMBINED / CATEGORY
    cat_out.mkdir(parents=True, exist_ok=True)

    # Symlink all examples from both datasets
    n_orig, n_wall = 0, 0
    for src_root in [ORIGINAL, WALLCROP]:
        cat_src = src_root / CATEGORY
        for ex_dir in sorted(cat_src.iterdir()):
            if not ex_dir.is_dir():
                continue
            link = cat_out / ex_dir.name
            if not link.exists():
                os.symlink(ex_dir.resolve(), link)
            n_orig += (src_root == ORIGINAL)
            n_wall += (src_root == WALLCROP)

    print(f"Linked {n_orig} original + {n_wall} wall-crop = {n_orig+n_wall} total examples")

    # Merge split lists
    rng = np.random.default_rng(99)
    for split in ("train", "val", "test"):
        orig_ids = read_lst(ORIGINAL / CATEGORY / f"{split}.lst")
        wall_ids = read_lst(WALLCROP  / CATEGORY / f"{split}.lst")
        merged   = orig_ids + wall_ids
        rng.shuffle(merged)
        write_lst(cat_out / f"{split}.lst", merged)
        print(f"  {split}: {len(orig_ids)} orig + {len(wall_ids)} wall = {len(merged)}")

    # Write metadata
    (COMBINED / "metadata.yaml").write_text(
        f"scannet_remove_hidden_surface:\n  id: {CATEGORY}\n  name: ScanNet Hidden Surface Combined\n")

    # Write fine-tune config
    cfg = {
        "dataset_root": str(COMBINED.resolve()),
        "category": CATEGORY,
        "input_pointcloud_n": 100000,
        "target_point_n": 2048,
        "input_feature_dim": 2,
        "batch_size": 8,
        "lr": 2e-5,
        "epochs": 100,
        "print_every": 25,
        "save_every": 5,
        "out_dir": str(Path("repo/scannet/completion/hidden_surface_finetune_out").resolve()),
        "predict_color": False,
        "color_weight": 0.5,
        "model": {
            "hidden_dim": 256,
            "latent_dim": 512,
            "output_points": 2048
        },
    }
    cfg_path = Path("repo/scannet/completion/hidden_surface_finetune_config.json")
    cfg_path.write_text(json.dumps(cfg, indent=2))
    print(f"\nConfig written: {cfg_path}")
    print(f"Combined dataset: {COMBINED}")


if __name__ == "__main__":
    main()
