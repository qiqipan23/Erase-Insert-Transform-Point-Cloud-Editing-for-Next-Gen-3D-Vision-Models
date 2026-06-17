import json
from pathlib import Path

full_inp = Path("dataset/hypo3d.json")
out = Path("dataset/hypo3d_scannet_only.json")

scannet_processed = Path.home() / "datasets" / "scannet" / "processed"
have = {p.name for p in scannet_processed.iterdir() if p.is_dir()}

with open(full_inp, "r", encoding="utf-8") as f:
    data = json.load(f)

scannet_only = {sid: v for sid, v in data.items() if sid in have}

with open(out, "w", encoding="utf-8") as f:
    json.dump(scannet_only, f, indent=2, sort_keys=True, ensure_ascii=False)

print("kept scenes:", len(scannet_only), "out of", len(data))
print("wrote:", out.resolve())