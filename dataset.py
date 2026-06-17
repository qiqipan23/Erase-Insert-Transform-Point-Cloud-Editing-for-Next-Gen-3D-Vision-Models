# dataset.py
import json
from datasets import Dataset

def load_json(json_file_path: str):
    with open(json_file_path, "r") as json_file:
        return json.load(json_file)

def save_json(data, json_file_path: str):
    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

if __name__ == "__main__":
    # Path on CX3 (repo root is ~/Hypo3D, file is in ~/Hypo3D/dataset/hypo3d.json)
    raw = load_json("dataset/hypo3d.json")   # dict: scene_id -> list[change]

    # Flatten: one row per context change (best for pairing with PlaceIt3D)
    rows = []
    for scene_id, changes in raw.items():
        for change_idx, change in enumerate(changes):
            rows.append({
                "scene_id": scene_id,
                "change_idx": change_idx,
                "context_change": change.get("context_change", ""),
                "change_type": change.get("change_type", ""),
                "questions_answers": change.get("questions_answers", []),  # list[dict]
            })

    dataset = Dataset.from_list(rows)
    print(dataset)
    print("Example row:", dataset[0])

    # OPTIONAL: push to your own HF repo (you must be logged in on the cluster)
    # dataset.push_to_hub("YOUR_HF_USERNAME/Hypo3D_changes")

