#!/usr/bin/env python3
"""
GPT-4o batch evaluation for MOVE pipeline outputs.

Reads:  dataset/contextvqa_move_eval.json
Writes: dataset/contextvqa_move_eval_gpt4o.json  (with predicted_answer per QA)

Usage:
  export OPENAI_API_KEY=sk-...
  cd /rds/general/user/qp23/home/Hypo3D
  source miniconda3/etc/profile.d/conda.sh && conda activate placeit3d

  # One-shot (submit + wait + save):
  python 2D-VLM/GPT4o/evaluate_move.py --run

  # Or two steps:
  python 2D-VLM/GPT4o/evaluate_move.py --submit   # prints batch_id, exits
  python 2D-VLM/GPT4o/evaluate_move.py --fetch     # polls saved batch IDs, writes output
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import pandas as pd
from openai import OpenAI
from PIL import Image

HYPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_JSON  = HYPO_ROOT / "dataset" / "contextvqa_move_eval_dense.json"
REQUESTS_DIR   = HYPO_ROOT / "dataset" / "move_eval_gpt4o_requests_beforeafter"

SYSTEM = "You are an AI assistant for 3D scene understanding. Always give a definitive answer — never say 'unknown', 'cannot determine', or similar. Answer with a single word or short phrase."
PROMPT = """You are given two top-down (bird's-eye) views of the same 3D indoor scene: the FIRST image is BEFORE the change, and the SECOND image is AFTER the change.

Scene orientation: {orientation}

What changed: {context_change}
Specifically: the {target_label} was moved to be {relation_phrase} the {anchor_label}.

Answer the following question about the AFTER scene (second image), using the BEFORE image (first image) for comparison where the question refers to how things changed:
{question}

You must give a definitive answer — do not say unknown or that you cannot tell. Give a single word or short phrase. The answer is:"""

MODEL = "gpt-4o"
REQUESTS_PER_FILE = 25   # two images per request -> halve batch size
AXIS_DEF = HYPO_ROOT / "dataset" / "axis_definition.xlsx"

RELATION_TO_PHRASE = {
    "RIGHT_OF": "to the right of",
    "LEFT_OF": "to the left of",
    "FRONT_OF": "in front of",
    "BACK_OF": "behind",
    "NEXT_TO": "next to",
    "ON_TOP_OF": "on top of",
    "UNDER": "under",
}

OUT_JSON   = HYPO_ROOT / "dataset" / "contextvqa_move_eval_gpt4o_beforeafter.json"
BATCH_IDS_FILE = HYPO_ROOT / "dataset" / "move_eval_gpt4o_batch_ids_beforeafter.json"


def load_orientation_map() -> dict[str, str]:
    """Return {scene_id: orientation_sentence} from axis_definition.xlsx."""
    df = pd.read_excel(AXIS_DEF, engine="openpyxl")
    orient: dict[str, str] = {}
    for _, row in df.iterrows():
        scene_id = str(row["scene_id"])
        parts = []
        for direction in ["Front", "Back", "Left", "Right"]:
            val = row.get(direction)
            if pd.notna(val) and str(val).strip():
                parts.append(f"The {val} is at the {direction.lower()} of the scene.")
        orient[scene_id] = " ".join(parts) if parts else ""
    return orient


def encode_image(path: Path, size: int = 512) -> str:
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def custom_id(scene_id: str, job_stem: str, qa_idx: int) -> str:
    return f"{scene_id}|{job_stem}|{qa_idx}"


def build_jsonl_files(data: dict) -> list[Path]:
    """Write JSONL batch request files; return list of file paths."""
    REQUESTS_DIR.mkdir(parents=True, exist_ok=True)
    # Clear old files
    for f in REQUESTS_DIR.glob("*.jsonl"):
        f.unlink()

    orientation_map = load_orientation_map()

    lines: list[str] = []
    for scene_id, entries in data.items():
        orientation = orientation_map.get(scene_id, "")
        for entry in entries:
            job_stem = entry["job_stem"]
            img_path = HYPO_ROOT / entry["image_path"]
            context_change = entry["context_change"]
            target_label = entry.get("target_label", "object")
            anchor_label = entry.get("anchor_label", "another object")
            relation_phrase = RELATION_TO_PHRASE.get(entry.get("relation", ""), "near")
            encoded = encode_image(img_path)
            orig_img = HYPO_ROOT / "dataset" / "2D_VLM_data" / "orig_scene_top_view_dense" / f"{scene_id}.png"
            encoded_orig = encode_image(orig_img)

            for qa_idx, qa in enumerate(entry["questions_answers"]):
                cid = custom_id(scene_id, job_stem, qa_idx)
                text = PROMPT.format(
                    orientation=orientation,
                    context_change=context_change,
                    target_label=target_label,
                    anchor_label=anchor_label,
                    relation_phrase=relation_phrase,
                    question=qa["question"],
                )
                request = {
                    "custom_id": cid,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": MODEL,
                        "max_tokens": 10,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": [
                                {"type": "text", "text": text},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_orig}",
                                    "detail": "low",
                                }},
                                {"type": "image_url", "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded}",
                                    "detail": "low",
                                }},
                            ]},
                        ],
                    },
                }
                lines.append(json.dumps(request))

    print(f"Total QA pairs: {len(lines)}")
    files = []
    for i in range(0, len(lines), REQUESTS_PER_FILE):
        chunk = lines[i : i + REQUESTS_PER_FILE]
        fpath = REQUESTS_DIR / f"batch_{i // REQUESTS_PER_FILE:03d}.jsonl"
        fpath.write_text("\n".join(chunk))
        files.append(fpath)
        print(f"  Wrote {fpath.name} ({len(chunk)} requests)")
    return files


def wait_for_batch(client: OpenAI, batch_id: str) -> dict[str, str]:
    """Poll a single batch until complete. Return {custom_id: answer}."""
    answers: dict[str, str] = {}
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        counts = batch.request_counts
        print(f"  {batch_id}: {status}  (total={counts.total} done={counts.completed} failed={counts.failed})", flush=True)
        if status == "completed":
            content = client.files.content(batch.output_file_id).read().decode("utf-8")
            for line in content.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = obj["custom_id"]
                try:
                    text = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
                except (KeyError, IndexError):
                    text = ""
                answers[cid] = text
            return answers
        elif status in ("failed", "expired", "cancelled"):
            print(f"  [warn] Batch {batch_id} ended with status={status}", flush=True)
            return answers
        else:
            print(f"  Waiting 30s...", flush=True)
            time.sleep(30)


def submit_batches(client: OpenAI, jsonl_files: list[Path]) -> list[str]:
    """Submit batches one at a time, waiting for each to finish before submitting the next.
    Returns list of completed batch IDs."""
    batch_ids = []
    for fpath in jsonl_files:
        uploaded = client.files.create(file=fpath.open("rb"), purpose="batch")
        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_ids.append(batch.id)
        print(f"  Submitted {fpath.name} → batch {batch.id}")
        # Save after each submission so IDs aren't lost on crash
        BATCH_IDS_FILE.write_text(json.dumps(batch_ids, indent=2))
        print(f"  Waiting for batch {batch.id} to complete before submitting next...")
        wait_for_batch(client, batch.id)  # block until done before next submit
    print(f"Batch IDs saved to {BATCH_IDS_FILE}")
    return batch_ids


def fetch_results(client: OpenAI, batch_ids: list[str]) -> dict[str, str]:
    """Fetch results for already-submitted batch IDs (parallel polling)."""
    answers: dict[str, str] = {}
    pending = list(batch_ids)

    while pending:
        still_pending = []
        for bid in pending:
            batch = client.batches.retrieve(bid)
            status = batch.status
            counts = batch.request_counts
            print(f"  {bid}: {status}  (total={counts.total} done={counts.completed} failed={counts.failed})")
            if status == "completed":
                content = client.files.content(batch.output_file_id).read().decode("utf-8")
                for line in content.splitlines():
                    if not line.strip():
                        continue
                    obj = json.loads(line)
                    cid = obj["custom_id"]
                    try:
                        text = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
                    except (KeyError, IndexError):
                        text = ""
                    answers[cid] = text
            elif status in ("failed", "expired", "cancelled"):
                print(f"  [warn] Batch {bid} ended with status={status}")
            else:
                still_pending.append(bid)

        pending = still_pending
        if pending:
            print(f"Waiting 30s for {len(pending)} batches...")
            time.sleep(30)

    return answers


def _is_token_limit_error(batch) -> bool:
    return (
        batch.errors
        and batch.errors.data
        and any(e.code == "token_limit_exceeded" for e in batch.errors.data)
    )


def submit_and_collect(client: OpenAI, jsonl_files: list[Path]) -> dict[str, str]:
    """Submit + collect results one batch at a time to stay under token enqueue limits.
    Retries with backoff if token_limit_exceeded (queue not yet clear from prior batch)."""
    all_answers: dict[str, str] = {}
    batch_ids: list[str] = []

    for i, fpath in enumerate(jsonl_files, 1):
        print(f"[{i}/{len(jsonl_files)}] Uploading {fpath.name}...", flush=True)
        uploaded = client.files.create(file=fpath.open("rb"), purpose="batch")

        # Retry loop for token_limit_exceeded (previous batch still clearing from queue)
        for attempt in range(1, 11):
            batch = client.batches.create(
                input_file_id=uploaded.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )
            # Give OpenAI a moment to validate and surface errors
            time.sleep(5)
            batch = client.batches.retrieve(batch.id)
            if batch.status == "failed" and _is_token_limit_error(batch):
                wait = 60 * attempt
                print(f"  token_limit_exceeded (attempt {attempt}), waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            break

        batch_ids.append(batch.id)
        BATCH_IDS_FILE.write_text(json.dumps(batch_ids, indent=2))
        print(f"  → batch {batch.id} submitted; waiting for completion...", flush=True)
        answers = wait_for_batch(client, batch.id)
        all_answers.update(answers)
        print(f"  Got {len(answers)} answers (running total: {len(all_answers)})", flush=True)

    return all_answers


def merge_answers(data: dict, answers: dict[str, str]) -> dict:
    missing = 0
    for scene_id, entries in data.items():
        for entry in entries:
            job_stem = entry["job_stem"]
            for qa_idx, qa in enumerate(entry["questions_answers"]):
                cid = custom_id(scene_id, job_stem, qa_idx)
                if cid in answers:
                    qa["predicted_answer"] = answers[cid]
                else:
                    missing += 1
    if missing:
        print(f"[warn] {missing} QA pairs have no predicted answer")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--submit", action="store_true", help="Build JSONL + submit batches, save IDs")
    grp.add_argument("--fetch",  action="store_true", help="Fetch results from saved batch IDs")
    grp.add_argument("--run",    action="store_true", help="Submit then poll until done (one-shot)")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N QA pairs (for testing)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY environment variable.")

    client = OpenAI(api_key=api_key)
    data = json.loads(EVAL_JSON.read_text())

    if args.limit:
        # Trim data to first --limit QA pairs across scenes
        trimmed: dict = {}
        count = 0
        for scene_id, entries in data.items():
            if count >= args.limit:
                break
            trimmed_entries = []
            for entry in entries:
                if count >= args.limit:
                    break
                qa_needed = min(len(entry["questions_answers"]), args.limit - count)
                e = dict(entry)
                e["questions_answers"] = entry["questions_answers"][:qa_needed]
                trimmed_entries.append(e)
                count += qa_needed
            trimmed[scene_id] = trimmed_entries
        data = trimmed
        print(f"[limit] Using {count} QA pairs from {len(data)} scene(s)")

    if args.run:
        jsonl_files = build_jsonl_files(data)
        answers = submit_and_collect(client, jsonl_files)
        print(f"Got {len(answers)} answers total.")
        data = merge_answers(data, answers)
        OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Written: {OUT_JSON}")
        return

    if args.submit:
        jsonl_files = build_jsonl_files(data)
        batch_ids = submit_batches(client, jsonl_files)
        print("Done. Run with --fetch when batches complete.")
        return

    if args.fetch:
        if not BATCH_IDS_FILE.exists():
            raise SystemExit(f"No batch IDs at {BATCH_IDS_FILE}. Run --submit first.")
        batch_ids = json.loads(BATCH_IDS_FILE.read_text())
        print(f"Fetching results from {len(batch_ids)} batch(es)...")
        answers = fetch_results(client, batch_ids)
        print(f"Got {len(answers)} answers.")
        data = merge_answers(data, answers)
        OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Written: {OUT_JSON}")


if __name__ == "__main__":
    main()
