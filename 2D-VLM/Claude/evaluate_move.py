#!/usr/bin/env python3
"""
VLM evaluation for MOVE pipeline outputs using Claude claude-sonnet-4-6 batch API.

Reads:  dataset/contextvqa_move_eval.json
        (produced by repo/scannet/build_move_eval_json.py)

Writes: dataset/contextvqa_move_eval_claude.json
        (same structure with 'predicted_answer' added to each QA entry)

Usage:
  ANTHROPIC_API_KEY=sk-ant-... python evaluate_move.py [--submit | --fetch | --run]

  --submit  Build and submit batch to API, print batch_id, exit
  --fetch   Poll all pending batches, merge results, write output JSON
  --run     Submit then poll until done (one-shot, for small eval sets)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
from io import BytesIO
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from PIL import Image

HYPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_JSON = HYPO_ROOT / "dataset" / "contextvqa_move_eval.json"
OUT_JSON = HYPO_ROOT / "dataset" / "contextvqa_move_eval_claude.json"
BATCH_IDS_FILE = HYPO_ROOT / "dataset" / "move_eval_batch_ids.json"

PROMPT_TEMPLATE = """You are given a top-down (bird's-eye) view image of a 3D indoor scene after an object has been moved.

Scene change: {context_change}

Answer the following question based on the image:
{question}

Give a single word or short phrase as your answer.

The answer is:"""

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 10
BATCH_SIZE = 100  # requests per batch (API limit is 10 000; keep smaller for reliability)


def encode_image(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_requests(data: dict) -> list[tuple[str, Request]]:
    """Return list of (custom_id, Request) for every QA pair."""
    requests = []
    for scene_id, entries in data.items():
        for entry in entries:
            job_stem = entry["job_stem"]
            img_path = HYPO_ROOT / entry["image_path"]
            context_change = entry["context_change"]

            encoded = encode_image(img_path)

            for qa_idx, qa in enumerate(entry["questions_answers"]):
                custom_id = f"{scene_id}|{job_stem}|{qa_idx}"
                text = PROMPT_TEMPLATE.format(
                    context_change=context_change,
                    question=qa["question"],
                )
                req = Request(
                    custom_id=custom_id,
                    params=MessageCreateParamsNonStreaming(
                        model=MODEL,
                        max_tokens=MAX_TOKENS,
                        system="You are an AI assistant for 3D scene understanding. Answer with a single word or short phrase.",
                        stop_sequences=[",", "(", "-", "."],
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": encoded,
                                        },
                                    },
                                    {"type": "text", "text": text},
                                ],
                            }
                        ],
                    ),
                )
                requests.append((custom_id, req))
    return requests


def submit_batches(client: anthropic.Anthropic, data: dict) -> list[str]:
    all_requests = build_requests(data)
    print(f"Total QA pairs: {len(all_requests)}")

    batch_ids = []
    for i in range(0, len(all_requests), BATCH_SIZE):
        chunk = all_requests[i : i + BATCH_SIZE]
        batch = client.messages.batches.create(requests=[r for _, r in chunk])
        batch_ids.append(batch.id)
        print(f"  Submitted batch {len(batch_ids)}: {batch.id} ({len(chunk)} requests)")

    BATCH_IDS_FILE.write_text(json.dumps(batch_ids, indent=2))
    print(f"Batch IDs saved to {BATCH_IDS_FILE}")
    return batch_ids


def fetch_results(client: anthropic.Anthropic, batch_ids: list[str]) -> dict[str, str]:
    """Poll until all batches complete. Return {custom_id: answer}."""
    answers: dict[str, str] = {}
    pending = list(batch_ids)

    while pending:
        still_pending = []
        for bid in pending:
            batch = client.messages.batches.retrieve(bid)
            status = batch.processing_status
            print(f"  {bid}: {status} (req_counts={batch.request_counts})")
            if status == "ended":
                for result in client.messages.batches.results(bid):
                    if result.result.type == "succeeded":
                        text = result.result.message.content[0].text.strip()
                        answers[result.custom_id] = text
                    else:
                        print(f"    [warn] {result.custom_id}: {result.result.type}")
            else:
                still_pending.append(bid)

        pending = still_pending
        if pending:
            print(f"Waiting 30s for {len(pending)} batches...")
            time.sleep(30)

    return answers


def merge_answers(data: dict, answers: dict[str, str]) -> dict:
    for scene_id, entries in data.items():
        for entry in entries:
            job_stem = entry["job_stem"]
            for qa_idx, qa in enumerate(entry["questions_answers"]):
                cid = f"{scene_id}|{job_stem}|{qa_idx}"
                if cid in answers:
                    qa["predicted_answer"] = answers[cid]
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--submit", action="store_true", help="Submit batches, save IDs, exit")
    group.add_argument("--fetch", action="store_true", help="Fetch results from saved batch IDs")
    group.add_argument("--run", action="store_true", help="Submit + poll until done")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY environment variable.")

    client = anthropic.Anthropic(api_key=api_key)
    data = json.loads(EVAL_JSON.read_text())

    if args.submit or args.run:
        batch_ids = submit_batches(client, data)
        if args.submit:
            return

    if args.fetch:
        if not BATCH_IDS_FILE.exists():
            raise SystemExit(f"No batch IDs found at {BATCH_IDS_FILE}. Run --submit first.")
        batch_ids = json.loads(BATCH_IDS_FILE.read_text())

    print(f"Fetching results from {len(batch_ids)} batches...")
    answers = fetch_results(client, batch_ids)
    print(f"Got {len(answers)} answers.")

    data = merge_answers(data, answers)
    OUT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Written: {OUT_JSON}")


if __name__ == "__main__":
    main()
