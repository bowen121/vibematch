"""Generate multi-caption descriptions for all images using Qwen2.5-VL.

Two-stage pipeline per image:
  Stage 1 (dense)   — Qwen2.5-VL describes the image in detail (image + text).
  Stage 2 (rewrite) — Qwen2.5-VL rewrites into 3 concise factual captions (text-only).

Adds caption_1, caption_2, caption_3 columns to data/processed/{movies,books}.csv.
Idempotent — rows with all 3 captions already filled are skipped.
Each split writes to its own file (e.g. movies_split1.csv) to avoid race conditions.
Run --merge after all splits finish to fold results back into the original CSVs.

Usage:
    # parallel splits across 3 GPUs
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_descriptions.py --split 1/3 --batch-size 16 &
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_descriptions.py --split 2/3 --batch-size 16 &
    CUDA_VISIBLE_DEVICES=2 python scripts/generate_descriptions.py --split 3/3 --batch-size 16 &

    # after all finish
    python scripts/generate_descriptions.py --merge
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.loaders.data_loader import PROCESSED_CSVS

_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
CAPTION_COLS = ["caption_1", "caption_2", "caption_3"]

_MEDIA_TYPES = {"movie": "movie poster", "book": "book cover"}

_STAGE1_TEMPLATE = (
    "Describe this {media_type} in detail. "
    "Include: main objects and their attributes, spatial relationships, "
    "lighting conditions, and scene context. Be precise and descriptive."
)

_STAGE2_TEMPLATE = (
    "Rewrite the following description into exactly 3 different concise sentences (15-30 words each).\n"
    "Requirements: factual and neutral tone, no artistic or emotional words, "
    "emphasize visual properties and relationships, avoid redundancy.\n"
    "Format: number each sentence as 1. 2. 3.\n\n"
    "Description: {dense}"
)


def _split_path(csv_path: Path, split_idx: int) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_split{split_idx}{csv_path.suffix}")


def load_model() -> tuple[Qwen2_5_VLForConditionalGeneration, AutoProcessor]:
    print(f"[qwen2.5-vl] loading {_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(_MODEL_ID)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        _MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"[qwen2.5-vl] loaded on {next(model.parameters()).device}")
    return model, processor


def _run_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    messages_batch: list,
    max_new_tokens: int,
) -> list[str]:
    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]
    image_inputs, _ = process_vision_info(messages_batch)
    processor_kwargs = {"images": image_inputs} if image_inputs else {}
    inputs = processor(
        text=texts, **processor_kwargs, padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )


def _parse_captions(text: str) -> list[str]:
    parts = re.split(r"\n?\d+\.\s+", text.strip())
    captions = [p.strip() for p in parts if p.strip()]
    while len(captions) < 3:
        captions.append(captions[-1] if captions else "")
    return captions[:3]


def describe_batch(
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    image_paths: list[Path],
    sources: list[str],
) -> list[list[str | None]]:
    """Return [[cap1, cap2, cap3], ...] per image; [None, None, None] on failure."""
    pil_images: list[Image.Image | None] = []
    valid_mask: list[bool] = []
    for p in image_paths:
        try:
            with Image.open(p) as raw:
                pil_images.append(raw.convert("RGB"))
            valid_mask.append(True)
        except Exception as exc:
            print(f"\n[warn] {p.name}: {exc}", flush=True)
            pil_images.append(None)
            valid_mask.append(False)

    valid_indices = [i for i, ok in enumerate(valid_mask) if ok]
    if not valid_indices:
        return [[None, None, None] for _ in image_paths]

    valid_images = [pil_images[i] for i in valid_indices]
    valid_sources = [sources[i] for i in valid_indices]

    stage1_msgs = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": _STAGE1_TEMPLATE.format(
                media_type=_MEDIA_TYPES.get(src, "cover image")
            )},
        ]}]
        for img, src in zip(valid_images, valid_sources)
    ]
    dense_outputs = _run_batch(model, processor, stage1_msgs, max_new_tokens=256)

    stage2_msgs = [
        [{"role": "user", "content": [
            {"type": "text", "text": _STAGE2_TEMPLATE.format(dense=dense)},
        ]}]
        for dense in dense_outputs
    ]
    stage2_outputs = _run_batch(model, processor, stage2_msgs, max_new_tokens=150)

    parsed = [_parse_captions(out) for out in stage2_outputs]

    results: list[list[str | None]] = [[None, None, None] for _ in image_paths]
    for out_idx, orig_idx in enumerate(valid_indices):
        results[orig_idx] = parsed[out_idx]
    return results


def process_csv(
    csv_path: Path,
    data_root: Path,
    model: Qwen2_5_VLForConditionalGeneration,
    processor: AutoProcessor,
    batch_size: int,
    save_every: int,
    split: tuple[int, int] | None = None,
) -> None:
    df = pd.read_csv(csv_path)
    for col in CAPTION_COLS:
        if col not in df.columns:
            df[col] = None

    # Determine output path — split file or original
    if split is not None:
        split_idx, total_splits = split
        out_path = _split_path(csv_path, split_idx)
        # Resume from existing split file if present
        if out_path.exists():
            split_df = pd.read_csv(out_path, index_col=0)
            for col in CAPTION_COLS:
                if col in split_df.columns:
                    df.loc[split_df.index, col] = split_df[col].values
    else:
        split_idx, total_splits, out_path = None, None, csv_path

    todo_mask = df[CAPTION_COLS].isna().any(axis=1) | (df[CAPTION_COLS] == "").any(axis=1)
    todo_idx = df.index[todo_mask].tolist()

    if not todo_idx:
        print(f"[{csv_path.name}] all captions present — skipping.")
        return

    if split is not None:
        chunk = math.ceil(len(todo_idx) / total_splits)
        todo_idx = todo_idx[(split_idx - 1) * chunk : split_idx * chunk]
        print(f"[{csv_path.name}] split {split_idx}/{total_splits} → {len(todo_idx)} images → {out_path.name}")

    if not todo_idx:
        print(f"[{csv_path.name}] nothing left for this split — skipping.")
        return

    print(f"[{csv_path.name}] generating captions for {len(todo_idx)} images (batch_size={batch_size}) ...")

    save_every_batches = max(1, save_every // batch_size)

    for batch_num, batch_start in enumerate(
        tqdm(range(0, len(todo_idx), batch_size), desc=csv_path.stem)
    ):
        batch_idx = todo_idx[batch_start : batch_start + batch_size]
        img_paths = [data_root / str(df.loc[i, "image_path"]) for i in batch_idx]
        sources = [str(df.loc[i, "source"]) for i in batch_idx]

        results = describe_batch(model, processor, img_paths, sources)
        for idx, captions in zip(batch_idx, results):
            for col, cap in zip(CAPTION_COLS, captions):
                if cap:
                    df.at[idx, col] = cap

        if (batch_num + 1) % save_every_batches == 0:
            # Save only this split's rows to the split file
            df.loc[todo_idx].to_csv(out_path, index=True)

    df.loc[todo_idx].to_csv(out_path, index=True)
    n_filled = df.loc[todo_idx, "caption_1"].notna().sum()
    print(f"[{out_path.name}] done — {n_filled}/{len(todo_idx)} rows have captions.")


def merge_splits(processed_dir: str) -> None:
    proc = Path(processed_dir)
    for fname in PROCESSED_CSVS:
        csv_path = proc / fname
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found.")
            continue

        df = pd.read_csv(csv_path)
        for col in CAPTION_COLS:
            if col not in df.columns:
                df[col] = None

        split_files = sorted(proc.glob(f"{csv_path.stem}_split*.csv"))
        if not split_files:
            print(f"[{fname}] no split files found — nothing to merge.")
            continue

        print(f"[{fname}] merging {len(split_files)} split files ...")
        for sf in split_files:
            sdf = pd.read_csv(sf, index_col=0)
            for col in CAPTION_COLS:
                if col not in sdf.columns:
                    continue
                mask = sdf[col].notna() & (sdf[col] != "")
                df.loc[sdf.index[mask], col] = sdf.loc[mask, col].values

        df.to_csv(csv_path, index=False)
        n_filled = df["caption_1"].notna().sum()
        print(f"[{fname}] merged — {n_filled}/{len(df)} rows have captions.")


def main(
    processed_dir: str,
    data_root: str,
    batch_size: int,
    save_every: int,
    split: tuple[int, int] | None,
    merge: bool,
) -> None:
    if merge:
        merge_splits(processed_dir)
        return

    model, processor = load_model()
    proc = Path(processed_dir)
    root = Path(data_root)

    for fname in PROCESSED_CSVS:
        csv_path = proc / fname
        if not csv_path.exists():
            print(f"[skip] {csv_path} not found — run scripts/prepare_data.py first.")
            continue
        process_csv(csv_path, root, model, processor, batch_size, save_every, split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--split", default=None, help="e.g. 1/3, 2/3, 3/3")
    parser.add_argument("--merge", action="store_true", help="Merge split files into original CSVs")
    args = parser.parse_args()

    split = None
    if args.split:
        idx, total = args.split.split("/")
        split = (int(idx), int(total))

    main(args.processed_dir, args.data_root, args.batch_size, args.save_every, split, args.merge)
