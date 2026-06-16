#!/usr/bin/env python3
"""test_npu_vlm_imagesize.py — where does the NPU's 1024-token VLM prompt
cap actually bite, and is the model still correct at the largest image that
fits?

Background: NPU VLMPipeline limits the prompt to ~1024 tokens *including image
tokens* (genai #3366 — MAX_PROMPT_LEN doesn't help on the NPU VLM path).
Qwen2.5-VL tokenizes images at one token per 28x28 px (patch 14 * merge 2),
and its default processor allows up to max_pixels=12.8M (~16k tokens), so a
normal photo blows the budget. This sweep feeds progressively larger images
through the *live* NPU pipeline to find:

  1. the largest image that loads + generates without tripping the cap, and
  2. whether the answer is still correct there (usable, not just runnable).

Each test image is a solid dominant color with a small contrasting square, so
"what is the main/background color?" has a verifiable right answer.

Usage:
    python test_npu_vlm_imagesize.py [model_dir] [--device NPU]
"""

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

import openvino as ov
import openvino_genai as ovg

FACTOR = 28  # Qwen2.5-VL: patch_size(14) * merge_size(2) -> one token per 28x28


def est_image_tokens(w, h):
    """Mirror Qwen2VL smart_resize token count: round each side to a multiple
    of 28, then (w/28)*(h/28). Assumes no min/max_pixels rescale (true for the
    sizes we test, all under the 12.8M default)."""
    rw = max(FACTOR, round(w / FACTOR) * FACTOR)
    rh = max(FACTOR, round(h / FACTOR) * FACTOR)
    return (rw // FACTOR) * (rh // FACTOR)


def make_image(w, h):
    """Mostly-blue image with a small red square. Dominant color = blue."""
    img = Image.new("RGB", (w, h), (30, 60, 200))      # blue field
    sq = max(8, min(w, h) // 8)
    for y in range(sq):
        for x in range(sq):
            img.putpixel((x, y), (200, 30, 30))         # small red corner
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None, ...])
    return ov.Tensor(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?",
                    default=os.path.expanduser("~/models/Qwen2.5-VL-3B-Instruct-ov-int4-npu"))
    ap.add_argument("--device", default="NPU")
    args = ap.parse_args()

    print("=== NPU VLM image-size sweep ===")
    print(f"openvino {ov.__version__}  genai {ovg.__version__}")
    print(f"device   {args.device}")
    print(f"model    {args.model}\n")

    t0 = time.perf_counter()
    pipe = ovg.VLMPipeline(args.model, device=args.device)
    print(f"VLMPipeline load: {time.perf_counter() - t0:.1f}s\n")

    # (label, w, h)
    sizes = [
        ("224x224", 224, 224),
        ("448x448", 448, 448),
        ("672x672", 672, 672),
        ("784x784", 784, 784),
        ("896x896", 896, 896),
        ("1024x768 (real)", 1024, 768),
        ("1280x720 (720p)", 1280, 720),
        ("1920x1080 (1080p)", 1920, 1080),
    ]

    prompt = "What is the main background color of this image? Answer with one word."
    print(f"{'image':<20} {'img_tok':>8} {'result':<8} {'latency':>8}  answer")
    print("-" * 70)

    largest_ok = None
    for label, w, h in sizes:
        toks = est_image_tokens(w, h)
        gen = ovg.GenerationConfig()
        gen.max_new_tokens = 10
        gen.do_sample = False
        try:
            t0 = time.perf_counter()
            r = pipe.generate(prompt=prompt, images=make_image(w, h), generation_config=gen)
            dt = time.perf_counter() - t0
            txt = (r if isinstance(r, str) else getattr(r, "texts", [str(r)])[0]).strip()
            correct = "blue" in txt.lower()
            mark = "OK" if correct else "RAN?"
            if correct:
                largest_ok = (label, toks)
            print(f"{label:<20} {toks:>8} {mark:<8} {dt:>7.1f}s  {txt!r}")
        except Exception as e:
            msg = str(e).replace("\n", " ")[:80]
            print(f"{label:<20} {toks:>8} {'FAIL':<8} {'-':>8}  {type(e).__name__}: {msg}")

    print("-" * 70)
    if largest_ok:
        print(f"Largest image answered correctly: {largest_ok[0]}  (~{largest_ok[1]} image tokens)")
    else:
        print("No image produced a correct answer.")


if __name__ == "__main__":
    main()
