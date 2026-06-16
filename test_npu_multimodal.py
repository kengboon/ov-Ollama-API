#!/usr/bin/env python3
"""test_npu_multimodal.py — can the new natively-multimodal Qwen3.5/3.6
exports actually run on the NPU, and via which pipeline?

Intel's Qwen3.5/Qwen3.6 OpenVINO exports are tagged image-text-to-text
(natively multimodal), unlike Qwen3-* which were plain text-generation.
NoLlama's routing assumes text->NPU (LLMPipeline) and images->GPU
(VLMPipeline). NPU VLM support is immature, so before we trust one of
these models on the NPU we need to know empirically:

  1. Does it load + generate TEXT on the NPU via LLMPipeline?
  2. Does VLMPipeline load on the NPU at all (text-only, then with an image)?
  3. What does is_vlm() (config.json sniff) classify it as?

This probes all three and prints a recommendation. It does NOT import the
Flask server — it reuses the same load conventions standalone.

Usage:
    python test_npu_multimodal.py                     # auto-find a Qwen3.5/3.6 model
    python test_npu_multimodal.py <model_dir>
    python test_npu_multimodal.py --model <dir> --device NPU
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

import openvino as ov
import openvino_genai as ovg


# --- model discovery -------------------------------------------------------

def find_model(explicit):
    """Resolve a model dir: explicit arg, then ./model, then ~/models/Qwen3.5*."""
    if explicit:
        if not os.path.isdir(explicit):
            sys.exit(f"ERROR: not a directory: {explicit}")
        return explicit

    candidates = []
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    if os.path.isfile(os.path.join(here, "config.json")):
        candidates.append(here)
    models_root = os.path.join(os.path.expanduser("~"), "models")
    for pat in ("Qwen3.5*", "Qwen3.6*"):
        candidates += sorted(glob.glob(os.path.join(models_root, pat)))

    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c

    sys.exit(
        "ERROR: no model found. Install one first, e.g.\n"
        "  hf download OpenVINO/Qwen3.5-4B-int4-ov --local-dir ~/models/Qwen3.5-4B-int4-ov\n"
        "then pass it:  python test_npu_multimodal.py ~/models/Qwen3.5-4B-int4-ov"
    )


def is_vlm(model_dir):
    """Mirror of nollama.is_vlm() — config.json architecture/model_type sniff."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False, "", ""
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        arch = (cfg.get("architectures", [""]) or [""])[0].lower()
        model_type = cfg.get("model_type", "").lower()
        vlm = any(
            k in arch or k in model_type
            for k in ("vl", "vision", "llava", "qwen2vl", "internvl", "minicpm")
        )
        return vlm, arch, model_type
    except Exception:
        return False, "", ""


# --- probes ----------------------------------------------------------------

def _gen_config(max_new=24):
    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_new
    gen.do_sample = False
    gen.top_k = 1
    return gen


def _throughput(result, elapsed, max_new):
    """Best-effort tokens/sec. Prefer perf_metrics, else estimate."""
    try:
        tp = result.perf_metrics.get_throughput().mean
        if tp and tp > 0:
            return f"{tp:.1f} tok/s (perf_metrics)"
    except Exception:
        pass
    return f"~{max_new / elapsed:.1f} tok/s (est, {max_new} tok / {elapsed:.1f}s)"


def probe_llm(model_dir, device):
    """Load as LLMPipeline (text) on the device and generate a short reply."""
    print(f"\n[1] LLMPipeline on {device} (text-only)")
    kwargs = {"MAX_PROMPT_LEN": 4096} if device == "NPU" else {}
    try:
        t0 = time.perf_counter()
        pipe = ovg.LLMPipeline(str(model_dir), device=device, **kwargs)
        print(f"    load... ok ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        print(f"    load... FAILED: {type(e).__name__}: {e}")
        return False

    try:
        gen = _gen_config()
        hist = ovg.ChatHistory()
        hist.append({"role": "user", "content": "Reply with exactly one word: hello."})
        t0 = time.perf_counter()
        result = pipe.generate(hist, gen)
        elapsed = time.perf_counter() - t0
        text = result if isinstance(result, str) else getattr(result, "texts", [str(result)])[0]
        print(f"    generate... ok: {text.strip()!r}")
        print(f"    {_throughput(result, elapsed, gen.max_new_tokens)}")
        print("    RESULT: PASS — runs as a text LLM on this device")
        return True
    except Exception as e:
        print(f"    generate... FAILED: {type(e).__name__}: {e}")
        print("    RESULT: FAIL — loads but cannot generate text")
        return False


def _synthetic_image():
    """A tiny 64x64 image as an NHWC uint8 ov.Tensor (matches pil_to_tensor)."""
    img = Image.new("RGB", (64, 64), (200, 30, 30))
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None, ...])
    return ov.Tensor(arr)


def probe_vlm(model_dir, device):
    """Load as VLMPipeline on the device; try text-only, then with an image."""
    print(f"\n[2] VLMPipeline on {device}")
    VLMPipe = getattr(ovg, "VLMPipeline", None)
    if VLMPipe is None:
        print("    skipped: no VLMPipeline in this openvino_genai build")
        return False, False
    try:
        t0 = time.perf_counter()
        pipe = VLMPipe(str(model_dir), device=device)
        print(f"    load... ok ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:
        print(f"    load... FAILED: {type(e).__name__}: {e}")
        print("    RESULT: FAIL — VLMPipeline not supported on this device/build")
        return False, False

    text_ok = False
    try:
        gen = _gen_config()
        result = pipe.generate(prompt="Reply with one word: ok.", generation_config=gen)
        text = result if isinstance(result, str) else getattr(result, "texts", [str(result)])[0]
        print(f"    text-only generate... ok: {text.strip()!r}")
        text_ok = True
    except Exception as e:
        print(f"    text-only generate... FAILED: {type(e).__name__}: {e}")

    image_ok = False
    try:
        gen = _gen_config()
        result = pipe.generate(
            prompt="What color is this image? One word.",
            images=_synthetic_image(),
            generation_config=gen,
        )
        text = result if isinstance(result, str) else getattr(result, "texts", [str(result)])[0]
        print(f"    image generate... ok: {text.strip()!r}")
        image_ok = True
    except Exception as e:
        print(f"    image generate... FAILED: {type(e).__name__}: {e}")

    verdict = "PASS" if image_ok else ("PARTIAL (text only)" if text_ok else "FAIL")
    print(f"    RESULT: {verdict}")
    return text_ok, image_ok


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Probe a multimodal model on the NPU.")
    ap.add_argument("model", nargs="?", help="model directory (auto-detected if omitted)")
    ap.add_argument("--model", dest="model_opt", help="model directory (alternative form)")
    ap.add_argument("--device", default="NPU", help="OpenVINO device (default: NPU)")
    args = ap.parse_args()

    model_dir = find_model(args.model_opt or args.model)
    device = args.device

    print("=== NoLlama NPU multimodal probe ===")
    print(f"openvino     {ov.__version__}")
    print(f"genai        {ovg.__version__}")
    print(f"device       {device}")
    print(f"model        {model_dir}")

    available = ov.Core().get_available_devices()
    if device not in available and not any(d.startswith(device) for d in available):
        print(f"\nERROR: device {device!r} not available. Found: {available}")
        sys.exit(2)

    vlm, arch, mtype = is_vlm(model_dir)
    print(f"config       architectures={arch or '?'}  model_type={mtype or '?'}")
    print(f"is_vlm()     {vlm}  -> nollama would route this as {'VLM' if vlm else 'LLM'}")

    llm_ok = probe_llm(model_dir, device)
    vlm_text_ok, vlm_image_ok = probe_vlm(model_dir, device)

    print("\n=== Summary ===")
    name = os.path.basename(os.path.normpath(os.path.realpath(model_dir)))
    if llm_ok:
        print(f"  {name} runs as a TEXT LLM on {device}  ✓")
        if not vlm:
            print("  is_vlm() agrees (LLM) — safe in the NPU registry as a text model.")
        else:
            print("  NOTE: is_vlm() says VLM, but text via LLMPipeline works. nollama would")
            print("        try VLMPipeline — see [2] for whether that path is viable here.")
    else:
        print(f"  {name} does NOT run as a text LLM on {device}  ✗")
        print(f"  Do not add it as an NPU text model. Route it to GPU instead.")

    if vlm_image_ok:
        print(f"  Vision (image) works on {device} via VLMPipeline  ✓")
    elif vlm_text_ok:
        print(f"  VLMPipeline loads on {device} but image input failed — text-only.")
    else:
        print(f"  VLMPipeline not viable on {device} — keep images on the GPU.")


if __name__ == "__main__":
    main()
