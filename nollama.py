#!/usr/bin/env python3
"""NoLlama — OpenAI-compatible API server for Intel NPU / ARC GPU.

Auto-detects available devices (NPU, GPU, CPU) and model type (VLM/LLM).
NPU-first: works on any Intel Core Ultra laptop. ARC GPU optional.
Dual mode: NPU for chat + GPU for vision, simultaneously.

Usage:
    python nollama.py                                        # auto-detect device
    python nollama.py --device NPU                           # force NPU
    python nollama.py --device GPU                           # force GPU
    python nollama.py --gpu-model-dir gpu-model              # dual: NPU chat + GPU vision
    python nollama.py --model-dir ~/models/qwen3-14b-int4-ov --device GPU  # big LLM on GPU
    python nollama.py --whisper-dir whisper-model             # add speech-to-text
"""

import argparse
import base64
import io
import itertools
import json
import os
import socket
import sys
import time
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from urllib.parse import unquote

import numpy as np
import openvino as ov
import openvino_genai as ovg
from flask import Flask, Response, jsonify, request, render_template
from PIL import Image
try:
    import soundfile as sf
except ImportError:
    sf = None

# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def is_vlm(model_dir):
    """Detect if a model is a VLM by checking config.json."""
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return False
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        arch = cfg.get("architectures", [""])[0].lower()
        model_type = cfg.get("model_type", "").lower()
        return any(
            k in arch or k in model_type
            for k in ("vl", "vision", "llava", "qwen2vl", "internvl", "minicpm")
        )
    except Exception:
        return False


def model_display_name(model_dir):
    """Extract a human-readable model name from directory or config."""
    resolved = os.path.realpath(model_dir)
    name = os.path.basename(os.path.normpath(resolved))
    for suffix in ("-ov", "-openvino", "-int8", "-int4"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    if name in ("model", "models", "gpu-model", "npu-model", ""):
        cfg_path = os.path.join(model_dir, "config.json")
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            name = cfg.get("model_type", name) or name
        except Exception:
            pass
    return name or "unknown"


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image(url_or_data, max_dim):
    """Load an image from a base64 data URI or file:// URI."""
    if url_or_data.startswith("data:"):
        header, b64data = url_or_data.split(",", 1)
        img_bytes = base64.b64decode(b64data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif url_or_data.startswith("file:///"):
        path = unquote(url_or_data[8:])
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        return Image.open(path).convert("RGB")
    else:
        raise ValueError(
            f"Unsupported image URL scheme. Use data:image/...;base64,... "
            f"or file:///path. Got: {url_or_data[:80]}"
        )


def pil_to_tensor(img, max_dim):
    """Convert PIL Image to OpenVINO Tensor (NHWC uint8)."""
    if max(img.width, img.height) > max_dim:
        ratio = max_dim / max(img.width, img.height)
        img = img.resize(
            (int(img.width * ratio), int(img.height * ratio)), Image.LANCZOS
        )
    arr = np.ascontiguousarray(np.asarray(img, dtype=np.uint8)[None, ...])
    return ov.Tensor(arr)


# ---------------------------------------------------------------------------
# Request parsing
# ---------------------------------------------------------------------------

def parse_messages(messages, max_dim):
    """Parse OpenAI messages. Returns (text_prompt, images, raw_messages)."""
    text_parts = []
    images = []
    raw_messages = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if isinstance(content, str):
            text_parts.append(content)
            raw_messages.append({"role": role, "content": content})
            continue

        msg_text = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                msg_text.append(block.get("text", ""))
            elif btype == "image_url":
                url = block.get("image_url", {}).get("url", "")
                if url:
                    img = load_image(url, max_dim)
                    images.append(pil_to_tensor(img, max_dim))

        joined = " ".join(msg_text)
        text_parts.append(joined)
        raw_messages.append({"role": role, "content": joined})

    return "\n".join(text_parts), images, raw_messages


# ---------------------------------------------------------------------------
# Response extraction
# ---------------------------------------------------------------------------

def extract_text(result):
    """Extract text from an openvino_genai generate result."""
    if isinstance(result, str):
        return result.strip()
    for attr in ("texts", "text", "output_text", "response"):
        if hasattr(result, attr):
            val = getattr(result, attr)
            if isinstance(val, (list, tuple)):
                return val[0].strip() if val else ""
            return val.strip()
    return str(result).strip()


# ---------------------------------------------------------------------------
# Device slot — holds one pipeline + its metadata
# ---------------------------------------------------------------------------

class DeviceSlot:
    """One loaded model on one device."""

    def __init__(self, device_name):
        self.device_name = device_name   # "NPU", "GPU", "CPU"
        self.device_full = ""            # "Intel(R) AI Boost"
        self.pipe = None
        self.model_name = ""
        self.model_type = ""             # "vlm" or "llm"
        self.status = "not_configured"   # not_configured -> loading -> warming_up -> ready / error / idle_unloaded
        self.lock = threading.Lock()
        self._cancel = threading.Event()  # signal to stop generation
        self.last_used = time.time()     # for idle-unload watchdog
        self.model_dir = None            # remembered so we can reload after unload

    def load(self, model_dir):
        """Load model, auto-detecting VLM vs LLM."""
        self.status = "loading"
        self.model_dir = model_dir
        self.model_name = model_display_name(model_dir)
        vlm = is_vlm(model_dir)
        self.model_type = "vlm" if vlm else "llm"

        print(f"  [{self.device_name}] Detected: {self.model_type.upper()} ({self.model_name})")
        print(f"  [{self.device_name}] Loading...", flush=True)

        if vlm:
            VLMPipe = getattr(ovg, "VLMPipeline", None)
            if VLMPipe is None:
                raise RuntimeError("No VLMPipeline in this openvino_genai build.")
            self.pipe = VLMPipe(str(model_dir), device=self.device_name)
        else:
            # NPU has a default prompt limit of 1024 tokens — raise it
            if self.device_name == "NPU":
                self.pipe = ovg.LLMPipeline(
                    str(model_dir), device=self.device_name,
                    MAX_PROMPT_LEN=4096,
                )
            else:
                self.pipe = ovg.LLMPipeline(str(model_dir), device=self.device_name)

    def warmup(self):
        self.status = "warming_up"
        print(f"  [{self.device_name}] Warmup...", end="", flush=True)
        t0 = time.perf_counter()
        gen = ovg.GenerationConfig()
        gen.max_new_tokens = 5
        gen.do_sample = False
        gen.top_k = 1
        try:
            if self.model_type == "vlm":
                self.pipe.generate(prompt="Hello", generation_config=gen)
            else:
                history = ovg.ChatHistory()
                history.append({"role": "user", "content": "Hi"})
                self.pipe.generate(history, gen)
            elapsed = time.perf_counter() - t0
            print(f" done ({elapsed:.1f}s)", flush=True)
            self.status = "ready"
        except Exception as e:
            print(f" failed: {e}", flush=True)
            self.status = "error"

    def unload(self):
        """Release the loaded pipeline. Caller must hold self.lock."""
        if self.pipe is None:
            return
        print(f"  [{self.device_name}] Idle — unloading {self.model_name}", flush=True)
        self.pipe = None
        self.status = "idle_unloaded"
        import gc
        gc.collect()

    def ensure_loaded(self):
        """Reload pipeline if it was unloaded. Blocks until ready."""
        if self.pipe is not None and self.status == "ready":
            return
        with self.lock:
            if self.pipe is not None and self.status == "ready":
                return  # someone else loaded it while we waited
            if self.model_dir is None:
                raise RuntimeError(f"Slot {self.device_name} has no model_dir")
            print(f"  [{self.device_name}] Reloading {self.model_name}...", flush=True)
            self.load(self.model_dir)
            self.warmup()

    def generate_vlm(self, text_prompt, images, gen):
        """VLM generate — images optional."""
        with self.lock:
            if images:
                imgs = images[0] if len(images) == 1 else images
                result = self.pipe.generate(
                    prompt=text_prompt, images=imgs, generation_config=gen,
                )
            else:
                result = self.pipe.generate(
                    prompt=text_prompt, generation_config=gen,
                )
            self.last_used = time.time()
        return extract_text(result)

    def generate_llm(self, raw_messages, gen):
        """LLM generate — non-streaming."""
        history = ovg.ChatHistory()
        for msg in raw_messages:
            history.append({"role": msg["role"], "content": msg["content"]})
        with self.lock:
            result = self.pipe.generate(history, gen)
            self.last_used = time.time()
        return extract_text(result)

    def cancel(self):
        """Signal the current generation to stop."""
        self._cancel.set()

    def stream_llm(self, raw_messages, gen, completion_id, created, t0):
        """LLM generate — SSE streaming."""
        history = ovg.ChatHistory()
        for msg in raw_messages:
            history.append({"role": msg["role"], "content": msg["content"]})

        token_queue = Queue()
        token_count = 0
        cancelled = False

        def streamer_callback(token):
            if self._cancel.is_set():
                return True  # stop generation
            token_queue.put(token)
            return False

        gen_error = [None]  # captured from generate thread

        def _generate():
            try:
                with self.lock:
                    # Clear inside the lock, just before generation, to avoid
                    # racing with the previous request's finally: _cancel.set()
                    self._cancel.clear()
                    self.pipe.generate(history, gen, streamer_callback)
                    self.last_used = time.time()
            except Exception as e:
                gen_error[0] = e
                print(f"{datetime.now():%H:%M:%S} !! [{self.device_name}] "
                      f"generate error: {e}", flush=True)
            finally:
                token_queue.put(None)

        t = threading.Thread(target=_generate, daemon=True)
        t.start()

        try:
            chunk = {
                "id": completion_id, "object": "chat.completion.chunk",
                "created": created, "model": self.model_name,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            while True:
                try:
                    token = token_queue.get(timeout=120)
                except Empty:
                    break
                if token is None:
                    break
                token_count += 1
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # Capture state BEFORE the finally-block safety-net sets _cancel
            was_cancelled = self._cancel.is_set()
            if gen_error[0] is not None:
                finish_reason = "error"
                err_chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {
                        "content": f"\n[error: {gen_error[0]}]"
                    }, "finish_reason": "error"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
            else:
                finish_reason = "cancelled" if was_cancelled else "stop"
                chunk = {
                    "id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": self.model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Safety net: if client disconnects, stop generation
            self._cancel.set()

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0
        tag = " (cancelled)" if was_cancelled else (" (error)" if gen_error[0] else "")
        print(f"{datetime.now():%H:%M:%S} -> [{self.device_name}] "
              f"{token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s){tag}",
              flush=True)

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device": self.device_full,
        }


# ---------------------------------------------------------------------------
# Whisper (speech-to-text) slot
# ---------------------------------------------------------------------------

def _load_audio(file_storage):
    """Read uploaded audio file to float32 numpy array at 16 kHz."""
    if sf is None:
        raise RuntimeError("soundfile not installed. pip install soundfile")
    audio, sr = sf.read(io.BytesIO(file_storage.read()), dtype="float32")
    # Stereo → mono
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # Resample to 16 kHz if needed
    if sr != 16000:
        target_len = int(len(audio) * 16000 / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, target_len),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


class WhisperSlot:
    """Holds a WhisperPipeline for speech-to-text."""

    def __init__(self, device_name):
        self.device_name = device_name
        self.device_full = ""
        self.pipe = None
        self.model_name = ""
        self.model_type = "stt"
        self.status = "not_configured"
        self.lock = threading.Lock()

    def load(self, model_dir):
        self.status = "loading"
        self.model_name = model_display_name(model_dir)
        print(f"  [{self.device_name}] Loading Whisper ({self.model_name})...",
              flush=True)
        WhisperPipe = getattr(ovg, "WhisperPipeline", None)
        if WhisperPipe is None:
            raise RuntimeError(
                "No WhisperPipeline in this openvino_genai build. "
                "Upgrade to openvino-genai >= 2025.1."
            )
        self.pipe = WhisperPipe(str(model_dir), self.device_name)

    def warmup(self):
        self.status = "ready"
        print(f"  [{self.device_name}] Whisper ready", flush=True)

    def transcribe(self, audio_samples, language=None):
        """Transcribe float32 audio at 16 kHz. Returns text."""
        kwargs = {}
        if language:
            kwargs["language"] = f"<|{language}|>"
            kwargs["task"] = "transcribe"
        with self.lock:
            result = self.pipe.generate(audio_samples, **kwargs)
        if hasattr(result, "texts") and result.texts:
            return result.texts[0].strip()
        return str(result).strip()

    @property
    def info(self):
        return {
            "status": self.status,
            "model": self.model_name,
            "type": self.model_type,
            "device": self.device_full,
        }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_REQUEST_BYTES = 50 * 1024 * 1024  # 50 MB — enough for large base64 images

app = Flask("NoLlama",
            template_folder=os.path.join(SCRIPT_DIR, "templates"),
            static_folder=os.path.join(SCRIPT_DIR, "static"))
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

# Device slots — filled in main()
primary = None        # main model (NPU, GPU, or CPU)
secondary = None      # optional second model (GPU, for vision or bigger LLM)
whisper_slot = None   # optional Whisper STT model
max_dim = 768
_request_counter = itertools.count(1)  # thread-safe id generator


def make_id():
    return f"arc-{next(_request_counter):04d}"


def overall_status():
    """Ready when all configured devices are ready."""
    slots = [s for s in (primary, secondary) if s and s.status != "not_configured"]
    if not slots:
        return "not_configured"
    # If ANY slot is ready or idle_unloaded (will reload on demand), we can
    # serve requests. A dead secondary shouldn't kill the primary.
    if any(s.status in ("ready", "idle_unloaded") for s in slots):
        return "ready"
    if all(s.status == "error" for s in slots):
        return "error"
    return "loading"


def openai_error(message, error_type="invalid_request_error", status=400):
    return jsonify({"error": {"message": message, "type": error_type}}), status


def _slot_serviceable(slot):
    """A slot can serve requests if loaded or just idle-unloaded (will reload)."""
    return slot and slot.status in ("ready", "idle_unloaded")


def _route_request(has_images, requested_model):
    """Pick which DeviceSlot handles this request."""
    # Explicit model@device selection overrides routing
    if requested_model:
        for slot in (primary, secondary):
            if not _slot_serviceable(slot):
                continue
            # Match "model@DEVICE" or just "model"
            slot_full = f"{slot.model_name}@{slot.device_name}"
            if requested_model in (slot_full, slot.model_name):
                return slot

    # Dual mode routing
    if _slot_serviceable(secondary):
        if has_images:
            # Images → whichever slot is a VLM
            for slot in (secondary, primary):
                if _slot_serviceable(slot) and slot.model_type == "vlm":
                    return slot
            return None  # no VLM loaded
        else:
            # Text → prefer the better/primary model
            # If GPU has a big LLM, use GPU. Otherwise use primary (NPU).
            if secondary.model_type == "llm":
                return secondary  # GPU has a big LLM — use it
            return primary  # GPU has VLM, text goes to NPU

    # Single mode — everything goes to primary
    return primary if _slot_serviceable(primary) else None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.route("/")
def gui():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    devices = {}
    if primary and primary.status != "not_configured":
        devices[primary.device_name.lower()] = primary.info
    if secondary and secondary.status != "not_configured":
        devices[secondary.device_name.lower()] = secondary.info
    result = {"status": overall_status(), "devices": devices}
    if whisper_slot and whisper_slot.status != "not_configured":
        result["whisper"] = whisper_slot.info
    return jsonify(result)


@app.route("/v1/models", methods=["GET"])
def list_models():
    data = []
    for slot in (primary, secondary):
        if slot and slot.status == "ready":
            data.append({
                "id": f"{slot.model_name}@{slot.device_name}",
                "object": "model",
                "created": 0,
                "owned_by": f"local-{slot.device_name.lower()}",
            })
    if whisper_slot and whisper_slot.status == "ready":
        data.append({
            "id": f"whisper@{whisper_slot.device_name}",
            "object": "model",
            "created": 0,
            "owned_by": f"local-{whisper_slot.device_name.lower()}",
        })
    return jsonify({"object": "list", "data": data})


@app.route("/v1/cancel", methods=["POST"])
def cancel_generation():
    """Stop any in-progress generation. Returns immediately."""
    for slot in (primary, secondary):
        if slot:
            slot.cancel()
    return jsonify({"status": "ok"})


@app.route("/v1/audio/transcriptions", methods=["POST"])
def audio_transcriptions():
    """OpenAI-compatible speech-to-text. Accepts multipart form with audio file."""
    if not whisper_slot or whisper_slot.status != "ready":
        return openai_error(
            "No speech-to-text model loaded. Use --whisper-dir.", "server_error", 503,
        )

    if "file" not in request.files:
        return openai_error("'file' is required (multipart form upload)")

    audio_file = request.files["file"]
    language = request.form.get("language")
    response_format = request.form.get("response_format", "json")

    try:
        audio_samples = _load_audio(audio_file)
    except Exception as e:
        return openai_error(f"Failed to read audio: {e}")

    duration = len(audio_samples) / 16000
    lang_tag = f", lang={language}" if language else ""
    print(f"\n{datetime.now():%H:%M:%S} <- [{whisper_slot.device_name}] "
          f"Whisper {duration:.1f}s audio{lang_tag}", flush=True)

    t0 = time.perf_counter()
    try:
        text = whisper_slot.transcribe(audio_samples, language=language)
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{whisper_slot.device_name}] "
              f"Whisper error: {e}", flush=True)
        return openai_error(f"Transcription failed: {e}", "server_error", 500)

    elapsed = time.perf_counter() - t0
    print(f"{datetime.now():%H:%M:%S} -> [{whisper_slot.device_name}] "
          f"Whisper {len(text)} chars in {elapsed:.1f}s", flush=True)

    if response_format == "text":
        return Response(text, mimetype="text/plain")

    return jsonify({"text": text})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    if overall_status() != "ready":
        return openai_error(
            f"Server not ready (status: {overall_status()}). "
            "Check GET /health.", "server_error", 503,
        )

    body = request.get_json(silent=True)
    if not body:
        return openai_error("Request body must be JSON")
    messages = body.get("messages")
    if not messages:
        return openai_error("'messages' is required")

    max_tokens = body.get("max_tokens", 4096)
    temperature = body.get("temperature", 0.0)
    top_p = body.get("top_p", 1.0)
    stream = body.get("stream", False)
    requested_model = body.get("model", "")

    # Parse messages
    try:
        text_prompt, images, raw_messages = parse_messages(messages, max_dim)
    except FileNotFoundError as e:
        return openai_error(str(e))
    except ValueError as e:
        return openai_error(str(e))
    except Exception as e:
        return openai_error(f"Failed to parse request: {e}")

    # Route to device
    slot = _route_request(bool(images), requested_model)
    if slot is None:
        if images:
            return openai_error("No vision model loaded. Send text only, or load a VLM.")
        return openai_error("No model ready to handle this request.", "server_error", 503)

    # Reject images on LLM
    if images and slot.model_type == "llm":
        return openai_error(
            f"Model '{slot.model_name}' on {slot.device_name} does not support images. "
            "Remove image content or load a VLM."
        )

    # Reload if the slot was idle-unloaded (blocks until ready)
    try:
        slot.ensure_loaded()
    except Exception as e:
        return openai_error(f"Failed to reload model: {e}", "server_error", 500)

    # Build generation config
    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
        gen.top_p = top_p
    else:
        gen.do_sample = False
        gen.top_k = 1
    gen.repetition_penalty = 1.05

    completion_id = make_id()
    created = int(time.time())
    n_images = len(images)
    tag = f"{n_images} image{'s' if n_images != 1 else ''}, " if n_images else ""
    stream_tag = " (stream)" if stream else ""
    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] {tag}"
          f"{len(text_prompt)} chars, max_tokens={max_tokens}{stream_tag}",
          flush=True)

    t0 = time.perf_counter()

    # --- VLM path ---
    if slot.model_type == "vlm":
        try:
            text = slot.generate_vlm(text_prompt, images, gen)
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] VLM error: {e}", flush=True)
            return openai_error(f"Inference failed: {e}", "server_error", 500)

        elapsed = time.perf_counter() - t0
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
              f"{len(text)} chars in {elapsed:.1f}s", flush=True)

        resp = jsonify({
            "id": completion_id, "object": "chat.completion",
            "created": created, "model": slot.model_name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
        })
        resp.headers["X-Device"] = slot.device_name
        resp.headers["X-Model"] = slot.model_name
        return resp

    # --- LLM path ---
    if stream:
        return Response(
            slot.stream_llm(raw_messages, gen, completion_id, created, t0),
            mimetype="text/event-stream",
            headers={"X-Device": slot.device_name, "X-Model": slot.model_name},
        )

    try:
        text = slot.generate_llm(raw_messages, gen)
    except Exception as e:
        print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] LLM error: {e}", flush=True)
        return openai_error(f"Inference failed: {e}", "server_error", 500)

    elapsed = time.perf_counter() - t0
    n_words = len(text.split())
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] "
          f"~{n_words} tokens in {elapsed:.1f}s ({n_words / elapsed:.1f} tok/s)",
          flush=True)

    resp = jsonify({
        "id": completion_id, "object": "chat.completion",
        "created": created, "model": slot.model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    })
    resp.headers["X-Device"] = slot.device_name
    resp.headers["X-Model"] = slot.model_name
    return resp


# ---------------------------------------------------------------------------
# Ollama-compatible API (port 11434)
# ---------------------------------------------------------------------------

ollama_app = Flask("NoLlama-Ollama")
ollama_app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

OLLAMA_PORT = 11434


@ollama_app.route("/")
def ollama_health():
    return "Ollama is running"


@ollama_app.route("/api/version", methods=["GET"])
def ollama_version():
    return jsonify({"version": "nollama-0.1.0"})


@ollama_app.route("/api/tags", methods=["GET"])
def ollama_tags():
    models = []
    for slot in (primary, secondary):
        if slot and slot.status == "ready":
            models.append({
                "name": slot.model_name,
                "model": slot.model_name,
                "size": slot.info.get("size", 0),
                "details": {
                    "family": slot.model_name.split("-")[0],
                    "parameter_size": "",
                    "quantization_level": "int4",
                },
            })
    return jsonify({"models": models})


@ollama_app.route("/api/show", methods=["POST"])
def ollama_show():
    body = request.get_json(silent=True) or {}
    model_name = body.get("model", "")
    for slot in (primary, secondary):
        if slot and slot.model_name == model_name:
            return jsonify({
                "model": model_name,
                "details": {
                    "family": model_name.split("-")[0],
                    "parameter_size": "",
                    "quantization_level": "int4",
                },
                "model_info": {},
            })
    return jsonify({"model": model_name, "details": {}, "model_info": {}})


@ollama_app.route("/api/chat", methods=["POST"])
def ollama_chat():
    if overall_status() != "ready":
        return jsonify({"error": "model not ready"}), 503

    body = request.get_json(silent=True) or {}
    ollama_messages = body.get("messages", [])
    stream = body.get("stream", True)  # Ollama defaults to streaming
    requested_model = body.get("model", "")

    max_tokens = body.get("options", {}).get("num_predict", 2048)
    temperature = body.get("options", {}).get("temperature", 0.0)

    # Translate Ollama messages to internal format
    has_images = False
    internal_messages = []
    for msg in ollama_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        msg_images = msg.get("images", [])

        if msg_images:
            has_images = True
            blocks = [{"type": "text", "text": content}]
            for img_b64 in msg_images:
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })
            internal_messages.append({"role": role, "content": blocks})
        else:
            internal_messages.append({"role": role, "content": content})

    # Parse through same pipeline as OpenAI
    try:
        text_prompt, images, raw_messages = parse_messages(internal_messages, max_dim)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Route to device
    slot = _route_request(has_images, requested_model)
    if slot is None:
        return jsonify({"error": "no model ready"}), 503

    if has_images and slot.model_type == "llm":
        return jsonify({"error": f"model '{slot.model_name}' does not support images"}), 400

    try:
        slot.ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {e}"}), 500

    # Build generation config
    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
    else:
        gen.do_sample = False
        gen.top_k = 1
    gen.repetition_penalty = 1.05

    print(f"\n{datetime.now():%H:%M:%S} <- [{slot.device_name}] [Ollama] "
          f"{'image, ' if has_images else ''}{len(text_prompt)} chars"
          f"{' (stream)' if stream else ''}", flush=True)

    t0 = time.perf_counter()

    # VLM path (no streaming)
    if slot.model_type == "vlm":
        try:
            text = slot.generate_vlm(text_prompt, images, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

        elapsed = time.perf_counter() - t0
        print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
              f"{len(text)} chars in {elapsed:.1f}s", flush=True)

        return jsonify({
            "model": slot.model_name,
            "message": {"role": "assistant", "content": text},
            "done": True,
            "total_duration": int(elapsed * 1e9),
        })

    # LLM path
    if stream:
        return Response(
            _ollama_stream_chat(slot, raw_messages, gen, t0),
            mimetype="application/x-ndjson",
        )

    # Non-streaming LLM
    try:
        text = slot.generate_llm(raw_messages, gen)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    elapsed = time.perf_counter() - t0
    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
          f"~{len(text.split())} tokens in {elapsed:.1f}s", flush=True)

    return jsonify({
        "model": slot.model_name,
        "message": {"role": "assistant", "content": text},
        "done": True,
        "total_duration": int(elapsed * 1e9),
    })


def _ollama_stream_chat(slot, raw_messages, gen, t0):
    """Ollama streaming: newline-delimited JSON (not SSE)."""
    history = ovg.ChatHistory()
    for msg in raw_messages:
        history.append({"role": msg["role"], "content": msg["content"]})

    token_queue = Queue()
    token_count = 0

    def streamer_callback(token):
        if slot._cancel.is_set():
            return True
        token_queue.put(token)
        return False

    def _generate():
        try:
            with slot.lock:
                slot._cancel.clear()
                slot.pipe.generate(history, gen, streamer_callback)
                slot.last_used = time.time()
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] [Ollama] "
                  f"generate error: {e}", flush=True)
        finally:
            token_queue.put(None)

    t = threading.Thread(target=_generate, daemon=True)
    t.start()

    try:
        while True:
            try:
                token = token_queue.get(timeout=120)
            except Empty:
                break
            if token is None:
                break
            token_count += 1
            yield json.dumps({
                "model": slot.model_name,
                "message": {"role": "assistant", "content": token},
                "done": False,
            }) + "\n"

        elapsed = time.perf_counter() - t0
        tps = token_count / elapsed if elapsed > 0 else 0

        yield json.dumps({
            "model": slot.model_name,
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": token_count,
        }) + "\n"
    finally:
        slot._cancel.set()

    print(f"{datetime.now():%H:%M:%S} -> [{slot.device_name}] [Ollama] "
          f"{token_count} tokens in {elapsed:.1f}s ({tps:.1f} tok/s)", flush=True)


@ollama_app.route("/api/generate", methods=["POST"])
def ollama_generate():
    """Single-turn completion (no chat history)."""
    if overall_status() != "ready":
        return jsonify({"error": "model not ready"}), 503

    body = request.get_json(silent=True) or {}
    prompt = body.get("prompt", "")
    stream = body.get("stream", True)
    requested_model = body.get("model", "")
    max_tokens = body.get("options", {}).get("num_predict", 2048)
    temperature = body.get("options", {}).get("temperature", 0.0)

    # Images in generate endpoint
    images_b64 = body.get("images", [])
    has_images = bool(images_b64)

    slot = _route_request(has_images, requested_model)
    if slot is None:
        return jsonify({"error": "no model ready"}), 503

    try:
        slot.ensure_loaded()
    except Exception as e:
        return jsonify({"error": f"Failed to reload model: {e}"}), 500

    gen = ovg.GenerationConfig()
    gen.max_new_tokens = max_tokens
    if temperature and temperature > 0.01:
        gen.do_sample = True
        gen.temperature = temperature
    else:
        gen.do_sample = False
        gen.top_k = 1
    gen.repetition_penalty = 1.05

    t0 = time.perf_counter()

    # VLM with images
    if has_images and slot.model_type == "vlm":
        img_tensors = []
        for b64 in images_b64:
            img = load_image(f"data:image/jpeg;base64,{b64}", max_dim)
            img_tensors.append(pil_to_tensor(img, max_dim))
        try:
            text = slot.generate_vlm(prompt, img_tensors, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        elapsed = time.perf_counter() - t0
        return jsonify({
            "model": slot.model_name,
            "response": text,
            "done": True,
            "total_duration": int(elapsed * 1e9),
        })

    if has_images and slot.model_type == "llm":
        return jsonify({"error": "model does not support images"}), 400

    # Text-only generate → wrap as single-turn chat
    raw_messages = [{"role": "user", "content": prompt}]

    if stream and slot.model_type == "llm":
        return Response(
            _ollama_stream_generate(slot, raw_messages, gen, t0),
            mimetype="application/x-ndjson",
        )

    # Non-streaming
    if slot.model_type == "vlm":
        try:
            text = slot.generate_vlm(prompt, [], gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        try:
            text = slot.generate_llm(raw_messages, gen)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    elapsed = time.perf_counter() - t0
    return jsonify({
        "model": slot.model_name,
        "response": text,
        "done": True,
        "total_duration": int(elapsed * 1e9),
    })


def _ollama_stream_generate(slot, raw_messages, gen, t0):
    """Ollama /api/generate streaming."""
    history = ovg.ChatHistory()
    for msg in raw_messages:
        history.append({"role": msg["role"], "content": msg["content"]})

    token_queue = Queue()
    token_count = 0

    def streamer_callback(token):
        if slot._cancel.is_set():
            return True
        token_queue.put(token)
        return False

    def _generate():
        try:
            with slot.lock:
                slot._cancel.clear()
                slot.pipe.generate(history, gen, streamer_callback)
                slot.last_used = time.time()
        except Exception as e:
            print(f"{datetime.now():%H:%M:%S} !! [{slot.device_name}] [Ollama] "
                  f"generate error: {e}", flush=True)
        finally:
            token_queue.put(None)

    t = threading.Thread(target=_generate, daemon=True)
    t.start()

    try:
        while True:
            try:
                token = token_queue.get(timeout=120)
            except Empty:
                break
            if token is None:
                break
            token_count += 1
            yield json.dumps({
                "model": slot.model_name,
                "response": token,
                "done": False,
            }) + "\n"

        elapsed = time.perf_counter() - t0
        yield json.dumps({
            "model": slot.model_name,
            "response": "",
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": token_count,
        }) + "\n"
    finally:
        slot._cancel.set()


# Stubs — clients expect these to exist
@ollama_app.route("/api/pull", methods=["POST"])
def ollama_pull():
    return jsonify({"status": "success"})


@ollama_app.route("/api/delete", methods=["DELETE"])
def ollama_delete():
    return "", 200


@ollama_app.route("/api/copy", methods=["POST"])
def ollama_copy():
    return "", 200


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def check_port(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False


def detect_devices():
    """Return dict of available devices: {name: full_name}.

    Non-Intel GPUs are filtered out: OpenVINO's intel_gpu plugin enumerates
    any OpenCL-capable GPU (NVIDIA, AMD), but its kernels only run on Intel
    hardware. Selecting a non-Intel GPU produces hundreds of compile errors
    and crashes at warmup with CL_INVALID_VALUE — better not to offer it.
    """
    devices = {}
    core = ov.Core()
    for d in core.get_available_devices():
        try:
            full_name = core.get_property(d, "FULL_DEVICE_NAME")
        except Exception:
            full_name = d
        if d.startswith("GPU") and "intel" not in full_name.lower():
            continue
        devices[d] = full_name
    return devices


def _idle_watchdog(slots, idle_timeout, check_interval=30):
    """Background thread: unload slots that have been idle too long."""
    while True:
        time.sleep(check_interval)
        now = time.time()
        for slot in slots:
            if not slot or slot.status != "ready":
                continue
            if now - slot.last_used < idle_timeout:
                continue
            # Try non-blocking lock acquire — skip if a request is in progress
            if not slot.lock.acquire(blocking=False):
                continue
            try:
                slot.unload()
            finally:
                slot.lock.release()


_banner_lock = threading.Lock()
_banner_printed = False


def _load_in_background(slot, model_dir, devices, port, ollama_port, banner_slots):
    """Background thread: load model + warmup on one device."""
    global _banner_printed
    try:
        slot.device_full = devices.get(slot.device_name, slot.device_name)
        slot.load(model_dir)
        slot.warmup()
    except Exception as e:
        slot.status = "error"
        print(f"\n  [{slot.device_name}] ERROR: Failed to load model: {e}")
        print(f"  Is another process using the {slot.device_name}?", flush=True)

    # Print banner when all slots are done — only one thread wins
    with _banner_lock:
        if _banner_printed:
            return
        all_done = all(
            s.status in ("ready", "error", "not_configured")
            for s in banner_slots
        )
        if not all_done:
            return
        _banner_printed = True

    if any(s.status == "ready" for s in banner_slots):
        lines = []
        for s in banner_slots:
            if s.status == "ready":
                lines.append(f"    {s.device_name:5s}: {s.model_name} ({s.model_type.upper()}) "
                             f"-- {s.device_full}")
        url = f"http://localhost:{port}"
        api_lines = [f"    API  : {url}  (OpenAI)"]
        if ollama_port:
            api_lines.append(f"    API  : http://localhost:{ollama_port}  (Ollama)")
        print(f"""
================================================
  NoLlama ready
{chr(10).join(lines)}
{chr(10).join(api_lines)}
================================================
""", flush=True)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    default_model = str(Path(__file__).parent / "model")
    p.add_argument("--model-dir", default=default_model,
                   help="Primary model directory (default: model/)")
    p.add_argument("--device", default="auto",
                   help="Device for primary model: NPU, GPU, CPU, or auto (default: auto)")
    p.add_argument("--gpu-model-dir", default=None,
                   help="Secondary GPU model (enables dual mode: NPU chat + GPU vision/LLM)")
    p.add_argument("--port", type=int, default=8000,
                   help="OpenAI API port (default: 8000)")
    p.add_argument("--ollama-port", type=int, default=11434,
                   help="Ollama API port (default: 11434, 0 to disable)")
    p.add_argument("--max-dim", type=int, default=768,
                   help="Max image dimension before resize (default: 768)")
    p.add_argument("--whisper-dir", default=None,
                   help="Whisper model directory for speech-to-text (enables /v1/audio/transcriptions)")
    p.add_argument("--whisper-device", default="CPU",
                   help="Device for Whisper: CPU or GPU (default: CPU)")
    p.add_argument("--idle-timeout", type=int, default=1800,
                   help="Change idle-unload timeout in seconds "
                        "(default: 1800 = 30 min). Use 0 to disable unloading.")
    return p.parse_args()


def main():
    global primary, secondary, whisper_slot, max_dim

    args = parse_args()
    model_dir = os.path.expanduser(args.model_dir)
    max_dim = args.max_dim

    print(flush=True)

    # 1. Check ports
    if not check_port(args.port):
        print(f"ERROR: Port {args.port} is already in use.")
        print(f"Use --port <number> to pick another port.")
        sys.exit(1)
    if args.ollama_port and not check_port(args.ollama_port):
        print(f"WARNING: Ollama port {args.ollama_port} is in use. "
              f"Ollama API disabled. (Is Ollama already running?)")
        args.ollama_port = 0

    # 2. Detect devices
    devices = detect_devices()
    print("  Devices:", flush=True)
    for d, name in devices.items():
        print(f"    {d}: {name}")
    print()

    # 3. Resolve primary device
    device = args.device.upper()
    if device == "AUTO":
        if args.gpu_model_dir:
            # Dual mode: primary goes on NPU (or CPU if no NPU)
            device = "NPU" if "NPU" in devices else "CPU"
        elif "NPU" in devices:
            device = "NPU"
        elif "GPU" in devices:
            device = "GPU"
        else:
            device = "CPU"

    if device not in devices and device != "CPU":
        print(f"ERROR: Device {device} not available. Found: {list(devices.keys())}")
        sys.exit(1)

    # 4. Verify model directories
    if not os.path.isdir(model_dir):
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)
    if args.gpu_model_dir and not os.path.isdir(args.gpu_model_dir):
        print(f"ERROR: GPU model directory not found: {args.gpu_model_dir}")
        sys.exit(1)
    if args.whisper_dir and not os.path.isdir(args.whisper_dir):
        print(f"ERROR: Whisper model directory not found: {args.whisper_dir}")
        sys.exit(1)

    # 5. Create device slots
    primary = DeviceSlot(device)
    all_slots = [primary]

    if args.gpu_model_dir:
        if "GPU" not in devices:
            print("WARNING: --gpu-model-dir given but no GPU detected. Ignoring.")
        else:
            secondary = DeviceSlot("GPU")
            all_slots.append(secondary)

    if args.whisper_dir:
        whisper_device = args.whisper_device.upper()
        if whisper_device not in devices and whisper_device != "CPU":
            print(f"WARNING: Whisper device {whisper_device} not available, falling back to CPU.")
            whisper_device = "CPU"
        whisper_slot = WhisperSlot(whisper_device)
        all_slots.append(whisper_slot)

    # 6. Start Flask, load models in background
    ports_msg = f"port {args.port}"
    if args.ollama_port:
        ports_msg += f" + Ollama on {args.ollama_port}"
    print(f"  Starting server on {ports_msg}...", flush=True)

    threads = []
    t = threading.Thread(
        target=_load_in_background,
        args=(primary, model_dir, devices, args.port, args.ollama_port, all_slots),
        daemon=True,
    )
    threads.append(t)

    if secondary:
        t2 = threading.Thread(
            target=_load_in_background,
            args=(secondary, args.gpu_model_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(t2)

    if whisper_slot:
        tw = threading.Thread(
            target=_load_in_background,
            args=(whisper_slot, args.whisper_dir, devices, args.port,
                  args.ollama_port, all_slots),
            daemon=True,
        )
        threads.append(tw)

    for t in threads:
        t.start()

    # Idle watchdog — unload models after inactivity
    if args.idle_timeout > 0:
        print(f"  Idle unload after {args.idle_timeout}s of inactivity", flush=True)
        watchdog = threading.Thread(
            target=_idle_watchdog,
            args=(all_slots, args.idle_timeout),
            daemon=True,
        )
        watchdog.start()

    # Suppress Flask's default "Serving Flask app" banner — we have our own
    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    # Start Ollama API on separate port in background thread
    if args.ollama_port:
        print(f"  Ollama API on port {args.ollama_port}", flush=True)
        def _run_ollama():
            try:
                ollama_app.run(
                    host="0.0.0.0", port=args.ollama_port, threaded=True,
                )
            except Exception as e:
                print(f"  WARNING: Ollama API failed to start: {e}", flush=True)
        ollama_thread = threading.Thread(target=_run_ollama, daemon=True)
        ollama_thread.start()

    # OpenAI API on main thread
    print(f"  OpenAI API on port {args.port}", flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
