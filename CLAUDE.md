# NoLlama

OpenAI-compatible LLM/VLM server for Intel hardware. NPU-first.

## Architecture

- `nollama.py` — Flask server, DeviceSlot class per device, auto-detects VLM/LLM from config.json
- NPU: LLMPipeline with MAX_PROMPT_LEN=4096, streaming via SSE
- GPU: VLMPipeline (images) or LLMPipeline (text), no streaming for VLM (OpenVINO limitation)
- Whisper: WhisperSlot + WhisperPipeline for STT, `POST /v1/audio/transcriptions`, CPU or GPU
- OpenVINO GenAI may unify VLM/LLMPipeline — when that happens, simplify the dual-pipeline routing
- Routing: images go to GPU, text goes to NPU (or GPU if no NPU)
- Web UI: `templates/index.html` + `static/css/style.css` + `static/js/app.js`
- Collapsible `<think>` blocks, "Just answer me, dammit!" button, temperature slider
- `threaded=True` on Flask, concurrency via per-device locks
- `models.json` — curated model registry (npu, gpu_vlm, gpu_llm, whisper categories)
- `install.ps1` detects devices, shows model menu, generates `start.ps1`

## Environment

- Windows 11, Python 3.10+
- Intel Core Ultra (NPU) + Intel ARC 140V 16GB (GPU)
- OpenVINO 2026.1+ with openvino_genai
- venv in `venv/`, activate before running

## Development preferences

- Keep it simple. One file (`nollama.py`) is fine. Don't split into modules unless it gets unwieldy.
- PowerShell for install/launch scripts (Windows-native users).
- Runtime flags over hardcoded config (e.g. `--port`, `--device`).
- When testing, use small payloads / short prompts. Don't run full model loads unless needed.
- VLM prompts must be dead simple for small models (3B). One question, one answer, minimal JSON. All logic in Python, not in the prompt.
- Qwen3-VL is now pre-exported by Intel (OpenVINO/Qwen3-VL-8B-Instruct-int4-ov, May 2026) — not yet tested here. Earlier note about optimum-intel support is obsolete.

## Known issues

- NPU default prompt limit is 1024 tokens — we override to MAX_PROMPT_LEN=4096
- VLMPipeline has no streaming support (OpenVINO limitation)
- Qwen3 thinking models can exhaust token budget on `<think>` before producing an answer
- Cancel (`/v1/cancel`) relies on OpenVINO invoking the streamer callback. If the native code blocks without yielding, cancel won't take effect — generation completes naturally.
- Chat history unbounded in web UI — user clears with Ctrl+N when long sessions approach MAX_PROMPT_LEN

## Verified models

- Qwen3-8B (INT4-CW) on NPU — recommended, needs MAX_PROMPT_LEN=4096
- Phi 3.5 Mini (INT4-CW) on NPU — smaller, faster
- DeepSeek-R1-1.5B (INT4-CW) on NPU — works but terrible quality (testing only)
- Gemma 3 4B Vision (INT4) on GPU — fast VLM
- Qwen2.5-VL-3B/7B (INT4/INT8) on GPU — proven for image tasks
- Qwen3-30B-A3B on GPU — needs >16GB VRAM, falls back to CPU silently on 16GB cards
