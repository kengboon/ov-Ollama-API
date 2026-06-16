# NoLlama

**Local LLM server for the full Intel stack.** NPU, ARC iGPU, ARC discrete, CPU.
OpenAI + Ollama APIs. One server, every Intel device.

No NVIDIA required. No Ollama install. No llama.cpp. **No problem.**

Runs on Intel Core Ultra laptops (NPU + ARC iGPU), desktops with ARC
discrete GPUs (A770, B580), or any Intel CPU. Automatically detects your
hardware, picks the best device, and exposes both OpenAI and Ollama
compatible APIs — so any client that speaks to either just works.

![NoLlama in action](docs/images/nollama-demo.gif)

## Quick start

```powershell
.\install.ps1
.\start.ps1
```

That's it. `install.ps1` detects your hardware, lets you pick a model,
downloads it, and generates `start.ps1`. The launcher waits for the
model to load (with a progress indicator), then opens the built-in
chat UI in your browser at http://localhost:8000.

## Recommended models

New here, or re-running `install.ps1`? This is the proven pairing for a
Core Ultra laptop (NPU + ARC iGPU) — just pick these two in the menu:

| Role | Pick in the menu | HuggingFace | Size |
|---|---|---|---|
| **NPU chat** | Qwen3 8B (INT4-CW) | `OpenVINO/Qwen3-8B-int4-cw-ov` | ~5 GB |
| **GPU vision** | Qwen3-VL 8B (INT8) | `OpenVINO/Qwen3-VL-8B-Instruct-int8-ov` | ~9 GB |

Qwen3 8B is the best-quality text model verified on the NPU. Qwen3-VL 8B
is the matching vision model — the INT8 build keeps fine detail (OCR,
small numbers) and fits a 16 GB ARC; drop to the ~6 GB INT4 build
(`…-int4-ov`) if you're tight on VRAM. Both are pre-exported (install
instantly, no conversion), and returning users see them flagged
**"Already on disk"** in the menu.

## What it does

- **OpenAI API** (`/v1/chat/completions`) — works with any OpenAI client, OpenWebUI, etc.
- **Ollama API** (`/api/chat`, `/api/generate`) — works with Ollama clients, OpenWebUI Ollama mode, etc.
- **Auto-detects** NPU, ARC iGPU, ARC discrete, CPU — picks the best available
- **VLM support** — send images via base64 or `file://` URIs for vision models
- **Streaming** — token-by-token for text chat, with collapsible thinking blocks
- **Dual device** — NPU for chat + GPU for vision, simultaneously
- **Built-in web UI** — chat, image drop zone, model selector, dark theme
- **Model menu** — curated list of verified models, no conversion nightmares

## Web UI

The server includes a built-in chat interface at http://localhost:8000.
No separate install, no Docker, no Node.js.

![NoLlama chat UI](docs/images/nollama-chat.gif)

A native Windows GUI is planned to replace the browser-based UI.

Features:
- Streaming chat with tokens appearing in real-time
- Collapsible "Thinking..." blocks (Qwen3 reasoning models)
- Drag-and-drop / paste images for VLM queries
- Model selector showing loaded models and their devices
- Device badge on each response (`[NPU 1.2s]`, `[GPU 2.8s]`)
- Dark theme
- Keyboard shortcuts: Enter to send, Shift+Enter for newline,
  Ctrl+V to paste images, Ctrl+N for new chat, Escape to cancel

## Device support

| Device | Examples | What it does | Streaming? |
|---|---|---|---|
| NPU (Intel AI Boost) | Core Ultra 7 258V | Text chat via LLMPipeline. Low power, sustained workload sweet spot. | Yes |
| ARC iGPU | ARC 140V (Core Ultra) | Vision + text, or bigger LLM | Yes (VLM streams in 2026.1+) |
| ARC discrete | A770, B580 | Same as iGPU, more VRAM for larger models | Yes (VLM streams in 2026.1+) |
| CPU | Any Intel CPU | Fallback for everything. On desktops with DDR5 and many cores, often *faster* than NPU — see benchmarks. | Yes |

### Intel only — by design

NoLlama is Intel-hardware-only and will stay that way. Non-Intel GPUs
(NVIDIA, AMD) are filtered out of device detection on purpose, even
though OpenVINO 2026 now ships an experimental NVIDIA plugin via
[`openvino-extensibility`](https://docs.openvino.ai/2026/documentation/openvino-extensibility/openvino-plugin-library/plugin.html).
That path drags CUDA/cuDNN into the stack — it's a developer-backend
extension, not a drop-in user feature, and it loses every reason
NoLlama exists in the first place (NPU-first, Intel-first, no CUDA).

If you have an NVIDIA GPU, **use Ollama**. Ollama will always do
Ollama better than NoLlama could, and that's the right tool for that
hardware. NoLlama's value is specifically the Intel NPU / ARC story
that Ollama doesn't tell.

### Benchmark (Core Ultra 7 258V, ARC 140V 16 GB) — laptop, LPDDR5X

Tested with `benchmark.py` — 1 warmup + 5 runs, outliers discarded.

```powershell
# Text-only (no images required)
python benchmark.py --llm-only

# With VLM tests — provide 4 images: two "same vehicle" + two "different"
python benchmark.py --images-dir C:\path\to\images
python benchmark.py --same-1 a.jpg --same-2 b.jpg --diff-1 c.jpg --diff-2 d.jpg
```

**LLM text (Qwen3 8B INT4-CW, same model on NPU and CPU):**

| Test | NPU | CPU |
|---|---|---|
| "Say hello" (thinking) | 11.7s, 5.2 tok/s | 8.1s, 7.4 tok/s |
| "Say hello" (no-think) | 10.6s, 4.6 tok/s | 8.6s, 7.3 tok/s |
| "What is 2+2?" (thinking) | 11.7s, 5.3 tok/s | 9.0s, 7.0 tok/s |
| "What is 2+2?" (no-think) | 5.5s, 0.7 tok/s | 2.7s, 1.5 tok/s |

**GPU (Qwen2.5-VL 3B on ARC 140V, non-streaming):**

| Test | Time |
|---|---|
| "Say hello" (thinking) | 2.6s |
| "Say hello" (no-think) | 2.6s |
| "What is 2+2?" (thinking) | 2.6s |
| "What is 2+2?" (no-think) | 2.4s |
| Same vehicle? (2 images) | 3.8s |
| Different vehicles? (2 images) | 3.8s |

Above benchmarks were captured before VLMPipeline gained streaming
support (openvino-genai 2026.1). VLM now streams on Arc 140V at
roughly 11 tok/s decode after prefill — see
`benchmark.py --backend vlm` for fresh numbers.

CPU beats NPU on throughput (~7.4 vs ~5.2 tok/s) for this model.
GPU text is fast but runs a smaller 3B model (not directly comparable).
VLM image responses take ~3-4s regardless of answer length.

### Benchmark (Core Ultra 9 285K, RTX 5090) — desktop, DDR5

Same Qwen3 8B INT4-CW model on every Intel device, plus the same model
served via Ollama (GGUF Q4_K_M) on the RTX 5090 for context. 1 warmup +
3 runs. The "count 1-100" test (`max_tokens=4096`, no-think) is the
cleanest cross-stack number — long output, steady-state, no thinking confound.

```powershell
# Each NoLlama device — restart the server with --device <name> first
python benchmark.py --label npu --runs 3 --llm-only
python benchmark.py --label igpu --runs 3 --llm-only
python benchmark.py --label cpu --runs 3 --llm-only

# Ollama (any backend it's running on — CUDA, ROCm, CPU)
python benchmark.py --backend ollama --model qwen3:8b --label rtx5090 --runs 3 --llm-only
```

**Decode throughput, count-1-100 test:**

| Backend | Device | TTFT | Decode tok/s | Speed vs CPU |
|---|---|---|---|---|
| Ollama (GGUF/CUDA) | RTX 5090 | 0.19s | 197 | 11.1× |
| NoLlama (OpenVINO) | CPU (8P + 16E @ DDR5) | 3.84s | 17.8 | 1.0× |
| NoLlama (OpenVINO) | iGPU (Xe-LPG, 4 cores) | 4.01s | 15.4 | 0.87× |
| NoLlama (OpenVINO) | NPU 3 (Intel AI Boost) | 10.6s | 10.0 | 0.56× |

**Surprises on this hardware:**

- **CPU beats iGPU.** Arrow Lake's 285K (8P + 16E at high clocks) plus
  OpenVINO's tuned INT4 CPU kernels add up to more decode throughput
  than the small Xe-LPG iGPU (only 4 Xe cores on the desktop part —
  the laptop's ARC 140V has 8). Both share the same DDR5 pool, so the
  iGPU has no bandwidth advantage, only a compute disadvantage.
- **NPU is the slowest Intel device on desktop**, opposite of the laptop
  story. NPU's value is power efficiency (laptop on battery), not
  throughput on mains.
- **Prefill scales differently than decode.** RTX 5090's TTFT advantage
  over NPU is ~55× (0.19s vs 10.6s); its decode advantage is ~20×.
  Long prompts amplify the gap.
- **The dGPU dominates** — if you have one, use it. NoLlama's CPU
  fallback is good for "Intel-only laptop on battery", not for
  competing with a discrete card.

**Why the desktop iGPU/NPU are slower than the laptop's:**
LPDDR5X-8533 (laptop, ~136 GB/s) vs DDR5-6400 dual-channel (desktop,
~100 GB/s). Decode throughput on INT4 LLMs is memory-bandwidth-bound,
so the laptop's faster system memory closes some of the gap that
silicon size alone would suggest. (The Core Ultra 7 258V Lunar Lake
NPU also has more compute units than the 285K Arrow Lake NPU.)

**Practical guidance:**

| Hardware | Best NoLlama device |
|---|---|
| Intel Core Ultra laptop (Lunar Lake) | NPU (efficiency) or ARC 140V iGPU |
| Intel Arrow Lake desktop, no dGPU | **CPU** — surprisingly best |
| Intel + ARC discrete (A770, B580) | ARC discrete |
| Intel + NVIDIA discrete | Use Ollama for the dGPU; NoLlama on CPU/NPU/iGPU as fallback |

### Dual mode (NPU + GPU)

When you have both, text requests go to the NPU (streaming) and image
requests go to the GPU (VLM). Or put a bigger LLM on the GPU for
smarter chat. The routing is automatic — send a request and the right
device handles it.

```
POST /v1/chat/completions
  "What is the capital of Norway?"  --> NPU (streaming)
  [image + "What vehicle is this?"] --> GPU (VLM)
```

## Why not OpenVINO Model Server (OVMS)?

Intel already ships OVMS — a production-grade OpenVINO inference server.
If you're deploying LLMs in a datacenter or on Kubernetes, use OVMS.
NoLlama is a different target: your laptop.

| | OVMS | NoLlama |
|---|---|---|
| Target | Production, datacenter, K8s | Laptop, desktop, local |
| Runtime | C++ | Python (Flask) |
| OpenAI API | Yes (recent versions) | Yes |
| Ollama API | No | **Yes** |
| Built-in web UI | No (add OpenWebUI) | **Yes** |
| Auto device detection | No | **Yes** |
| Dual-device routing | One model per instance | **NPU chat + GPU vision, simultaneously** |
| Config | JSON, manual | Zero — `install.ps1` and go |

OVMS is a proper inference server. NoLlama is the thing that makes
your Core Ultra feel like Ollama already ran on it.

## Usage

```powershell
# Auto-detect (picks best device)
python nollama.py

# Force a specific device
python nollama.py --device NPU
python nollama.py --device GPU
python nollama.py --device CPU

# Dual mode: NPU chat + GPU vision
python nollama.py --model-dir model --gpu-model-dir gpu-model

# Different port
python nollama.py --port 9000

# Change the default idle-unload timeout (default is 1800 = 30 min)
python nollama.py --idle-timeout 600     # unload after 10 min idle
python nollama.py --idle-timeout 0       # never unload — keep models loaded forever
```

### Idle unload

NoLlama frees model memory after **30 minutes of inactivity by default**
(an 8B INT4 model holds ~5 GB of RAM; a VLM another ~3 GB). The next
request automatically reloads the model — the client just sees a slow
first response (~30-60s for an 8B model on NPU). The web UI shows
"Reloading model..." while it waits.

Change with `--idle-timeout <seconds>`. Use `0` to keep models loaded
forever (the old behavior).

`/health` reports `idle_unloaded` slots; the overall status stays
`ready` because requests can still be served (with a reload).

## API

Standard OpenAI `/v1/chat/completions`. Works with any OpenAI client.

### Text chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello!"}]}'
```

### Image (VLM, requires GPU with vision model)

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages":[{"role":"user","content":[
      {"type":"text","text":"What is in this image?"},
      {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}
    ]}]
  }'
```

### Local file shortcut

When client and server are on the same machine, skip base64:

```python
{"type": "image_url", "image_url": {"url": "file:///C:/path/to/image.jpg"}}
```

**Note:** `file://` URIs only work locally. Remote clients must use base64.

### Streaming

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Tell me a story"}],"stream":true}'
```

### Other endpoints

- `GET /health` — device status, model names, readiness
- `GET /v1/models` — list loaded models (OpenAI format)

### Response headers

Every response includes `X-Device` and `X-Model` headers so you can
see which device handled it:

```
X-Device: NPU
X-Model: qwen3-8b
```

## Using with the openai Python package

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen3-8b",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="")
```

## Ollama API

NoLlama also serves a full Ollama-compatible API on port 11434 (the
Ollama default). Any tool or client that talks to Ollama works without
modification — it thinks it's talking to a real Ollama instance.

Supported endpoints:

- `POST /api/chat` — chat with streaming (newline-delimited JSON)
- `POST /api/generate` — single-turn completion
- `GET /api/tags` — list models
- `POST /api/show` — model info

```bash
curl http://localhost:11434/api/chat \
  -d '{"model":"qwen3-8b-int4-cw","messages":[{"role":"user","content":"Hello!"}]}'
```

Disable with `--ollama-port 0` if you don't need it or port 11434 is taken.

## Using with OpenWebUI

OpenWebUI can connect via either API:

**OpenAI mode** (recommended):

| Field | Value |
|---|---|
| Base URL | `http://host.docker.internal:8000/v1` |
| API Key | `not-needed` |

**Ollama mode** (no config needed if NoLlama runs on default port):

| Field | Value |
|---|---|
| Ollama Base URL | `http://host.docker.internal:11434` |

## Models

`install.ps1` shows a curated menu of models known to work on Intel
hardware. All pre-exported models are download-only (no conversion).
The menu is defined in `models.json` — add entries when new models
are verified.

### Adding models outside the menu

Use `download-model.ps1` to grab any HuggingFace model:

```powershell
# Pre-exported OpenVINO model (just download)
.\download-model.ps1 OpenVINO/Qwen3-8B-int4-cw-ov

# Convert a HuggingFace model to OpenVINO
.\download-model.ps1 Qwen/Qwen2.5-VL-3B-Instruct --convert --weight int8

# With trust-remote-code (some models require this)
.\download-model.ps1 Qwen/Qwen2.5-VL-3B-Instruct --convert --weight int4 --trust
```

Models download to `~/models/<name>/`. Point NoLlama at them:

```powershell
python nollama.py --model-dir ~/models/my-model --device GPU
python nollama.py --gpu-model-dir ~/models/my-vlm
```

### Finding newer/better models

The model menus rot fast — new architectures appear monthly. The
authoritative place to look is the OpenVINO org on HuggingFace:

**[huggingface.co/OpenVINO](https://huggingface.co/OpenVINO)**

These are pre-exported by Intel, so they install instantly (no
conversion). What to look for:

| Suffix | Where it runs | Notes |
|---|---|---|
| `-int4-cw-ov` | NPU + GPU | Channel-wise INT4. NPU's preferred format. |
| `-int4-ov` | GPU only | Standard INT4. Not always NPU-compatible. |
| `-int8-ov` | GPU + CPU | Better fine-detail retention than INT4 (OCR, numbers). |
| `-fp16-ov` | GPU + CPU | Full precision. Largest, slowest, sharpest. |

Quick rules of thumb:
- **NPU chat:** must be `-int4-cw-ov` and ≤ ~10 GB.
- **GPU vision (VLM):** any `-int4-ov` or `-int8-ov` model marked
  "Image-Text-to-Text" on HF.
- **GPU LLM (smarter than NPU):** any `-int4-ov` model up to your
  VRAM. Above ~16 GB falls back to CPU silently.
- **Whisper (STT):** OpenVINO ships pre-quantized whisper variants
  (`whisper-{tiny,base,small,medium,large-v3}-{int4,int8,fp16}-ov`).

Once a model proves itself, add it to `models.json` so it appears in
the install menu. Keep "Untested" tags on entries that haven't been
verified yet — be honest about what's measured vs. assumed.

> **Notable as of May 2026:** OpenVINO ships
> [Qwen3-VL-8B](https://huggingface.co/OpenVINO/Qwen3-VL-8B-Instruct-int4-ov)
> pre-exported in INT4/INT8/FP16. This is the natural vision sibling
> to the proven Qwen3-8B NPU chat model and is the recommended VLM in
> `models.json` after a 2026-05-26 QA pass on Xe-LPG iGPU.

### NPU models (chat)

| Model | Size | Notes |
|---|---|---|
| Qwen3 8B (INT4-CW) | ~5 GB | Recommended. Best quality. |
| Phi 3.5 Mini (INT4-CW) | ~2 GB | Smaller, faster. |
| DeepSeek R1 Distill 7B (INT4-CW) | ~4 GB | Reasoning. |
| DeepSeek R1 Distill 1.5B (INT4-CW) | ~1 GB | Testing only. |
| Mistral 7B v0.3 (INT4-CW) | ~4 GB | General purpose. |

### GPU vision models

| Model | Size | Notes |
|---|---|---|
| Qwen3-VL 8B (INT8) | ~9 GB | Recommended pairing for 16 GB ARC. Keeps fine detail (OCR, numbers). |
| Qwen3-VL 8B (INT4) | ~6 GB | Lighter alternative. Newer Qwen-VL generation; verified on Xe-LPG. |
| Qwen2.5-VL 3B (INT8, convert) | ~4 GB | Proven. INT8 better at fine detail (OCR, numbers). |
| Gemma 3 4B Vision (INT4) | ~3 GB | Untested. |
| Gemma 3 12B Vision (INT4) | ~7 GB | Untested. Needs ~12 GB RAM with KV cache. |
| InternVL2 4B (INT4) | ~3 GB | Untested. |
| Phi 3.5 Vision (INT4) | ~3 GB | Untested. |

### GPU large LLMs (smarter than NPU)

| Model | Size | Notes |
|---|---|---|
| Qwen3 14B (INT4) | ~8 GB | Great reasoning. |
| Qwen3 30B-A3B MoE (INT4) | ~17 GB | 30B brain, 3B speed. |
| Phi 4 (INT4) | ~8 GB | Strong reasoning. |
| Phi 4 Reasoning (INT4) | ~8 GB | Chain-of-thought. |

## How it works

The server auto-detects your model type (VLM or LLM) from
`config.json` and loads the right OpenVINO GenAI pipeline:

- **VLMPipeline** for vision models — handles images + text
- **LLMPipeline** for text models — handles chat with streaming

In dual mode, both pipelines run on separate devices with separate
locks. They don't interfere with each other.

> **Future simplification:** OpenVINO GenAI may unify VLMPipeline and
> LLMPipeline into a single pipeline that handles both text and images.
> When that lands, the dual-pipeline detection and routing logic in
> NoLlama can be collapsed into one code path.

## Files

```
nollama.py              The server
install.ps1             Setup wizard
download-model.ps1      Download/convert any HuggingFace model
benchmark.py            Device performance benchmark
start.ps1               Auto-generated launcher (after install)
models.json             Curated model registry
model/                  Primary model (NPU or GPU)
gpu-model/              Secondary GPU model (dual mode)
venv/                   Python virtual environment
```

`model/`, `gpu-model/`, `venv/`, and `start.ps1` are gitignored.
The repo is pure code.

## Requirements

- PowerShell 7+ (Windows PowerShell 5.1 is not supported; on Linux,
  see [Microsoft's install
  instructions](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-linux))
- Python 3.10+
- OpenVINO 2026.1+ with openvino-genai
- At least one of:
  - Intel Core Ultra (NPU + ARC iGPU)
  - Intel ARC discrete GPU (A770, B580, etc.)
  - Any Intel CPU (slower, but works)
- ~1-17 GB disk per model

`install.ps1` handles the venv, dependencies, and model download.
It runs on Windows, Linux, and (untested) macOS — paths and link
creation are branched on `$IsWindows`. The Windows path is the
primary tested platform; Linux is informally supported.

## Known limitations

These are known and intentionally not fixed — either because the cause
is upstream, the fix would hurt simplicity, or it doesn't matter for a
local single-user tool.

- **Cancel may not interrupt mid-generation.** The cancel endpoint
  signals OpenVINO's streamer callback to stop. If OpenVINO is blocked
  inside a native call and not invoking the callback, there's no way
  to interrupt it from Python. Generation completes; lock releases
  when it does.
- **NPU prompt limit is 4096 tokens.** Long chat histories will
  eventually exceed this. The UI doesn't trim history — use Ctrl+N to
  start fresh if you hit the limit.
- **Vision runs on the GPU, not the NPU — by design.** The NPU *can*
  load a VLM (Qwen2.5-VL-3B compiles and runs via VLMPipeline; Qwen3.5
  and MiniCPM-V don't compile at all), but the NPU caps the prompt at
  ~1024 tokens *including image tokens*, and Qwen2.5-VL spends one token
  per 28×28 px. That leaves a usable ceiling around **768×768 (~784
  image tokens)**: at that size — or smaller — it answers correctly, so
  NPU vision works **well-ish on very small images** (a 256–512px crop is
  fine). But prefill already takes ~17s at the ceiling, and a plain
  1024×768 photo overflows the cap and fails outright (720p/1080p never
  stand a chance). So vision stays on the GPU, which has no such cap,
  runs at full resolution, and is faster. Measured with
  `test_npu_vlm_imagesize.py`.
- **Ollama management endpoints are stubs.** `/api/pull`, `/api/delete`,
  `/api/copy` return success but don't do anything. Model management is
  via `install.ps1` or `download-model.ps1`, not the API.
- **No graceful shutdown.** Ctrl+C is abrupt. If you hit it mid-load,
  NPU/GPU resources may not free cleanly — usually resolves on next
  launch, occasionally needs a reboot.
- **Flask dev server, not production.** Single-user local tool. Don't
  put it on the internet without a reverse proxy.

## A note about small models

During initial NPU testing with DeepSeek R1 1.5B, we asked:
"What is the capital of Norway?"

The model's response:

> "I need to figure out the capital of Norway. I know it's a country
> in Norway. I remember that Norway is a small island..."

Norway is, in fact, not a small island.

Or *is* it? To paraphrase the greatest detective of all time, Ford
Fairlane: "...an island in an ocean of diarrhea."

The point: 1.5B parameter models are for testing the plumbing, not
for geography. Use Qwen3-8B or larger for actual chat. The small
models will catch up — they're getting smarter every month.

## License

MIT

## Author

Tommy Leonhardsen
