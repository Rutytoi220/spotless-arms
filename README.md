# Spotless-Arms — Pipelined Vision-Language-Action (VLA) Engine

**Spotless-Arms** is a low-latency, CPU-pipelined Vision-Language-Action (VLA) engine for Linux Wayland / Hyprland. Built on 4-bit `openbmb/MiniCPM-V-2_6` (`Qwen2Model`, `hidden_size=3584`) and a recurrent **Adaptive-$K$ `SpotlessLatentAdapter`**, it bypasses autoregressive JSON text decoding and drives the Linux cursor directly via a continuous 6-DoF `MotorHead` and persistent `/dev/uinput` kernel device in under 1 ms.

---

## Architecture

```text
Stage 1 (Background Thread)   AsyncScreenBuffer: grim -t jpeg -q 80 - + proportional 1280px downscale
Stage 1.5 (Overlapped Prep)   ThreadPoolExecutor: tokenize + slice frame_{t+1} while GPU runs tick_t
Stage 2 (CUDA bfloat16/NF4)   SpotlessVLA.forward_motor(): SigLIP (bf16) -> Prefill -> Adaptive-K (4->1) -> MotorHead [1, 6]
Stage 3 (Kernel Actuator)     LinuxVirtualArm: persistent evdev.UInput absolute pointer (<0.3 ms) + auto_halt_on_click
```

### Core Modules
- `spotless_arms/models/latent_core.py`: 4-bit NF4 `MiniCPM-V-2_6` loader (`llm_int8_skip_modules=["vpm", "resampler"]`, TF32 enabled) + `SpotlessLatentAdapter` (`checkpoints/best_vlm_adapter.pt`).
- `spotless_arms/models/motor_head.py`: `MotorHead(3584 -> 512 -> 6)` outputting `[x_norm, y_norm, move_logit, left_click_logit, right_click_logit, done_logit]` directly from `RMSNorm(h_verb)`, plus dual-mode `SpotlessVLA` (`forward_motor` & `forward_text`).
- `spotless_arms/actuators/uinput_arm.py`: Zero-subprocess `LinuxVirtualArm` via `evdev.UInput` (automatic `ydotool` / `dry_run` fallback).
- `spotless_arms/pipeline/pipelined_runner.py`: Overlapped 3-stage reflex loop with Adaptive-$K$ (`k_initial=4` on Tick 1, `k_track=1` on Ticks 2+) and single-click `auto_halt_on_click`.
- `spotless_arms/train_motor_head.py`: `<8 GB` VRAM trainer freezing VLM + adapter to train only `MotorHead` (~1.84M params) via balanced 75% active / 25% terminal (`done=1`) self-distillation or recorded desktop demos.
- `spotless_arms/record_demos.py`: Live desktop demonstration recorder supporting passive `/dev/input/event*` click listening with automatic unprivileged **Hyprland Hover-Snap Mode** (`grim` + `hyprctl cursorpos`).

---

## Verified Benchmark (RTX 4060 Laptop 8 GB, Wayland `3000x1920`)

| Metric | Value |
| :--- | :---: |
| **Steady-State Tracking ($K=1$)** | **`269.6 ms` (`3.71 Hz`)** |
| **Tick 1 Deliberation ($K=4$)** | **`352.9 ms` (`2.83 Hz`)** |
| **Overlapped Frame Fetch Wait** | **`0.01 ms`** (`11+ FPS` background grabber) |
| **Overlapped CPU `Prep` (steady-state)** | **`0.4–0.8 ms`** |
| **Kernel `/dev/uinput` Actuation** | **`0.26 ms`** |
| **Speedup vs `1325.2 ms` JSON Baseline** | **`4.92x`** |
| **VRAM Usage (Steady / Peak)** | **`5.45 GB` / `6.55 GB`** (`< 7.20 GB` budget) |

---

## Quickstart

### 1. Install Dependencies
```bash
uv sync
```

### 2. Authenticate to `openbmb/MiniCPM-V-2_6`
Accept the model license at `https://huggingface.co/openbmb/MiniCPM-V-2_6`, then log in:
```bash
uv run hf auth login
```

### 3. Place Checkpoints (Local Only — Ignored by Git)
Place pretrained weights in `./checkpoints/`:
- `checkpoints/best_vlm_adapter.pt`
- `checkpoints/best_motor_head.pt` (auto-loaded by `SpotlessVLA.from_pretrained()`)

### 4. Run Live Wayland Benchmark (`move_only=True` Safety Default)
```bash
uv run python -m spotless_arms.benchmark --ticks 8
```

### 5. Self-Distill or Fine-Tune `MotorHead` on Your Desktop
```bash
# Balanced 75% active / 25% halt self-distillation (no manual clicks required)
uv run python -m spotless_arms.train_motor_head --mode distill --live-capture 32 --epochs 35

# Or record live desktop demonstrations (auto-falls back to Hyprland Hover-Snap Mode)
uv run python -m spotless_arms.record_demos --max-samples 30
uv run python -m spotless_arms.train_motor_head --mode demos --manifest demos/manifest.jsonl
```

### 6. Run Unit & Integration Tests
```bash
uv run python -m unittest discover -s tests -v
```
