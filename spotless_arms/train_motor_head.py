"""Fast <8GB VRAM Distillation & Demonstration Trainer for Spotless-Arms MotorHead.

Freezes vpm, resampler, Qwen2Model, and the K=4 SpotlessLatentAdapter, training ONLY
the ~1.84M-parameter MotorHead via SmoothL1 (x, y) + BCEWithLogitsLoss (move, clicks, done).

Supports two modes:
  1. ``distill``: Teacher Distillation from existing JSON outputs of ``best_vlm_adapter.pt``.
  2. ``demos``:   Supervised fine-tuning on user demonstrations recorded by ``record_demos.py``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance

from spotless_arms.models.latent_core import DEFAULT_MODEL_ID
from spotless_arms.models.motor_head import (
    DEFAULT_MOTOR_HEAD_PATH,
    MotorHead,
    SpotlessVLA,
    decode_motor_vector,
)
from spotless_arms.synthetic_ui import (
    TERMINAL_PROMPTS,
    UISample,
    generate_synthetic_ui_samples,
    harvest_hyprland_samples,
)


def _augment_brightness_contrast(img: Image.Image, rng: random.Random) -> Image.Image:
    """Spatial-safe visual augmentation (brightness/contrast jitter, zero pixel shift)."""
    b_factor = rng.uniform(0.90, 1.10)
    c_factor = rng.uniform(0.90, 1.10)
    out = ImageEnhance.Brightness(img).enhance(b_factor)
    out = ImageEnhance.Contrast(out).enhance(c_factor)
    return out


def _resize_for_vlm(img: Image.Image, max_width: Optional[int] = 1280) -> Image.Image:
    if max_width is not None and img.width > max_width:
        scale = max_width / float(img.width)
        new_h = max(1, int(round(img.height * scale)))
        return img.resize((max_width, new_h), Image.Resampling.BILINEAR)
    return img


def parse_teacher_json_to_target(
    teacher_result: Dict[str, Any],
    coord_scale: float = 1000.0,
) -> Optional[torch.Tensor]:
    """Convert teacher JSON ``{"action": "click", "point": [x, y]}`` into 6D supervision target.

    Target format: ``[x_norm, y_norm, move, left_click, right_click, done]``.
    """
    action = str(teacher_result.get("action", "unknown")).lower().strip()
    point = teacher_result.get("point", [0, 0])

    # Fallback: extract embedded JSON object if raw_output has extra wrapper text
    if action == "unknown" and point == [0, 0]:
        raw = str(teacher_result.get("raw_output", ""))
        match = re.search(r"\{[^{}]*\}", raw)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    action = str(parsed.get("action", "unknown")).lower().strip()
                    if isinstance(parsed.get("point"), list) and len(parsed["point"]) == 2:
                        point = parsed["point"]
            except json.JSONDecodeError:
                pass

    if action == "unknown" and point == [0, 0]:
        return None

    x_raw, y_raw = float(point[0]), float(point[1])
    if x_raw > 1.0 or y_raw > 1.0:
        x_norm = max(0.0, min(1.0, x_raw / coord_scale))
        y_norm = max(0.0, min(1.0, y_raw / coord_scale))
    else:
        x_norm = max(0.0, min(1.0, x_raw))
        y_norm = max(0.0, min(1.0, y_raw))

    move = 1.0 if action in {"click", "left_click", "right_click", "double_click", "move", "hover"} else 0.0
    left_click = 1.0 if action in {"click", "left_click", "double_click"} else 0.0
    right_click = 1.0 if action in {"right_click", "context_click"} else 0.0
    done = 1.0 if action in {"done", "finish", "complete", "stop"} else 0.0

    return torch.tensor([x_norm, y_norm, move, left_click, right_click, done], dtype=torch.float32)


def compute_motor_loss(
    pred_motor: torch.Tensor,
    target: torch.Tensor,
    coord_weight: float = 20.0,
    smooth_l1_beta: float = 0.02,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SmoothL1 (beta=0.02) on (x_norm, y_norm) + BCEWithLogitsLoss on (move, left, right, done)."""
    pred_coords = pred_motor[:, :2].float()
    pred_logits = pred_motor[:, 2:].float()
    tgt_coords = target[:, :2].float()
    tgt_flags = target[:, 2:].float()

    # Mask coordinate loss when target doesn't require movement
    move_mask = tgt_flags[:, 0:1]
    if move_mask.sum() > 0:
        coord_loss = (
            F.smooth_l1_loss(pred_coords, tgt_coords, beta=smooth_l1_beta, reduction="none") * move_mask
        ).sum() / (move_mask.sum() * 2.0)
    else:
        coord_loss = torch.tensor(0.0, device=pred_motor.device)

    flag_loss = F.binary_cross_entropy_with_logits(pred_logits, tgt_flags)
    total_loss = coord_weight * coord_loss + flag_loss
    return total_loss, {
        "total": float(total_loss.item()),
        "coord": float(coord_loss.item()),
        "flags": float(flag_loss.item()),
    }


def extract_features_from_ui_samples(
    vla: SpotlessVLA,
    samples: List[UISample],
    K: int = 4,
    label: str = "auto",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen ``normed_h`` hidden states for a list of ``UISample`` instances."""
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    t0 = time.perf_counter()

    for idx, s in enumerate(samples, start=1):
        data = vla.prepare_multimodal_data(s.image, s.prompt)
        with torch.inference_mode():
            if vla.device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    normed_h, _ = vla.extract_latent_state(data, k_steps=K, return_pkv=False)
            else:
                normed_h, _ = vla.extract_latent_state(data, k_steps=K, return_pkv=False)
        del data
        features.append(normed_h.detach().float().cpu().reshape(1, -1))
        targets.append(torch.tensor([s.target_tensor_values()], dtype=torch.float32))

        if idx == 1 or idx % 25 == 0 or idx == len(samples):
            elapsed = time.perf_counter() - t0
            print(
                f"  [{label} {idx:03d}/{len(samples)}] ({elapsed:.1f}s) "
                f"src={s.source} ({s.x_norm:.3f},{s.y_norm:.3f}) done={int(s.done)} prompt={s.prompt!r}"
            )

    if not features:
        raise RuntimeError(f"No samples provided to extract_features_from_ui_samples ({label}).")

    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def collect_distillation_features(
    vla: SpotlessVLA,
    image_paths: List[Path],
    prompts: List[str],
    K: int = 4,
    coord_scale: float = 1000.0,
    terminal_ratio: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract balanced ``(normed_h, target)`` pairs: ~75% active actions + ~25% terminal (done=1).

    Active samples are supervised by the frozen teacher's JSON output (``done=0``).
    Terminal/no-op samples pair screenshots with halt prompts (``move=0, clicks=0, done=1``)
    so ``MotorHead`` learns crisp stopping boundaries.
    """
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    active_count = 0
    terminal_count = 0

    total = len(image_paths)
    n_terminal = max(1, int(round(total * terminal_ratio))) if total >= 2 and terminal_ratio > 0 else 0
    n_active = total - n_terminal

    # 1. Active samples (75%): teacher JSON distillation
    for idx, img_path in enumerate(image_paths[:n_active]):
        img = _resize_for_vlm(Image.open(img_path).convert("RGB"), max_width=1280)
        for prompt in prompts:
            res = vla.forward_text(image=img, prompt=prompt, K=K, return_normed_h=True)
            tgt = parse_teacher_json_to_target(res, coord_scale=coord_scale)
            if tgt is None:
                print(
                    f"  [skip] {img_path.name} | prompt={prompt!r} -> unparseable teacher output: "
                    f"{res['raw_output']!r}"
                )
                continue
            h_vec = res["normed_h"].detach().float().cpu().reshape(1, -1)
            features.append(h_vec)
            targets.append(tgt.unsqueeze(0))
            active_count += 1
            print(
                f"  [active {active_count:03d}/{n_active * len(prompts)}] {img_path.name} -> "
                f"{res['action']} point={res['point']} ({res['latency_ms']:.0f}ms)"
            )

    # 2. Terminal / No-Op samples (25%): done=1, move=0, clicks=0
    if n_terminal > 0:
        terminal_target = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        for idx, img_path in enumerate(image_paths[n_active:]):
            img = _resize_for_vlm(Image.open(img_path).convert("RGB"), max_width=1280)
            halt_prompt = TERMINAL_PROMPTS[idx % len(TERMINAL_PROMPTS)]
            t0 = time.perf_counter()
            data = vla.prepare_multimodal_data(img, halt_prompt)
            with torch.inference_mode():
                normed_h, _ = vla.extract_latent_state(data, k_steps=K, return_pkv=False)
            del data
            ms = (time.perf_counter() - t0) * 1000.0
            h_vec = normed_h.detach().float().cpu().reshape(1, -1)
            features.append(h_vec)
            targets.append(terminal_target.clone())
            terminal_count += 1
            print(
                f"  [halt   {terminal_count:03d}/{n_terminal}] {img_path.name} -> "
                f"done=1 prompt={halt_prompt!r} ({ms:.0f}ms)"
            )

    if not features:
        raise RuntimeError("No valid distillation samples were generated.")

    print(
        f"[train_motor_head] Balanced dataset ready: "
        f"{active_count} active (done=0) + {terminal_count} terminal (done=1) = {len(features)} total."
    )
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def load_demo_records(manifest_path: Path) -> List[Dict[str, Any]]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_demo_features(
    vla: SpotlessVLA,
    manifest_path: Path,
    K: int = 4,
    records: Optional[List[Dict[str, Any]]] = None,
    oversample: int = 1,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract frozen ``normed_h`` hidden states for recorded demonstrations.

    When ``oversample > 1``, pass 0 is clean and passes ``1..oversample-1`` apply
    spatial-safe brightness/contrast jitter (preserving exact normalized coordinates).
    """
    base_dir = manifest_path.parent
    recs = records if records is not None else load_demo_records(manifest_path)
    if not recs:
        raise RuntimeError(f"No samples found in {manifest_path}")

    rng = random.Random(seed)
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []

    for idx, rec in enumerate(recs, start=1):
        img_path = base_dir / rec["image"]
        base_img = _resize_for_vlm(Image.open(img_path).convert("RGB"), max_width=1280)
        tgt = torch.tensor(
            [
                float(rec["x_norm"]),
                float(rec["y_norm"]),
                float(rec.get("move", 1.0)),
                float(rec.get("left_click", 1.0)),
                float(rec.get("right_click", 0.0)),
                float(rec.get("done", 0.0)),
            ],
            dtype=torch.float32,
        ).unsqueeze(0)

        for aug_i in range(max(1, oversample)):
            img = base_img if aug_i == 0 else _augment_brightness_contrast(base_img, rng)
            data = vla.prepare_multimodal_data(img, rec["prompt"])
            with torch.inference_mode():
                normed_h, _ = vla.extract_latent_state(data, k_steps=K, return_pkv=False)
            del data
            features.append(normed_h.detach().float().cpu().reshape(1, -1))
            targets.append(tgt.clone())

        print(
            f"  [demo {idx:03d}/{len(recs)}] {img_path.name} -> "
            f"({rec['x_norm']:.3f}, {rec['y_norm']:.3f}) x{max(1, oversample)}"
        )

    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def evaluate_motor_head(
    motor_head: MotorHead,
    val_features: torch.Tensor,
    val_targets: torch.Tensor,
    screen_w: int = 3000,
    screen_h: int = 1920,
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute held-out Mean Pixel Error, Hit Rate @35px/@50px, and Halt Accuracy."""
    motor_head.eval()
    with torch.no_grad():
        preds = motor_head(val_features.to(device)).float().cpu()

    tgts = val_targets.float().cpu()
    active_mask = tgts[:, 5] < 0.5
    halt_mask = tgts[:, 5] >= 0.5

    px_errors: List[float] = []
    if active_mask.any():
        dx = (preds[active_mask, 0] - tgts[active_mask, 0]) * float(screen_w)
        dy = (preds[active_mask, 1] - tgts[active_mask, 1]) * float(screen_h)
        dists = torch.sqrt(dx * dx + dy * dy)
        px_errors = [float(d.item()) for d in dists]

    mean_px_err = sum(px_errors) / len(px_errors) if px_errors else 0.0
    hit_35 = (sum(1 for d in px_errors if d <= 35.0) / len(px_errors)) if px_errors else 0.0
    hit_50 = (sum(1 for d in px_errors if d <= 50.0) / len(px_errors)) if px_errors else 0.0

    done_logits = preds[:, 5]
    pred_halt = done_logits > 0.0
    halt_correct = (pred_halt == halt_mask).float().mean().item() if len(tgts) > 0 else 0.0

    mean_active_done = float(done_logits[active_mask].mean().item()) if active_mask.any() else 0.0
    mean_halt_done = float(done_logits[halt_mask].mean().item()) if halt_mask.any() else 0.0

    return {
        "n_val_total": int(len(tgts)),
        "n_val_active": int(active_mask.sum().item()),
        "n_val_halt": int(halt_mask.sum().item()),
        "mean_px_err": mean_px_err,
        "hit_rate_35px": hit_35,
        "hit_rate_50px": hit_50,
        "halt_accuracy": float(halt_correct),
        "mean_active_done_logit": mean_active_done,
        "mean_halt_done_logit": mean_halt_done,
    }


def train_motor_head_from_cache(
    motor_head: MotorHead,
    features: torch.Tensor,
    targets: torch.Tensor,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    val_features: Optional[torch.Tensor] = None,
    val_targets: Optional[torch.Tensor] = None,
    scheduler_type: str = "plateau",
) -> MotorHead:
    """Train ONLY the MotorHead on cached ``normed_h`` features, keeping best validation weights."""
    import copy

    motor_head = motor_head.to(device=device, dtype=torch.float32)
    motor_head.train()
    for p in motor_head.parameters():
        p.requires_grad = True

    n_params = sum(p.numel() for p in motor_head.parameters() if p.requires_grad)
    print(f"[train_motor_head] Training {n_params:,} MotorHead parameters on {len(features)} samples...")

    optimizer = torch.optim.AdamW(motor_head.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_type == "plateau":
        scheduler: Any = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=6
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    best_val_score: Optional[Tuple[float, float]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None

    n = features.shape[0]
    for epoch in range(1, epochs + 1):
        motor_head.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        epoch_coord = 0.0
        epoch_flags = 0.0
        steps = 0

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = features[idx].to(device)
            yb = targets[idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = motor_head(xb)
            loss, metrics = compute_motor_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(motor_head.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += metrics["total"]
            epoch_coord += metrics["coord"]
            epoch_flags += metrics["flags"]
            steps += 1

        avg_loss = epoch_loss / max(steps, 1)
        if scheduler_type == "plateau":
            scheduler.step(avg_loss)
        else:
            scheduler.step()

        val_str = ""
        if val_features is not None and val_targets is not None:
            vm = evaluate_motor_head(motor_head, val_features, val_targets, 3000, 1920, device=device)
            score = (vm["halt_accuracy"], -vm["mean_px_err"])
            if best_val_score is None or score > best_val_score:
                best_val_score = score
                best_state = copy.deepcopy(motor_head.state_dict())
            val_str = (
                f" | val_px={vm['mean_px_err']:.1f}px "
                f"@35={vm['hit_rate_35px']*100:.0f}% halt={vm['halt_accuracy']*100:.0f}%"
            )

        if epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs:
            print(
                f"  Epoch {epoch:03d}/{epochs:03d} | "
                f"loss={avg_loss:.4f} | "
                f"coord={epoch_coord / steps:.4f} | "
                f"flags={epoch_flags / steps:.4f}{val_str}"
            )

    if best_state is not None:
        motor_head.load_state_dict(best_state)
    motor_head.eval()
    return motor_head


def capture_live_screenshots(out_dir: Path, count: int, interval_s: float = 0.5) -> List[Path]:
    """Helper to capture N live desktop frames via grim for quick self-distillation."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(count):
        proc = subprocess.run(
            ["grim", "-t", "jpeg", "-q", "85", "-"],
            capture_output=True,
            check=True,
        )
        p = out_dir / f"live_{i:04d}.jpg"
        p.write_bytes(proc.stdout)
        paths.append(p)
        if i + 1 < count:
            time.sleep(interval_s)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Spotless-Arms MotorHead (<8GB VRAM).")
    parser.add_argument(
        "--mode",
        choices=["auto", "mixed", "distill", "demos"],
        default=None,
        help="Training mode (defaults to 'auto' when demos/manifest.jsonl is absent or --ignore-ambiguous-demos is set).",
    )
    parser.add_argument("--ignore-ambiguous-demos", action="store_true", help="Ignore demos/manifest.jsonl and use --mode auto.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_MOTOR_HEAD_PATH)
    parser.add_argument("--images-dir", type=Path, default=None, help="Directory of screenshots for distill/mixed mode.")
    parser.add_argument("--live-capture", type=int, default=0, help="Capture N live Wayland frames if --images-dir is unset (defaults to 24).")
    parser.add_argument("--synthetic-samples", type=int, default=180, help="Procedural UI train samples for --mode auto.")
    parser.add_argument("--val-synthetic-samples", type=int, default=44, help="Unseen procedural UI validation samples (seed=999).")
    parser.add_argument("--hyprland-frames", type=int, default=24, help="Live Hyprland compositor-grounded frames for --mode auto.")
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["Click the primary action button on screen."],
    )
    parser.add_argument("--manifest", type=Path, default=Path("demos/manifest.jsonl"), help="Manifest JSONL for demos/mixed mode.")
    parser.add_argument("--demo-oversample", type=int, default=4, help="Spatial-safe oversampling factor for real user demos in mixed mode.")
    parser.add_argument("--bottleneck", type=int, default=768, help="MotorHead hidden bottleneck width.")
    parser.add_argument("--dropout", type=float, default=0.15, help="MotorHead dropout rate.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    mode = args.mode
    if mode is None:
        if args.ignore_ambiguous_demos or not args.manifest.is_file():
            mode = "auto"
        else:
            mode = "mixed"
        print(f"[train_motor_head] Auto-selected --mode {mode} (manifest exists={args.manifest.is_file()})")

    if mode in {"demos", "mixed"} and not args.manifest.is_file():
        print(
            f"[train_motor_head] Demo manifest not found at: {args.manifest}\n"
            f"  Run zero-click self-supervised UI grounding training:\n"
            f"    uv run python -m spotless_arms.train_motor_head --mode auto --epochs 50\n"
            f"  Or record desktop demonstrations first:\n"
            f"    uv run python -m spotless_arms.record_demos --max-samples 30"
        )
        return

    vla = SpotlessVLA.from_pretrained(
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        motor_head_path=None,
        device=args.device,
    )
    vla.motor_head = MotorHead(
        hidden_size=3584,
        bottleneck=args.bottleneck,
        out_dim=6,
        dropout=args.dropout,
    ).to(device=args.device, dtype=torch.float32)

    val_features: Optional[torch.Tensor] = None
    val_targets: Optional[torch.Tensor] = None

    if mode == "auto":
        print(
            f"[train_motor_head] Zero-click AUTO mode: {args.synthetic_samples} synthetic UI (seed=42) "
            f"+ {args.hyprland_frames} live Hyprland frames | {args.val_synthetic_samples} unseen val (seed=999)"
        )
        train_ui = generate_synthetic_ui_samples(
            n_samples=args.synthetic_samples, seed=42, terminal_ratio=0.25
        )
        hypr_samples = harvest_hyprland_samples(
            n_frames=args.hyprland_frames, max_width=1280, terminal_ratio=0.25
        )
        if hypr_samples:
            print(f"[train_motor_head] Harvested {len(hypr_samples)} live Hyprland compositor-grounded samples.")
        else:
            print("[train_motor_head] No live Hyprland clients available; proceeding with procedural UI samples.")

        all_train_samples = train_ui + hypr_samples
        val_samples = generate_synthetic_ui_samples(
            n_samples=args.val_synthetic_samples, seed=999, terminal_ratio=0.25
        )

        features, targets = extract_features_from_ui_samples(
            vla, all_train_samples, K=args.k, label="auto-train"
        )
        val_features, val_targets = extract_features_from_ui_samples(
            vla, val_samples, K=args.k, label="auto-val"
        )
    elif mode == "distill":
        if args.images_dir is not None:
            image_paths = sorted(
                p for p in args.images_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
        else:
            live_count = args.live_capture if args.live_capture > 0 else 24
            print(f"[train_motor_head] Capturing {live_count} live Wayland screenshots for distillation...")
            image_paths = capture_live_screenshots(Path("distill_frames"), live_count, interval_s=0.15)

        features, targets = collect_distillation_features(vla, image_paths, args.prompts, K=args.k)
    elif mode == "demos":
        all_recs = load_demo_records(args.manifest)
        n_val = max(1, int(round(len(all_recs) * 0.2))) if len(all_recs) >= 5 else 0
        val_set = set(range(1, len(all_recs), max(1, len(all_recs) // max(n_val, 1)))) if n_val > 0 else set()
        val_set = set(sorted(val_set)[:n_val])
        train_recs = [r for i, r in enumerate(all_recs) if i not in val_set]
        val_recs = [r for i, r in enumerate(all_recs) if i in val_set]
        features, targets = collect_demo_features(
            vla, args.manifest, K=args.k, records=train_recs, oversample=1
        )
        if val_recs:
            val_features, val_targets = collect_demo_features(
                vla, args.manifest, K=args.k, records=val_recs, oversample=1
            )
    else:
        # Mixed mode: 4x oversampled real user demos + teacher-distilled active + terminal halt (done=1)
        all_recs = load_demo_records(args.manifest)
        n_val_demos = max(2, int(round(len(all_recs) * 0.2))) if len(all_recs) >= 6 else (1 if len(all_recs) >= 3 else 0)
        # Deterministic interleaved split so validation spans the full spatial range
        step = max(1, len(all_recs) // max(n_val_demos, 1)) if n_val_demos > 0 else len(all_recs)
        val_indices = sorted(range(step // 2, len(all_recs), step))[:n_val_demos] if n_val_demos > 0 else []
        val_set = set(val_indices)
        train_recs = [r for i, r in enumerate(all_recs) if i not in val_set]
        val_recs = [r for i, r in enumerate(all_recs) if i in val_set]

        print(
            f"[train_motor_head] Mixed mode: {len(train_recs)} train demos (x{args.demo_oversample} aug) "
            f"+ {len(val_recs)} held-out val demos"
        )
        demo_train_x, demo_train_y = collect_demo_features(
            vla, args.manifest, K=args.k, records=train_recs, oversample=args.demo_oversample
        )
        demo_val_x, demo_val_y = (
            collect_demo_features(vla, args.manifest, K=args.k, records=val_recs, oversample=1)
            if val_recs
            else (None, None)
        )

        if args.images_dir is not None:
            image_paths = sorted(
                p for p in args.images_dir.iterdir()
                if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            )
        else:
            live_count = args.live_capture if args.live_capture > 0 else 24
            print(f"[train_motor_head] Capturing {live_count} live Wayland screenshots for mixed distillation...")
            image_paths = capture_live_screenshots(Path("distill_frames"), live_count, interval_s=0.15)

        distill_x, distill_y = collect_distillation_features(
            vla, image_paths, args.prompts, K=args.k, terminal_ratio=0.35
        )
        # Hold out 2 terminal (done=1) samples for halt classification validation
        halt_indices = (distill_y[:, 5] > 0.5).nonzero(as_tuple=True)[0]
        active_indices = (distill_y[:, 5] < 0.5).nonzero(as_tuple=True)[0]
        n_val_halt = min(2, max(1, len(halt_indices) // 4)) if len(halt_indices) >= 3 else 0
        train_halt_idx = halt_indices[n_val_halt:]
        val_halt_idx = halt_indices[:n_val_halt]

        train_distill_idx = torch.cat([active_indices, train_halt_idx], dim=0)
        features = torch.cat([demo_train_x, distill_x[train_distill_idx]], dim=0)
        targets = torch.cat([demo_train_y, distill_y[train_distill_idx]], dim=0)

        val_x_parts = []
        val_y_parts = []
        if demo_val_x is not None and demo_val_y is not None:
            val_x_parts.append(demo_val_x)
            val_y_parts.append(demo_val_y)
        if len(val_halt_idx) > 0:
            val_x_parts.append(distill_x[val_halt_idx])
            val_y_parts.append(distill_y[val_halt_idx])
        if val_x_parts:
            val_features = torch.cat(val_x_parts, dim=0)
            val_targets = torch.cat(val_y_parts, dim=0)

        print(
            f"[train_motor_head] Mixed train set: {len(features)} samples "
            f"(demos={len(demo_train_x)}, active_distill={len(active_indices)}, halt={len(train_halt_idx)})"
        )

    trained_head = train_motor_head_from_cache(
        motor_head=vla.motor_head,
        features=features,
        targets=targets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        val_features=val_features,
        val_targets=val_targets,
        scheduler_type="plateau",
    )

    if val_features is not None and val_targets is not None:
        val_metrics = evaluate_motor_head(
            trained_head, val_features, val_targets, screen_w=3000, screen_h=1920, device=args.device
        )
        print(
            f"[train_motor_head] HELD-OUT VALIDATION (3000x1920): "
            f"mean_px_err={val_metrics['mean_px_err']:.1f}px | "
            f"hit@35px={val_metrics['hit_rate_35px']*100:.1f}% | "
            f"hit@50px={val_metrics['hit_rate_50px']*100:.1f}% | "
            f"halt_acc={val_metrics['halt_accuracy']*100:.1f}% | "
            f"active_done={val_metrics['mean_active_done_logit']:+.2f} | "
            f"halt_done={val_metrics['mean_halt_done_logit']:+.2f}"
        )

    # Sanity-check learned active vs terminal (done=1) predictions
    with torch.no_grad():
        preds = trained_head(features.to(args.device)).cpu()
        active_idx = (targets[:, 5] < 0.5).nonzero(as_tuple=True)[0]
        halt_idx = (targets[:, 5] > 0.5).nonzero(as_tuple=True)[0]
        if len(active_idx) > 0:
            a_dec = decode_motor_vector(preds[active_idx[0]])
            print(
                f"[train_motor_head] Verify ACTIVE sample -> "
                f"({a_dec['x_norm']:.3f},{a_dec['y_norm']:.3f}) "
                f"mv={int(a_dec['should_move'])} lclk={int(a_dec['left_click'])} "
                f"done_logit={a_dec['done_logit']:+.2f} (p={a_dec['done_prob']:.3f})"
            )
        if len(halt_idx) > 0:
            h_dec = decode_motor_vector(preds[halt_idx[0]])
            print(
                f"[train_motor_head] Verify HALT   sample -> "
                f"({h_dec['x_norm']:.3f},{h_dec['y_norm']:.3f}) "
                f"mv={int(h_dec['should_move'])} lclk={int(h_dec['left_click'])} "
                f"done_logit={h_dec['done_logit']:+.2f} (p={h_dec['done_prob']:.3f})"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_head.to(torch.bfloat16).state_dict(), args.output)
    print(f"[train_motor_head] Saved MotorHead checkpoint -> {args.output}")


if __name__ == "__main__":
    main()
