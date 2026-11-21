"""Comprehensive Evaluation Harness & 50-Tick Continuous Soak Test for Spotless-Arms.

Usage:
    unset HF_TOKEN && uv run python -m spotless_arms.eval
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from spotless_arms.actuators.uinput_arm import LinuxVirtualArm
from spotless_arms.models.latent_core import DEFAULT_MODEL_ID
from spotless_arms.models.motor_head import (
    SpotlessVLA,
    _AUTO_MOTOR_HEAD,
    decode_motor_vector,
)
from spotless_arms.pipeline.pipelined_runner import AsyncScreenBuffer, PipelinedVLARunner
from spotless_arms.train_motor_head import (
    TERMINAL_PROMPTS,
    collect_demo_features,
    evaluate_motor_head,
    load_demo_records,
)


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * (pct / 100.0)
    lo = int(idx)
    hi = min(len(ordered) - 1, lo + 1)
    frac = idx - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def run_eval(
    manifest_path: Path = Path("demos/manifest.jsonl"),
    soak_ticks: int = 50,
    k_initial: int = 4,
    k_track: int = 1,
    model_id: str = DEFAULT_MODEL_ID,
    adapter_path: Optional[str] = None,
    motor_head_path: Optional[str] = _AUTO_MOTOR_HEAD,
    vram_budget_gb: float = 7.20,
    vram_drift_budget_gb: float = 0.05,
    screen_w: int = 3000,
    screen_h: int = 1920,
) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("SPOTLESS-ARMS — Comprehensive Eval Harness & 50-Tick Soak Test")
    print("=" * 80)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    vla = SpotlessVLA.from_pretrained(
        model_id=model_id,
        adapter_path=adapter_path,
        motor_head_path=motor_head_path,
        device=device,
    )

    scorecard: Dict[str, Any] = {
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "demo_eval": {},
        "stop_boundary": {},
        "soak_test": {},
        "passed": True,
    }

    # 1. Held-Out & Full Demo Grounding Accuracy
    if manifest_path.is_file():
        all_recs = load_demo_records(manifest_path)
        n_val = max(2, int(round(len(all_recs) * 0.2))) if len(all_recs) >= 6 else len(all_recs)
        step = max(1, len(all_recs) // max(n_val, 1)) if n_val > 0 else len(all_recs)
        val_indices = sorted(range(step // 2, len(all_recs), step))[:n_val]
        val_recs = [all_recs[i] for i in val_indices]

        print(f"\n[Eval 1/3] Evaluating demo grounding ({len(val_recs)} held-out / {len(all_recs)} total demos)...")
        all_x, all_y = collect_demo_features(vla, manifest_path, K=k_initial, records=all_recs, oversample=1)
        val_x = all_x[val_indices]
        val_y = all_y[val_indices]

        val_metrics = evaluate_motor_head(
            vla.motor_head, val_x, val_y, screen_w=screen_w, screen_h=screen_h, device=device
        )
        full_metrics = evaluate_motor_head(
            vla.motor_head, all_x, all_y, screen_w=screen_w, screen_h=screen_h, device=device
        )
        scorecard["demo_eval"] = {
            "held_out": val_metrics,
            "full_manifest": full_metrics,
        }
    else:
        print(f"\n[Eval 1/3] Skipping demo grounding (manifest not found: {manifest_path})")

    # 2. Stop-Boundary Classification + 50-Tick Continuous Pipelined Soak Test (move_only=True)
    with AsyncScreenBuffer(jpeg_quality=80, max_width=1280) as sb, LinuxVirtualArm(dry_run=False) as arm:
        pkt, _ = sb.get_latest(timeout=5.0)

        print("\n[Eval 2/3] Testing Active vs. Halt stop-boundary classification...")
        active_prompts = [
            "Click the primary action button on screen.",
            "click target UI element",
            "Locate the center of the active window.",
        ]
        stop_correct = 0
        stop_total = 0
        active_done_logits: List[float] = []
        halt_done_logits: List[float] = []

        with torch.inference_mode():
            for p in active_prompts:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad():
                    mv = vla.forward_motor(image=pkt.image, prompt=p, k_steps=k_initial)
                dec = decode_motor_vector(mv)
                active_done_logits.append(dec["done_logit"])
                stop_correct += int(dec["done_logit"] < 0.0)
                stop_total += 1
            for p in TERMINAL_PROMPTS:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad():
                    mv = vla.forward_motor(image=pkt.image, prompt=p, k_steps=k_initial)
                dec = decode_motor_vector(mv)
                halt_done_logits.append(dec["done_logit"])
                stop_correct += int(dec["done_logit"] > 0.0)
                stop_total += 1

        stop_acc = stop_correct / max(stop_total, 1)
        scorecard["stop_boundary"] = {
            "accuracy": stop_acc,
            "correct": stop_correct,
            "total": stop_total,
            "mean_active_done_logit": statistics.mean(active_done_logits),
            "mean_halt_done_logit": statistics.mean(halt_done_logits),
        }

        print(f"\n[Eval 3/3] Running {soak_ticks}-tick continuous pipelined soak test (move_only=True)...")
        runner = PipelinedVLARunner(
            vla=vla,
            arm=arm,
            screen_buffer=sb,
            k_initial=k_initial,
            k_track=k_track,
            auto_halt_on_click=True,
            ema_alpha=0.75,
            halt_on_convergence=False,
        )
        telemetry = runner.run(
            prompt="Click the primary action button on screen.",
            max_ticks=soak_ticks,
            done_logit_threshold=999.0,
            move_only=True,
            verbose=False,
        )

    tick1 = telemetry[0]
    track = telemetry[1:] if len(telemetry) > 1 else telemetry
    track_ms = [t["tick_ms"] for t in track]
    track_motor_ms = [t["latent_motor_ms"] for t in track]

    vram_t5 = telemetry[min(4, len(telemetry) - 1)]["vram_gb"]
    vram_t_end = telemetry[-1]["vram_gb"]
    vram_drift = abs(vram_t_end - vram_t5)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0

    vram_ok = peak_vram < vram_budget_gb
    drift_ok = vram_drift < vram_drift_budget_gb

    scorecard["soak_test"] = {
        "ticks": len(telemetry),
        "tick1_k4_ms": round(tick1["tick_ms"], 2),
        "track_k1_mean_ms": round(statistics.mean(track_ms), 2),
        "track_k1_motor_mean_ms": round(statistics.mean(track_motor_ms), 2),
        "track_k1_hz": round(1000.0 / max(statistics.mean(track_ms), 1e-3), 2),
        "p50_tick_ms": round(_percentile(track_ms, 50.0), 2),
        "p95_tick_ms": round(_percentile(track_ms, 95.0), 2),
        "mean_jitter_px": round(runner.last_summary.get("jitter_px", 0.0), 2),
        "converged": runner.last_summary.get("converged", False),
        "converged_tick": runner.last_summary.get("converged_tick"),
        "vram_tick5_gb": round(vram_t5, 4),
        "vram_tick_end_gb": round(vram_t_end, 4),
        "vram_drift_gb": round(vram_drift, 4),
        "peak_vram_gb": round(peak_vram, 4),
        "vram_budget_pass": vram_ok,
        "vram_drift_pass": drift_ok,
    }
    scorecard["passed"] = bool(vram_ok and drift_ok and stop_acc >= 0.8)

    print("\n" + "=" * 80)
    print("SPOTLESS-ARMS EVAL SCORECARD")
    print("=" * 80)
    print(json.dumps(scorecard, indent=2))
    print("=" * 80)

    if not vram_ok:
        raise SystemExit(f"FAIL: Peak VRAM {peak_vram:.2f} GB >= {vram_budget_gb:.2f} GB")
    if not drift_ok:
        raise SystemExit(f"FAIL: VRAM drift {vram_drift:.4f} GB >= {vram_drift_budget_gb:.2f} GB")

    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser(description="Spotless-Arms eval harness & 50-tick soak test.")
    parser.add_argument("--manifest", type=Path, default=Path("demos/manifest.jsonl"))
    parser.add_argument("--soak-ticks", type=int, default=50)
    parser.add_argument("--k-initial", type=int, default=4)
    parser.add_argument("--k-track", type=int, default=1)
    parser.add_argument("--motor-head-path", type=str, default=_AUTO_MOTOR_HEAD)
    args = parser.parse_args()

    run_eval(
        manifest_path=args.manifest,
        soak_ticks=args.soak_ticks,
        k_initial=args.k_initial,
        k_track=args.k_track,
        motor_head_path=args.motor_head_path,
    )


if __name__ == "__main__":
    main()
