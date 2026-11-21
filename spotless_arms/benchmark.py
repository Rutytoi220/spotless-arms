"""End-to-end live Wayland benchmark for Spotless-Arms VLA Engine.

Usage:
    uv run python -m spotless_arms.benchmark --ticks 10 --move-only
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

import torch

from spotless_arms.actuators.uinput_arm import LinuxVirtualArm
from spotless_arms.models.latent_core import DEFAULT_MODEL_ID
from spotless_arms.models.motor_head import SpotlessVLA, _AUTO_MOTOR_HEAD, decode_motor_vector
from spotless_arms.pipeline.pipelined_runner import AsyncScreenBuffer, PipelinedVLARunner


def run_benchmark(
    prompt: str = "Click the primary action button on screen.",
    ticks: int = 12,
    baseline_runs: int = 2,
    k_initial: int = 4,
    k_track: int = 1,
    move_only: bool = True,
    dry_run_arm: bool = False,
    model_id: str = DEFAULT_MODEL_ID,
    adapter_path: Optional[str] = None,
    motor_head_path: Optional[str] = _AUTO_MOTOR_HEAD,
    max_width: int = 1280,
    vram_budget_gb: float = 7.2,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("SPOTLESS-ARMS — Adaptive-K Pipelined Vision-Language-Action (VLA) Benchmark")
    print("=" * 80)
    print(f"Device:       {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    print(f"Model:        {model_id}")
    print(f"Adaptive-K:   k_initial={k_initial} (Tick 1), k_track={k_track} (Ticks 2+)")
    print(f"Safety Mode:  move_only={move_only}, dry_run_arm={dry_run_arm}")
    print(f"VRAM Budget:  < {vram_budget_gb:.2f} GB")
    print("-" * 80)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t_load0 = time.perf_counter()
    vla = SpotlessVLA.from_pretrained(
        model_id=model_id,
        adapter_path=adapter_path,
        motor_head_path=motor_head_path,
        device=device,
    )
    load_s = time.perf_counter() - t_load0
    print(f"[Benchmark] Model + Adapter + MotorHead loaded in {load_s:.2f}s")

    with AsyncScreenBuffer(jpeg_quality=80, max_width=max_width) as screen_buf, LinuxVirtualArm(
        dry_run=dry_run_arm
    ) as arm:
        print(f"[Benchmark] Actuator backend: {arm.backend}")

        # Grab initial live Wayland screenshot
        pkt, first_fetch_ms = screen_buf.get_latest(timeout=5.0)
        print(
            f"[Benchmark] Live Wayland frame #{pkt.frame_id}: "
            f"display={pkt.width}x{pkt.height}, vlm_input={pkt.image.width}x{pkt.image.height} "
            f"(initial grab {first_fetch_ms:.1f}ms, grim+resize {pkt.capture_ms:.1f}ms)"
        )

        # Warmup CUDA kernels (1 K=4 + 1 K=1 pass)
        print("[Benchmark] Warming up CUDA kernels...")
        warm_data = vla.prepare_multimodal_data(pkt.image, prompt)
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad():
                _ = vla.forward_motor(data=warm_data, k_steps=k_initial)
                _ = vla.forward_motor(data=warm_data, k_steps=k_track)
        del warm_data
        if device == "cuda":
            torch.cuda.synchronize()

        # 1. Baseline: Autoregressive text-JSON forward_text()
        baseline_latencies = []
        last_raw = ""
        print(f"\n[Benchmark] Running {baseline_runs}x baseline text-JSON passes (forward_text)...")
        for i in range(baseline_runs):
            res = vla.forward_text(image=pkt.image, prompt=prompt, K=k_initial, max_tokens=32)
            baseline_latencies.append(res["latency_ms"])
            last_raw = res["raw_output"]
            print(
                f"  Baseline #{i+1}: {res['latency_ms']:.1f}ms "
                f"({res['num_tokens']} tokens) -> {last_raw!r}"
            )
        baseline_avg_ms = statistics.mean(baseline_latencies)

        # 2. Pipelined VLA continuous motor reflex loop (Adaptive-K)
        print(
            f"\n[Benchmark] Running {ticks}-tick Adaptive-K pipelined VLA reflex loop "
            f"(k_initial={k_initial}, k_track={k_track}, move_only={move_only})..."
        )
        runner = PipelinedVLARunner(
            vla=vla,
            arm=arm,
            screen_buffer=screen_buf,
            k_initial=k_initial,
            k_track=k_track,
            auto_halt_on_click=True,
        )
        telemetry = runner.run(
            prompt=prompt,
            max_ticks=ticks,
            threshold=0.5,
            done_logit_threshold=999.0,  # Don't early-halt during fixed-length benchmark
            move_only=move_only,
            wait_for_fresh_frame=False,
            verbose=True,
        )

        # 3. Verify done_logit boundary on halt prompt vs active prompt
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.no_grad():
                halt_vec = vla.forward_motor(image=pkt.image, prompt="task complete, stop", k_steps=k_initial)
                active_vec = vla.forward_motor(image=pkt.image, prompt=prompt, k_steps=k_initial)
        halt_dec = decode_motor_vector(halt_vec)
        active_dec = decode_motor_vector(active_vec)

    tick1 = telemetry[0]
    track_ticks = telemetry[1:] if len(telemetry) > 1 else telemetry

    avg_prep_ms = statistics.mean(t["prep_ms"] for t in telemetry)
    avg_fetch_ms = statistics.mean(t["fetch_ms"] for t in telemetry)
    avg_act_ms = statistics.mean(t["actuate_ms"] for t in telemetry)
    steady_act_ms = statistics.mean(t["actuate_ms"] for t in track_ticks)
    avg_tick_ms = statistics.mean(t["tick_ms"] for t in telemetry)
    overall_hz = 1000.0 / max(avg_tick_ms, 1e-3)

    track_motor_ms = statistics.mean(t["latent_motor_ms"] for t in track_ticks)
    track_tick_ms = statistics.mean(t["tick_ms"] for t in track_ticks)
    track_hz = 1000.0 / max(track_tick_ms, 1e-3)

    speedup_vs_live_baseline = baseline_avg_ms / max(track_tick_ms, 1e-3)
    speedup_vs_1325ms = 1325.2 / max(track_tick_ms, 1e-3)
    speedup_overall_vs_1325ms = 1325.2 / max(avg_tick_ms, 1e-3)

    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3) if device == "cuda" else 0.0
    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3) if device == "cuda" else 0.0
    vram_ok = peak_alloc_gb < vram_budget_gb

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"Baseline text-JSON (live):        {baseline_avg_ms:7.1f} ms")
    print(f"Pipelined Fetch wait (avg):       {avg_fetch_ms:7.2f} ms  (overlapped background grabber)")
    print(f"CPU Prep (avg):                   {avg_prep_ms:7.1f} ms")
    print(f"Tick 1 (K={k_initial}) Motor / Total:      {tick1['latent_motor_ms']:7.1f} ms / {tick1['tick_ms']:.1f} ms ({tick1['effective_hz']:.2f} Hz)")
    print(f"Tracking (K={k_track}) Motor / Total:    {track_motor_ms:7.1f} ms / {track_tick_ms:.1f} ms ({track_hz:.2f} Hz)")
    print(f"Kernel Actuation (avg / steady):  {avg_act_ms:7.3f} ms / {steady_act_ms:.3f} ms (backend={arm.backend})")
    print(f"Overall Avg Tick ({len(telemetry)} ticks):      {avg_tick_ms:7.1f} ms  ({overall_hz:.2f} Hz)")
    print(f"Speedup vs live JSON (K={k_track}):       {speedup_vs_live_baseline:7.2f}x")
    print(f"Speedup vs 1325.2ms JSON (K={k_track}):   {speedup_vs_1325ms:7.2f}x  (overall {speedup_overall_vs_1325ms:.2f}x)")
    print(
        f"Active Prompt Check:              mv={int(active_dec['should_move'])} "
        f"lclk={int(active_dec['left_click'])} done_logit={active_dec['done_logit']:+.2f}"
    )
    print(
        f"Halt Prompt Check ('stop'):       mv={int(halt_dec['should_move'])} "
        f"lclk={int(halt_dec['left_click'])} done_logit={halt_dec['done_logit']:+.2f}"
    )
    print(f"Peak VRAM Allocated:              {peak_alloc_gb:7.2f} GB  (budget < {vram_budget_gb:.2f} GB -> {'PASS' if vram_ok else 'FAIL'})")
    print(f"Peak VRAM Reserved:               {peak_reserved_gb:7.2f} GB")
    print("=" * 80)

    if not vram_ok:
        raise SystemExit(f"VRAM budget exceeded: {peak_alloc_gb:.2f} GB >= {vram_budget_gb:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(description="Spotless-Arms live Wayland benchmark.")
    parser.add_argument("--prompt", type=str, default="Click the primary action button on screen.")
    parser.add_argument("--ticks", type=int, default=12)
    parser.add_argument("--baseline-runs", type=int, default=2)
    parser.add_argument("--k-initial", type=int, default=4)
    parser.add_argument("--k-track", type=int, default=1)
    parser.add_argument("--allow-clicks", action="store_true", help="Disable move_only safety flag.")
    parser.add_argument("--dry-run-arm", action="store_true", help="Do not emit real uinput events.")
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--motor-head-path", type=str, default=_AUTO_MOTOR_HEAD)
    parser.add_argument("--max-width", type=int, default=1280)
    args = parser.parse_args()

    run_benchmark(
        prompt=args.prompt,
        ticks=args.ticks,
        baseline_runs=args.baseline_runs,
        k_initial=args.k_initial,
        k_track=args.k_track,
        move_only=not args.allow_clicks,
        dry_run_arm=args.dry_run_arm,
        model_id=args.model_id,
        adapter_path=args.adapter_path,
        motor_head_path=args.motor_head_path,
        max_width=args.max_width,
    )


if __name__ == "__main__":
    main()
