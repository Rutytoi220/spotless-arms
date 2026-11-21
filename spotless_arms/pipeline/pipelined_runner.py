"""CPU-style 3-stage overlapped VLA reflex pipeline.

Stage 1 (Background CPU Thread): AsyncScreenBuffer continuously captures Wayland frames via grim.
Stage 2 (Main CUDA Thread):      SpotlessVLA.forward_motor() runs K=4 latent loop + MotorHead.
Stage 3 (Kernel uinput):         LinuxVirtualArm executes cursor/click in <1ms and immediately
                                 consumes the already-captured next frame.
"""

from __future__ import annotations

import io
import math
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
from PIL import Image

from spotless_arms.actuators.uinput_arm import LinuxVirtualArm
from spotless_arms.models.latent_core import to_device
from spotless_arms.models.motor_head import SpotlessVLA


@dataclass
class FramePacket:
    image: Image.Image
    frame_id: int
    width: int
    height: int
    capture_ms: float
    timestamp: float


class AsyncScreenBuffer:
    """Thread-safe double-buffered Wayland screen grabber using ``grim -t jpeg -q 80 -``."""

    def __init__(
        self,
        jpeg_quality: int = 80,
        max_width: Optional[int] = 1280,
        resize_max: Optional[Tuple[int, int]] = None,
        grim_bin: str = "grim",
    ) -> None:
        self.jpeg_quality = jpeg_quality
        self.max_width = max_width
        self.resize_max = resize_max
        self.grim_bin = grim_bin

        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._latest: Optional[FramePacket] = None
        self._frame_id: int = 0
        self._capture_times: Deque[float] = deque(maxlen=30)
        self._last_error: Optional[str] = None

    def _capture_once(self) -> FramePacket:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [self.grim_bin, "-t", "jpeg", "-q", str(self.jpeg_quality), "-"],
            capture_output=True,
            check=True,
        )
        img = Image.open(io.BytesIO(proc.stdout))
        img.load()
        orig_w, orig_h = img.size
        if img.mode != "RGB":
            img = img.convert("RGB")
        if self.max_width is not None and orig_w > self.max_width:
            scale = self.max_width / float(orig_w)
            new_h = max(1, int(round(orig_h * scale)))
            img = img.resize((self.max_width, new_h), Image.Resampling.BILINEAR)
        if self.resize_max is not None:
            img.thumbnail(self.resize_max, Image.Resampling.BILINEAR)

        capture_ms = (time.perf_counter() - t0) * 1000.0
        now = time.perf_counter()
        self._frame_id += 1
        return FramePacket(
            image=img,
            frame_id=self._frame_id,
            width=orig_w,
            height=orig_h,
            capture_ms=capture_ms,
            timestamp=now,
        )

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                pkt = self._capture_once()
                with self._lock:
                    self._latest = pkt
                    self._capture_times.append(pkt.timestamp)
                    self._last_error = None
                self._ready.set()
            except Exception as exc:
                self._last_error = str(exc)
                time.sleep(0.05)

    def start(self) -> "AsyncScreenBuffer":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            name="AsyncScreenBuffer",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def capture_fps(self) -> float:
        with self._lock:
            if len(self._capture_times) < 2:
                if self._latest and self._latest.capture_ms > 0:
                    return 1000.0 / self._latest.capture_ms
                return 0.0
            span = self._capture_times[-1] - self._capture_times[0]
            if span <= 0:
                return 0.0
            return (len(self._capture_times) - 1) / span

    def get_latest(
        self,
        min_frame_id: int = 0,
        timeout: float = 5.0,
    ) -> Tuple[FramePacket, float]:
        """Return ``(FramePacket, fetch_wait_ms)``.

        When the background grabber already has a frame newer than ``min_frame_id``,
        ``fetch_wait_ms`` is near 0ms (zero stall).
        """
        t0 = time.perf_counter()
        if not self._ready.wait(timeout=timeout):
            raise TimeoutError(
                f"AsyncScreenBuffer timed out waiting for first frame (last_error={self._last_error})"
            )

        deadline = t0 + timeout
        while True:
            with self._lock:
                pkt = self._latest
            if pkt is not None and pkt.frame_id >= min_frame_id:
                fetch_ms = (time.perf_counter() - t0) * 1000.0
                return pkt, fetch_ms
            if time.perf_counter() >= deadline:
                assert pkt is not None
                fetch_ms = (time.perf_counter() - t0) * 1000.0
                return pkt, fetch_ms
            time.sleep(0.001)

    def __enter__(self) -> "AsyncScreenBuffer":
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.stop()


class PipelinedVLARunner:
    """Overlapped 3-stage Fetch -> Think (Adaptive-K Latent + MotorHead) -> Act reflex loop."""

    def __init__(
        self,
        vla: SpotlessVLA,
        arm: LinuxVirtualArm,
        screen_buffer: AsyncScreenBuffer,
        K: Optional[int] = None,
        k_initial: int = 4,
        k_track: int = 1,
        auto_halt_on_click: bool = True,
        ema_alpha: float = 0.75,
        convergence_tol: float = 0.008,
        convergence_patience: int = 2,
        halt_on_convergence: bool = False,
        max_slice_nums: int = 1,
    ) -> None:
        self.vla = vla
        self.arm = arm
        self.screen_buffer = screen_buffer
        self.k_initial = K if K is not None else k_initial
        self.k_track = k_track
        self.auto_halt_on_click = auto_halt_on_click
        self.ema_alpha = float(ema_alpha)
        self.convergence_tol = float(convergence_tol)
        self.convergence_patience = int(convergence_patience)
        self.halt_on_convergence = bool(halt_on_convergence)
        self.max_slice_nums = max_slice_nums
        self.last_summary: Dict[str, Any] = {}

    def run(
        self,
        prompt: str,
        max_ticks: int = 20,
        threshold: float = 0.5,
        done_logit_threshold: Optional[float] = None,
        move_only: bool = False,
        auto_halt_on_click: Optional[bool] = None,
        halt_on_convergence: Optional[bool] = None,
        wait_for_fresh_frame: bool = False,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Execute the continuous pipelined reflex loop with Adaptive-K, EMA damping & convergence."""
        self.screen_buffer.start()
        self.vla.eval()

        halt_on_click = self.auto_halt_on_click if auto_halt_on_click is None else auto_halt_on_click
        halt_on_conv = self.halt_on_convergence if halt_on_convergence is None else halt_on_convergence
        telemetry: List[Dict[str, Any]] = []
        last_frame_id = 0
        use_cuda = self.vla.device.type == "cuda"

        ema_x: Optional[float] = None
        ema_y: Optional[float] = None
        stable_streak = 0
        converged = False
        converged_tick: Optional[int] = None
        jitter_samples: List[float] = []

        if verbose:
            print(
                f"{'Tick':>4} | {'K':>1} | {'Frm':>4} | {'Fetch':>7} | {'Prep':>7} | {'Motor':>8} | "
                f"{'Act':>6} | {'Hz':>6} | {'CapFPS':>6} | {'VRAM':>6} | Action"
            )
            print("-" * 109)

        # Pre-warm chat template cache once per episode
        self.vla.format_prompt(prompt)

        def _fetch_and_prep_cpu(min_id: int) -> Tuple[FramePacket, float, Dict[str, Any]]:
            pkt_local, fetch_local_ms = self.screen_buffer.get_latest(min_frame_id=min_id)
            cpu_data = self.vla.prepare_multimodal_data_cpu(
                pkt_local.image,
                prompt,
                max_slice_nums=self.max_slice_nums,
            )
            return pkt_local, fetch_local_ms, cpu_data

        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="VLAPrep") as prep_pool:
            next_future: Optional[Future[Tuple[FramePacket, float, Dict[str, Any]]]] = None

            for tick in range(1, max_ticks + 1):
                tick_t0 = time.perf_counter()
                k_steps = self.k_initial if tick == 1 else self.k_track

                # Stage 1 + 1.5: Consume prefetched CPU-preprocessed packet (overlapped with previous GPU pass)
                prep_t0 = time.perf_counter()
                if next_future is not None:
                    pkt, fetch_ms, cpu_data = next_future.result()
                    next_future = None
                else:
                    min_id = (last_frame_id + 1) if wait_for_fresh_frame else 0
                    pkt, fetch_ms, cpu_data = _fetch_and_prep_cpu(min_id)
                last_frame_id = pkt.frame_id

                # Non-blocking H2D transfer for current tick
                data = to_device(cpu_data, self.vla.device)
                del cpu_data
                prep_ms = (time.perf_counter() - prep_t0) * 1000.0

                # Launch overlapped fetch + CPU prep for tick t+1 WHILE GPU executes tick t
                if tick < max_ticks:
                    next_min_id = (last_frame_id + 1) if wait_for_fresh_frame else 0
                    next_future = prep_pool.submit(_fetch_and_prep_cpu, next_min_id)

                # Stage 2: Overlapping Adaptive-K Latent + MotorHead single forward pass
                motor_t0 = time.perf_counter()
                with torch.inference_mode():
                    if use_cuda:
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            motor_vec = self.vla.forward_motor(data=data, k_steps=k_steps)
                        torch.cuda.synchronize()
                    else:
                        motor_vec = self.vla.forward_motor(data=data, k_steps=k_steps)
                latent_motor_ms = (time.perf_counter() - motor_t0) * 1000.0
                del data

                # EMA Cursor Smoothing (Tick 1 snaps at alpha=1.0; Ticks 2+ smooth at ema_alpha)
                mv_cpu = motor_vec.detach().float().cpu().reshape(-1).clone()
                raw_x = float(mv_cpu[0].clamp(0.0, 1.0).item())
                raw_y = float(mv_cpu[1].clamp(0.0, 1.0).item())
                if ema_x is None or ema_y is None:
                    smooth_x, smooth_y = raw_x, raw_y
                    delta_norm = 0.0
                    jitter_px = 0.0
                else:
                    smooth_x = self.ema_alpha * raw_x + (1.0 - self.ema_alpha) * ema_x
                    smooth_y = self.ema_alpha * raw_y + (1.0 - self.ema_alpha) * ema_y
                    delta_norm = math.hypot(smooth_x - ema_x, smooth_y - ema_y)
                    jitter_px = math.hypot(
                        (smooth_x - ema_x) * float(pkt.width),
                        (smooth_y - ema_y) * float(pkt.height),
                    )
                    jitter_samples.append(jitter_px)
                    if delta_norm <= self.convergence_tol:
                        stable_streak += 1
                    else:
                        stable_streak = 0
                    if (not converged) and stable_streak >= self.convergence_patience:
                        converged = True
                        converged_tick = tick
                ema_x, ema_y = smooth_x, smooth_y
                mv_cpu[0] = smooth_x
                mv_cpu[1] = smooth_y

                # Stage 3: Sub-millisecond kernel actuation
                act = self.arm.execute_motor_vector(
                    mv_cpu.unsqueeze(0),
                    threshold=threshold,
                    move_only=move_only,
                    screen_w=pkt.width,
                    screen_h=pkt.height,
                )
                actuate_ms = act["actuate_ms"]

                tick_ms = (time.perf_counter() - tick_t0) * 1000.0
                effective_hz = 1000.0 / max(tick_ms, 1e-3)
                vram_gb = (
                    torch.cuda.memory_allocated() / (1024**3)
                    if use_cuda
                    else 0.0
                )

                record = {
                    "tick": tick,
                    "k_steps": k_steps,
                    "frame_id": pkt.frame_id,
                    "fetch_ms": fetch_ms,
                    "prep_ms": prep_ms,
                    "latent_motor_ms": latent_motor_ms,
                    "actuate_ms": actuate_ms,
                    "tick_ms": tick_ms,
                    "effective_hz": effective_hz,
                    "capture_fps": self.screen_buffer.capture_fps,
                    "vram_gb": vram_gb,
                    "jitter_px": jitter_px,
                    "converged": converged,
                    "converged_tick": converged_tick,
                    "action": act,
                    "halted_reason": None,
                }
                telemetry.append(record)

                if verbose:
                    act_str = (
                        f"({act['x_norm']:.3f},{act['y_norm']:.3f}) "
                        f"mv={int(act['moved'])} clk={act['clicked'] or '-'} "
                        f"done={act['done_logit']:+.2f} jit={jitter_px:.1f}px"
                    )
                    print(
                        f"{tick:4d} | {k_steps:1d} | {pkt.frame_id:4d} | {fetch_ms:5.2f}ms | {prep_ms:5.1f}ms | "
                        f"{latent_motor_ms:6.1f}ms | {actuate_ms:4.2f}ms | {effective_hz:6.2f} | "
                        f"{self.screen_buffer.capture_fps:6.1f} | {vram_gb:5.2f}G | {act_str}"
                    )

                if halt_on_click and (not move_only) and act["clicked"] is not None:
                    record["halted_reason"] = f"click:{act['clicked']}"
                    if verbose:
                        print(
                            f"[PipelinedVLARunner] Auto-halted at tick {tick}: "
                            f"single {act['clicked']} click executed."
                        )
                    break

                should_halt = (
                    act["done_logit"] > done_logit_threshold
                    if done_logit_threshold is not None
                    else act["done"]
                )
                if should_halt:
                    record["halted_reason"] = "done"
                    if verbose:
                        print(f"[PipelinedVLARunner] Halted at tick {tick}: done signal triggered.")
                    break

                if halt_on_conv and converged:
                    record["halted_reason"] = "converged"
                    if verbose:
                        print(
                            f"[PipelinedVLARunner] Early-halted at tick {tick}: "
                            f"spatial convergence reached (tol={self.convergence_tol}, patience={self.convergence_patience})."
                        )
                    break

        self.last_summary = {
            "ticks_executed": len(telemetry),
            "converged": converged,
            "converged_tick": converged_tick,
            "jitter_px": (sum(jitter_samples) / len(jitter_samples)) if jitter_samples else 0.0,
        }
        return telemetry
