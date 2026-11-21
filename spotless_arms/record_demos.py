"""Live Desktop Demonstration Recorder for Spotless-Arms MotorHead fine-tuning.

Captures ``(screenshot, prompt, x_norm, y_norm, left_click, right_click, done)`` samples
whenever the user clicks on their Wayland desktop, writing frames and ``manifest.jsonl``.

If ``/dev/input/event*`` is not readable (user not in ``input`` group), automatically
falls back to unprivileged **Hyprland Hover-Snap Mode** using ``grim`` + ``hyprctl cursorpos``.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import select
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

try:
    from evdev import InputDevice, ecodes, list_devices

    HAS_EVDEV = True
except ImportError:
    HAS_EVDEV = False
    InputDevice = None  # type: ignore
    ecodes = None  # type: ignore
    list_devices = None  # type: ignore


def get_cursor_position_wayland() -> Optional[Tuple[int, int]]:
    """Query current cursor pixel position via compositor IPC (Hyprland)."""
    if shutil.which("hyprctl"):
        try:
            out = subprocess.check_output(["hyprctl", "cursorpos"], text=True, timeout=0.5).strip()
            m = re.match(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*$", out)
            if m:
                return int(m.group(1)), int(m.group(2))
        except Exception:
            pass
    return None


def get_display_dimensions_wayland() -> Optional[Tuple[int, int]]:
    """Query active monitor dimensions via ``hyprctl monitors -j`` if available."""
    if shutil.which("hyprctl"):
        try:
            raw = subprocess.check_output(["hyprctl", "monitors", "-j"], text=True, timeout=0.5)
            monitors = json.loads(raw)
            if isinstance(monitors, list) and monitors:
                focused = next((m for m in monitors if m.get("focused")), monitors[0])
                w = int(focused.get("width", 0))
                h = int(focused.get("height", 0))
                if w > 0 and h > 0:
                    return w, h
        except Exception:
            pass
    return None


def capture_frame_jpeg(quality: int = 85) -> Tuple[bytes, int, int]:
    """Capture current Wayland screen via grim and return ``(jpeg_bytes, width, height)``."""
    proc = subprocess.run(
        ["grim", "-t", "jpeg", "-q", str(quality), "-"],
        capture_output=True,
        check=True,
    )
    img = Image.open(io.BytesIO(proc.stdout))
    w, h = img.size
    return proc.stdout, w, h


def find_pointer_devices() -> List[InputDevice]:
    """Discover readable Linux evdev pointer/mouse devices."""
    if not HAS_EVDEV:
        return []
    devices: List[InputDevice] = []
    for path in list_devices():
        try:
            dev = InputDevice(path)
            caps = dev.capabilities()
            keys = caps.get(ecodes.EV_KEY, [])
            if ecodes.BTN_LEFT in keys:
                devices.append(dev)
            else:
                dev.close()
        except (PermissionError, OSError):
            continue
    return devices


def _record_hover_snap_mode(
    output_dir: Path,
    manifest_path: Path,
    prompt: str,
    max_samples: int,
    jpeg_quality: int,
) -> Path:
    """Unprivileged Hyprland Hover-Snap fallback using ``grim`` + ``hyprctl cursorpos``."""
    print(
        "[record_demos] /dev/input/event* is restricted (optional: run "
        "`sudo usermod -aG input $USER` and re-login for passive click listening)."
    )
    print("[record_demos] Switching automatically to Hyprland Hover-Snap Mode (grim + hyprctl cursorpos).")
    print(f"[record_demos] Default prompt: {prompt!r}")
    print(f"[record_demos] Output: {manifest_path}")

    count = 0
    try:
        with manifest_path.open("a", encoding="utf-8") as mf:
            while count < max_samples:
                try:
                    entered = input(
                        "Hover cursor over target & press Enter (or type label + Enter, 'q' to finish): "
                    ).strip()
                except EOFError:
                    print("\n[record_demos] EOF reached; finishing Hover-Snap session.")
                    break

                if entered.lower() in {"q", "quit", "exit"}:
                    print("[record_demos] Quit requested; finishing Hover-Snap session.")
                    break

                sample_prompt = entered if entered else prompt
                now = time.time()
                pos = get_cursor_position_wayland()
                jpeg_bytes, img_w, img_h = capture_frame_jpeg(quality=jpeg_quality)
                mon_dims = get_display_dimensions_wayland()
                disp_w, disp_h = mon_dims if mon_dims is not None else (img_w, img_h)

                if pos is not None:
                    px, py = pos
                    x_norm = max(0.0, min(1.0, px / max(disp_w, 1)))
                    y_norm = max(0.0, min(1.0, py / max(disp_h, 1)))
                else:
                    px, py = -1, -1
                    x_norm, y_norm = 0.5, 0.5

                ts_ms = int(now * 1000)
                frame_rel = f"frames/demo_{ts_ms}_{count:04d}.jpg"
                (output_dir / frame_rel).write_bytes(jpeg_bytes)

                sample = {
                    "image": frame_rel,
                    "prompt": sample_prompt,
                    "x_norm": round(x_norm, 6),
                    "y_norm": round(y_norm, 6),
                    "px": px,
                    "py": py,
                    "screen_w": disp_w,
                    "screen_h": disp_h,
                    "button": "left",
                    "move": 1.0,
                    "left_click": 1.0,
                    "right_click": 0.0,
                    "done": 0.0,
                    "timestamp": now,
                }
                mf.write(json.dumps(sample) + "\n")
                mf.flush()
                count += 1
                print(
                    f"  [{count:03d}/{max_samples}] HOVER-SNAP ({px},{py}) "
                    f"norm=({x_norm:.3f},{y_norm:.3f}) prompt={sample_prompt!r} -> {frame_rel}"
                )
    except KeyboardInterrupt:
        print("\n[record_demos] Stopped by user.")

    print(f"[record_demos] Saved {count} demonstration(s) to {manifest_path}")
    return manifest_path


def record_demonstrations(
    output_dir: Path,
    prompt: str = "click target UI element",
    max_samples: int = 50,
    jpeg_quality: int = 85,
    interactive_prompt: bool = False,
) -> Path:
    """Listen for mouse clicks (or fall back to Hyprland Hover-Snap) and save demo pairs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"

    devices = find_pointer_devices()
    if not devices:
        return _record_hover_snap_mode(
            output_dir=output_dir,
            manifest_path=manifest_path,
            prompt=prompt,
            max_samples=max_samples,
            jpeg_quality=jpeg_quality,
        )

    print(f"[record_demos] Watching {len(devices)} pointer device(s):")
    for d in devices:
        print(f"  - {d.path}: {d.name}")
    print(f"[record_demos] Default prompt: {prompt!r} (interactive={interactive_prompt})")
    print(f"[record_demos] Output: {manifest_path}")
    print("[record_demos] Click anywhere to record a sample. Press Ctrl+C to finish.")

    dev_map: Dict[int, InputDevice] = {d.fd: d for d in devices}
    count = 0
    last_click_ts = 0.0

    try:
        with manifest_path.open("a", encoding="utf-8") as mf:
            while count < max_samples:
                rlist, _, _ = select.select(list(dev_map.keys()), [], [], 0.25)
                for fd in rlist:
                    dev = dev_map[fd]
                    for event in dev.read():
                        if event.type != ecodes.EV_KEY or event.value != 1:
                            continue
                        if event.code not in (ecodes.BTN_LEFT, ecodes.BTN_RIGHT):
                            continue

                        now = time.time()
                        if now - last_click_ts < 0.15:
                            continue
                        last_click_ts = now

                        pos = get_cursor_position_wayland()
                        jpeg_bytes, w, h = capture_frame_jpeg(quality=jpeg_quality)

                        if pos is not None:
                            px, py = pos
                            x_norm = max(0.0, min(1.0, px / max(w, 1)))
                            y_norm = max(0.0, min(1.0, py / max(h, 1)))
                        else:
                            px, py = -1, -1
                            x_norm, y_norm = 0.5, 0.5

                        ts_ms = int(now * 1000)
                        frame_rel = f"frames/demo_{ts_ms}_{count:04d}.jpg"
                        (output_dir / frame_rel).write_bytes(jpeg_bytes)

                        is_left = event.code == ecodes.BTN_LEFT
                        is_right = event.code == ecodes.BTN_RIGHT

                        sample_prompt = prompt
                        if interactive_prompt:
                            try:
                                entered = input(f"  Label for click ({px},{py}) [{prompt}]: ").strip()
                                if entered:
                                    sample_prompt = entered
                            except EOFError:
                                pass

                        sample = {
                            "image": frame_rel,
                            "prompt": sample_prompt,
                            "x_norm": round(x_norm, 6),
                            "y_norm": round(y_norm, 6),
                            "px": px,
                            "py": py,
                            "screen_w": w,
                            "screen_h": h,
                            "button": "left" if is_left else "right",
                            "move": 1.0,
                            "left_click": 1.0 if is_left else 0.0,
                            "right_click": 1.0 if is_right else 0.0,
                            "done": 0.0,
                            "timestamp": now,
                        }
                        mf.write(json.dumps(sample) + "\n")
                        mf.flush()
                        count += 1
                        btn_name = "LEFT" if is_left else "RIGHT"
                        print(
                            f"  [{count:03d}/{max_samples}] {btn_name} click @ "
                            f"({px},{py}) norm=({x_norm:.3f},{y_norm:.3f}) -> {frame_rel}"
                        )
                        if count >= max_samples:
                            break
    except KeyboardInterrupt:
        print("\n[record_demos] Stopped by user.")
    finally:
        for d in devices:
            d.close()

    print(f"[record_demos] Saved {count} demonstration(s) to {manifest_path}")
    return manifest_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record live desktop demonstrations for MotorHead.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="click target UI element",
        help="Default task prompt for recorded clicks (optional; defaults to 'click target UI element').",
    )
    parser.add_argument(
        "--interactive-prompt",
        action="store_true",
        help="Prompt on stdin to type a custom label after each recorded click.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("demos"), help="Directory to store frames & manifest.jsonl.")
    parser.add_argument("--max-samples", type=int, default=50, help="Maximum clicks to record.")
    parser.add_argument("--quality", type=int, default=85, help="JPEG quality for grim capture.")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    record_demonstrations(
        output_dir=args.output_dir,
        prompt=args.prompt,
        max_samples=args.max_samples,
        jpeg_quality=args.quality,
        interactive_prompt=args.interactive_prompt,
    )


if __name__ == "__main__":
    main()
