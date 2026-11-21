"""Zero-subprocess Linux kernel input driver via evdev.UInput (with ydotool fallback)."""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any, Dict, Optional

import torch

from spotless_arms.models.motor_head import decode_motor_vector

try:
    from evdev import AbsInfo, UInput, ecodes

    HAS_EVDEV = True
except ImportError:
    HAS_EVDEV = False
    AbsInfo = None  # type: ignore
    UInput = None  # type: ignore
    ecodes = None  # type: ignore


ABS_MAX = 32767


class LinuxVirtualArm:
    """Sub-millisecond Wayland cursor & click actuator via persistent /dev/uinput device.

    Automatically falls back to ``ydotool`` if ``/dev/uinput`` is unavailable/unpermitted,
    or to a no-op ``dry_run`` backend if explicitly requested or neither driver is usable.
    """

    def __init__(
        self,
        device_name: str = "spotless-arms-virtual-pointer",
        screen_w: int = 1920,
        screen_h: int = 1080,
        dry_run: bool = False,
        click_hold_s: float = 0.008,
    ) -> None:
        self.device_name = device_name
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.dry_run = dry_run
        self.click_hold_s = click_hold_s

        self._uinput: Optional[Any] = None
        self._ydotool_bin: Optional[str] = None
        self.backend: str = "dry_run"

        if not dry_run:
            self._init_backend()

    def _init_backend(self) -> None:
        # Primary: Direct kernel /dev/uinput absolute pointer (zero subprocesses, <0.2ms)
        if HAS_EVDEV:
            try:
                capabilities = {
                    ecodes.EV_KEY: [
                        ecodes.BTN_LEFT,
                        ecodes.BTN_RIGHT,
                        ecodes.BTN_MIDDLE,
                    ],
                    ecodes.EV_ABS: [
                        (
                            ecodes.ABS_X,
                            AbsInfo(value=0, min=0, max=ABS_MAX, fuzz=0, flat=0, resolution=0),
                        ),
                        (
                            ecodes.ABS_Y,
                            AbsInfo(value=0, min=0, max=ABS_MAX, fuzz=0, flat=0, resolution=0),
                        ),
                    ],
                }
                self._uinput = UInput(
                    events=capabilities,
                    name=self.device_name,
                    vendor=0x1D6B,
                    product=0x0104,
                    version=1,
                    input_props=[ecodes.INPUT_PROP_DIRECT],
                )
                self.backend = "uinput"
                return
            except Exception as exc:
                print(f"[LinuxVirtualArm] /dev/uinput init failed ({exc}); trying ydotool fallback...")

        # Fallback: ydotool CLI wrapper
        ydotool_path = shutil.which("ydotool")
        if ydotool_path is not None:
            self._ydotool_bin = ydotool_path
            self.backend = "ydotool"
            return

        print("[LinuxVirtualArm] Neither /dev/uinput nor ydotool available; falling back to dry_run.")
        self.backend = "dry_run"

    def move_abs(
        self,
        x_norm: float,
        y_norm: float,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ) -> None:
        """Move cursor to normalized ``[0.0, 1.0]`` coordinates in <1ms."""
        x_clamped = max(0.0, min(1.0, float(x_norm)))
        y_clamped = max(0.0, min(1.0, float(y_norm)))
        w = screen_w or self.screen_w
        h = screen_h or self.screen_h

        if self.backend == "uinput" and self._uinput is not None:
            abs_x = int(round(x_clamped * ABS_MAX))
            abs_y = int(round(y_clamped * ABS_MAX))
            self._uinput.write(ecodes.EV_ABS, ecodes.ABS_X, abs_x)
            self._uinput.write(ecodes.EV_ABS, ecodes.ABS_Y, abs_y)
            self._uinput.syn()
        elif self.backend == "ydotool" and self._ydotool_bin is not None:
            px = int(round(x_clamped * w))
            py = int(round(y_clamped * h))
            subprocess.run(
                [self._ydotool_bin, "mousemove", "--absolute", "-x", str(px), "-y", str(py)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def click(self, button: str = "left") -> None:
        """Dispatch a mouse button press+release in <1ms (plus optional hold time)."""
        btn = button.lower()
        if self.backend == "uinput" and self._uinput is not None:
            code_map = {
                "left": ecodes.BTN_LEFT,
                "right": ecodes.BTN_RIGHT,
                "middle": ecodes.BTN_MIDDLE,
            }
            if btn not in code_map:
                raise ValueError(f"Unsupported mouse button: {button!r}")
            code = code_map[btn]
            self._uinput.write(ecodes.EV_KEY, code, 1)
            self._uinput.syn()
            if self.click_hold_s > 0:
                time.sleep(self.click_hold_s)
            self._uinput.write(ecodes.EV_KEY, code, 0)
            self._uinput.syn()
        elif self.backend == "ydotool" and self._ydotool_bin is not None:
            # ydotool click codes: 0xC0 = left, 0xC1 = right, 0xC2 = middle
            code_map = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
            if btn not in code_map:
                raise ValueError(f"Unsupported mouse button: {button!r}")
            subprocess.run(
                [self._ydotool_bin, "click", code_map[btn]],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def execute_motor_vector(
        self,
        motor_vec: torch.Tensor,
        threshold: float = 0.5,
        move_only: bool = False,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Decode ``[1, 6]`` motor tensor and execute movement/clicks immediately.

        Args:
            motor_vec: Tensor ``[1, 6]`` or ``[6]`` from ``MotorHead``.
            threshold: Probability threshold for ``move``, ``left_click``, ``right_click``, ``done``.
            move_only: Safety flag — suppresses mouse clicks while still allowing cursor movement.
            screen_w: Current display width in pixels.
            screen_h: Current display height in pixels.
        """
        t0 = time.perf_counter()
        cmd = decode_motor_vector(motor_vec, threshold=threshold)
        w = screen_w or self.screen_w
        h = screen_h or self.screen_h

        moved = False
        clicked: Optional[str] = None

        if cmd["should_move"] or move_only:
            if cmd["should_move"]:
                self.move_abs(cmd["x_norm"], cmd["y_norm"], screen_w=w, screen_h=h)
                moved = True

        if not move_only:
            if cmd["left_click"]:
                self.click("left")
                clicked = "left"
            elif cmd["right_click"]:
                self.click("right")
                clicked = "right"

        actuate_ms = (time.perf_counter() - t0) * 1000.0
        cmd.update(
            {
                "backend": self.backend,
                "px": int(round(cmd["x_norm"] * w)),
                "py": int(round(cmd["y_norm"] * h)),
                "moved": moved,
                "clicked": clicked,
                "move_only": move_only,
                "actuate_ms": actuate_ms,
            }
        )
        return cmd

    def close(self) -> None:
        if self._uinput is not None:
            try:
                self._uinput.close()
            except Exception:
                pass
            self._uinput = None

    def __enter__(self) -> "LinuxVirtualArm":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
