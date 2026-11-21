"""Zero-subprocess Linux kernel input driver via evdev.UInput (with ydotool fallback).

Supports sub-millisecond Wayland cursor movement, mouse clicks, scroll wheel, and
keyboard keystrokes/text typing without spawning subprocesses per tick.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

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


def _build_key_tables() -> Tuple[Dict[str, int], Dict[str, Tuple[int, bool]], List[int]]:
    """Build named-key map, printable ASCII -> (keycode, shift) map, and full EV_KEY list."""
    if not HAS_EVDEV:
        return {}, {}, []

    named: Dict[str, int] = {
        "enter": ecodes.KEY_ENTER,
        "return": ecodes.KEY_ENTER,
        "esc": ecodes.KEY_ESC,
        "escape": ecodes.KEY_ESC,
        "tab": ecodes.KEY_TAB,
        "backspace": ecodes.KEY_BACKSPACE,
        "space": ecodes.KEY_SPACE,
        "up": ecodes.KEY_UP,
        "down": ecodes.KEY_DOWN,
        "left": ecodes.KEY_LEFT,
        "right": ecodes.KEY_RIGHT,
        "shift": ecodes.KEY_LEFTSHIFT,
        "ctrl": ecodes.KEY_LEFTCTRL,
        "alt": ecodes.KEY_LEFTALT,
    }

    char_map: Dict[str, Tuple[int, bool]] = {
        " ": (ecodes.KEY_SPACE, False),
        "\n": (ecodes.KEY_ENTER, False),
        "\t": (ecodes.KEY_TAB, False),
        "-": (ecodes.KEY_MINUS, False),
        "_": (ecodes.KEY_MINUS, True),
        "=": (ecodes.KEY_EQUAL, False),
        "+": (ecodes.KEY_EQUAL, True),
        "[": (ecodes.KEY_LEFTBRACE, False),
        "{": (ecodes.KEY_LEFTBRACE, True),
        "]": (ecodes.KEY_RIGHTBRACE, False),
        "}": (ecodes.KEY_RIGHTBRACE, True),
        ";": (ecodes.KEY_SEMICOLON, False),
        ":": (ecodes.KEY_SEMICOLON, True),
        "'": (ecodes.KEY_APOSTROPHE, False),
        '"': (ecodes.KEY_APOSTROPHE, True),
        "`": (ecodes.KEY_GRAVE, False),
        "~": (ecodes.KEY_GRAVE, True),
        "\\": (ecodes.KEY_BACKSLASH, False),
        "|": (ecodes.KEY_BACKSLASH, True),
        ",": (ecodes.KEY_COMMA, False),
        "<": (ecodes.KEY_COMMA, True),
        ".": (ecodes.KEY_DOT, False),
        ">": (ecodes.KEY_DOT, True),
        "/": (ecodes.KEY_SLASH, False),
        "?": (ecodes.KEY_SLASH, True),
    }

    shifted_digits = {
        "1": "!", "2": "@", "3": "#", "4": "$", "5": "%",
        "6": "^", "7": "&", "8": "*", "9": "(", "0": ")",
    }
    for d, sym in shifted_digits.items():
        code = getattr(ecodes, f"KEY_{d}")
        named[d] = code
        char_map[d] = (code, False)
        char_map[sym] = (code, True)

    for ch in "abcdefghijklmnopqrstuvwxyz":
        code = getattr(ecodes, f"KEY_{ch.upper()}")
        named[ch] = code
        char_map[ch] = (code, False)
        char_map[ch.upper()] = (code, True)

    all_keys = sorted(
        set(named.values())
        | {code for code, _ in char_map.values()}
        | {ecodes.BTN_LEFT, ecodes.BTN_RIGHT, ecodes.BTN_MIDDLE}
    )
    return named, char_map, all_keys


_NAMED_KEYS, _CHAR_KEYMAP, _ALL_EV_KEYS = _build_key_tables()


def build_uinput_capabilities() -> Dict[int, Any]:
    """Return the evdev UInput capability dict for pointer + wheel + keyboard."""
    if not HAS_EVDEV:
        return {}
    return {
        ecodes.EV_KEY: _ALL_EV_KEYS,
        ecodes.EV_REL: [ecodes.REL_WHEEL],
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


class LinuxVirtualArm:
    """Sub-millisecond Wayland pointer, scroll & keyboard actuator via persistent /dev/uinput.

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
        # Primary: Direct kernel /dev/uinput pointer + wheel + keyboard (zero subprocesses)
        if HAS_EVDEV:
            try:
                self._uinput = UInput(
                    events=build_uinput_capabilities(),
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
            code_map = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}
            if btn not in code_map:
                raise ValueError(f"Unsupported mouse button: {button!r}")
            subprocess.run(
                [self._ydotool_bin, "click", code_map[btn]],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def scroll(self, delta: int) -> None:
        """Emit vertical mouse wheel scroll (positive=up, negative=down)."""
        step = int(delta)
        if step == 0:
            return
        if self.backend == "uinput" and self._uinput is not None:
            self._uinput.write(ecodes.EV_REL, ecodes.REL_WHEEL, step)
            self._uinput.syn()
        elif self.backend == "ydotool" and self._ydotool_bin is not None:
            subprocess.run(
                [self._ydotool_bin, "mousemove", "--wheel", "-x", "0", "-y", str(step)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def _emit_keycode(self, code: int, shift: bool = False) -> None:
        if self.backend == "uinput" and self._uinput is not None:
            if shift:
                self._uinput.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 1)
                self._uinput.syn()
            self._uinput.write(ecodes.EV_KEY, code, 1)
            self._uinput.syn()
            self._uinput.write(ecodes.EV_KEY, code, 0)
            self._uinput.syn()
            if shift:
                self._uinput.write(ecodes.EV_KEY, ecodes.KEY_LEFTSHIFT, 0)
                self._uinput.syn()
        elif self.backend == "ydotool" and self._ydotool_bin is not None:
            seq = []
            if shift:
                seq.append(f"{ecodes.KEY_LEFTSHIFT}:1")
            seq.extend([f"{code}:1", f"{code}:0"])
            if shift:
                seq.append(f"{ecodes.KEY_LEFTSHIFT}:0")
            subprocess.run(
                [self._ydotool_bin, "key", *seq],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def press_key(self, key_name: str) -> None:
        """Press and release a named key (e.g. ``'enter'``, ``'esc'``, ``'tab'``, ``'a'``)."""
        norm = key_name.strip().lower()
        if norm.startswith("key_"):
            norm = norm[4:]
        if HAS_EVDEV:
            if norm in _NAMED_KEYS:
                self._emit_keycode(_NAMED_KEYS[norm], shift=False)
                return
            if len(key_name) == 1 and key_name in _CHAR_KEYMAP:
                code, shift = _CHAR_KEYMAP[key_name]
                self._emit_keycode(code, shift=shift)
                return
            raise ValueError(f"Unsupported key_name: {key_name!r}")

    def type_text(self, text: str, inter_key_delay_ms: float = 1.0) -> int:
        """Type an ASCII string via kernel keycodes. Returns number of keystrokes emitted."""
        if not text:
            return 0
        if self.backend == "ydotool" and self._ydotool_bin is not None:
            subprocess.run(
                [self._ydotool_bin, "type", "--key-delay", str(int(max(0, inter_key_delay_ms))), text],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return len(text)

        emitted = 0
        delay_s = max(0.0, float(inter_key_delay_ms)) / 1000.0
        for idx, ch in enumerate(text):
            if HAS_EVDEV and ch in _CHAR_KEYMAP:
                code, shift = _CHAR_KEYMAP[ch]
                self._emit_keycode(code, shift=shift)
                emitted += 1
            elif not HAS_EVDEV:
                emitted += 1
            if delay_s > 0.0 and idx + 1 < len(text) and self.backend != "dry_run":
                time.sleep(delay_s)
        return emitted

    def execute_motor_vector(
        self,
        motor_vec: torch.Tensor,
        threshold: float = 0.5,
        move_only: bool = False,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Decode ``[1, 6]`` motor tensor and execute movement/clicks immediately."""
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
