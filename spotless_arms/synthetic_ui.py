"""Zero-click procedural UI renderer and live Hyprland compositor grounding harvester.

Generates pixel-exact ``(PIL.Image, prompt, x_norm, y_norm, move, left_click, right_click, done)``
samples for self-supervised MotorHead training and held-out validation with zero human clicks.
"""

from __future__ import annotations

import io
import json
import random
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

TERMINAL_PROMPTS = (
    "task complete, stop",
    "do nothing, wait",
    "target already clicked",
)

WIDGET_LABELS = (
    "Send",
    "Submit",
    "Close",
    "Cancel",
    "Settings",
    "Search",
    "New Tab",
    "Terminal",
    "Files",
    "Play",
    "Save",
    "Minimize",
    "Maximize",
    "Confirm",
)

PROMPT_TEMPLATES = (
    "Click the {label} button.",
    "Press {label}.",
    "Locate and click {label}.",
    "Select the {label} control.",
)

# Dark/slate/catppuccin-inspired themes: (desktop_bg, window_bg, bar_bg, sidebar_bg, border, text_fg)
THEMES = (
    ((17, 17, 27), (30, 30, 46), (24, 24, 37), (24, 24, 37), (69, 71, 90), (205, 214, 244)),       # Catppuccin Mocha
    ((15, 23, 42), (30, 41, 59), (15, 23, 42), (22, 33, 52), (71, 85, 105), (241, 245, 249)),      # Slate Dark
    ((18, 18, 22), (28, 28, 34), (22, 22, 28), (24, 24, 30), (63, 63, 70), (244, 244, 245)),       # Zinc Dark
    ((22, 25, 34), (35, 39, 52), (28, 31, 42), (26, 29, 40), (76, 86, 106), (236, 239, 244)),      # Nord Night
)

ACCENT_COLORS = (
    (137, 180, 250),  # Blue
    (166, 227, 161),  # Green
    (243, 139, 168),  # Red/Pink
    (250, 179, 135),  # Peach
    (203, 166, 247),  # Mauve
    (148, 226, 213),  # Teal
    (249, 226, 175),  # Yellow
    (59, 130, 246),   # Vivid Blue
    (16, 185, 129),   # Emerald
    (239, 68, 68),    # Crimson
)


@dataclass
class UISample:
    image: Image.Image
    prompt: str
    x_norm: float
    y_norm: float
    move: float = 1.0
    left_click: float = 1.0
    right_click: float = 0.0
    done: float = 0.0
    source: str = "synthetic"
    bboxes: Optional[List[Tuple[int, int, int, int]]] = None

    def target_tensor_values(self) -> List[float]:
        return [
            float(self.x_norm),
            float(self.y_norm),
            float(self.move),
            float(self.left_click),
            float(self.right_click),
            float(self.done),
        ]


def _load_font(size: int) -> ImageFont.ImageFont:
    font_candidates = (
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/noto/NotoSans-Bold.ttf",
        "/usr/share/fonts/noto/NotoSans-Regular.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for path in font_candidates:
        if Path(path).is_file():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _boxes_overlap(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
    pad: int = 12,
) -> bool:
    return not (
        a[2] + pad <= b[0]
        or b[2] + pad <= a[0]
        or a[3] + pad <= b[1]
        or b[3] + pad <= a[1]
    )


def _render_scene(
    rng: random.Random,
    width: int = 1280,
    height: int = 819,
) -> Tuple[Image.Image, List[Tuple[str, Tuple[int, int, int, int]]]]:
    """Render a realistic dark-mode desktop + app scene with crisp, non-overlapping UI widgets."""
    theme = rng.choice(THEMES)
    desktop_bg, win_bg, bar_bg, side_bg, border_col, text_fg = theme

    img = Image.new("RGB", (width, height), desktop_bg)
    draw = ImageDraw.Draw(img)

    title_font = _load_font(18)
    widget_font = _load_font(22)
    small_font = _load_font(14)

    # Top Waybar / status bar
    top_h = 36
    draw.rectangle((0, 0, width, top_h), fill=bar_bg)
    draw.line((0, top_h, width, top_h), fill=border_col, width=1)
    draw.text((16, 9), "1  2  3  4  5   |   Hyprland", fill=text_fg, font=small_font)
    draw.text((width - 150, 9), "07:30  RTX4060", fill=text_fg, font=small_font)

    # Main application window with realistic desktop margins
    win_x0 = rng.randint(36, 68)
    win_y0 = rng.randint(48, 68)
    win_x1 = width - rng.randint(36, 68)
    win_y1 = height - rng.randint(28, 52)

    draw.rounded_rectangle(
        (win_x0, win_y0, win_x1, win_y1),
        radius=12,
        fill=win_bg,
        outline=border_col,
        width=2,
    )

    # Window titlebar
    tb_h = 46
    draw.rounded_rectangle(
        (win_x0, win_y0, win_x1, win_y0 + tb_h),
        radius=12,
        fill=bar_bg,
        outline=border_col,
        width=1,
    )
    scene_kind = rng.choice(("chat", "settings", "browser", "dialog", "editor"))
    draw.text((win_x0 + 18, win_y0 + 12), f"Spotless Workspace — {scene_kind.title()}", fill=text_fg, font=title_font)

    # Left sidebar
    sb_w = rng.randint(180, 200)
    draw.rectangle((win_x0 + 2, win_y0 + tb_h, win_x0 + sb_w, win_y1 - 2), fill=side_bg)
    draw.line((win_x0 + sb_w, win_y0 + tb_h, win_x0 + sb_w, win_y1 - 2), fill=border_col, width=1)
    content_x0 = win_x0 + sb_w + 20

    # Semantic UI anchor regions (realistic desktop ergonomics with per-frame jitter)
    # Each entry: (label, anchor_x, anchor_y, btn_w, btn_h, accent_color)
    anchor_specs = [
        ("Close",    win_x1 - 110, win_y0 + 6,   96,  34, (239, 68, 68)),
        ("Maximize", win_x1 - 224, win_y0 + 6,  104,  34, (250, 179, 135)),
        ("Minimize", win_x1 - 338, win_y0 + 6,  104,  34, (249, 226, 175)),
        ("Terminal", win_x0 + 16,  win_y0 + 72, 148,  46, (137, 180, 250)),
        ("Files",    win_x0 + 16,  win_y0 + 136,148,  46, (148, 226, 213)),
        ("Settings", win_x0 + 16,  win_y1 - 68, 148,  46, (203, 166, 247)),
        ("Search",   content_x0 + 20, win_y0 + 72, 220, 46, (59, 130, 246)),
        ("New Tab",  content_x0 + 260, win_y0 + 72, 140, 46, (148, 226, 213)),
        ("Play",     content_x0 + 20,  win_y0 + 280, 128, 48, (16, 185, 129)),
        ("Save",     content_x0 + 170, win_y0 + 280, 128, 48, (137, 180, 250)),
        ("Cancel",   win_x1 - 310, win_y1 - 74, 132,  48, (243, 139, 168)),
        ("Submit",   win_x1 - 160, win_y1 - 74, 136,  48, (16, 185, 129)),
        ("Send",     win_x1 - 160, win_y1 - 142, 136, 48, (59, 130, 246)),
        ("Confirm",  content_x0 + 320, win_y1 - 142, 142, 48, (166, 227, 161)),
    ]

    # Sample 6-9 widgets per frame with small realistic jitter (+/- 12px)
    n_widgets = rng.randint(6, 9)
    chosen_specs = rng.sample(anchor_specs, k=n_widgets)
    placed: List[Tuple[str, Tuple[int, int, int, int]]] = []
    placed_boxes: List[Tuple[int, int, int, int]] = []

    for label, ax, ay, btn_w, btn_h, accent in chosen_specs:
        luma = 0.299 * accent[0] + 0.587 * accent[1] + 0.114 * accent[2]
        fg = (17, 17, 27) if luma > 140 else (245, 245, 250)

        for _attempt in range(30):
            jx = rng.randint(-10, 10)
            jy = rng.randint(-8, 8)
            x0 = max(8, min(width - btn_w - 8, ax + jx))
            y0 = max(top_h + 2, min(height - btn_h - 8, ay + jy))
            box = (x0, y0, x0 + btn_w, y0 + btn_h)
            if not any(_boxes_overlap(box, existing, pad=6) for existing in placed_boxes):
                placed_boxes.append(box)
                placed.append((label, box))

                draw.rounded_rectangle(
                    box,
                    radius=8,
                    fill=accent,
                    outline=(255, 255, 255),
                    width=2,
                )
                tb = draw.textbbox((0, 0), label, font=widget_font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
                tx = x0 + (btn_w - tw) // 2
                ty = y0 + (btn_h - th) // 2 - 2
                draw.text((tx, ty), label, fill=fg, font=widget_font)
                break

    return img, placed


def generate_synthetic_ui_samples(
    n_samples: int = 180,
    seed: int = 42,
    width: int = 1280,
    height: int = 819,
    terminal_ratio: float = 0.25,
) -> List[UISample]:
    """Generate pixel-exact procedural UI grounding + terminal halt samples."""
    rng = random.Random(seed)
    n_terminal = max(1, int(round(n_samples * terminal_ratio))) if n_samples >= 4 and terminal_ratio > 0 else 0
    n_active = n_samples - n_terminal

    samples: List[UISample] = []

    for _ in range(n_active):
        img, placed = _render_scene(rng, width=width, height=height)
        if not placed:
            continue
        label, (x0, y0, x1, y1) = rng.choice(placed)
        cx = (x0 + x1) * 0.5 / float(width)
        cy = (y0 + y1) * 0.5 / float(height)
        prompt = rng.choice(PROMPT_TEMPLATES).format(label=label)
        samples.append(
            UISample(
                image=img,
                prompt=prompt,
                x_norm=max(0.0, min(1.0, cx)),
                y_norm=max(0.0, min(1.0, cy)),
                move=1.0,
                left_click=1.0,
                right_click=0.0,
                done=0.0,
                source="synthetic_active",
                bboxes=[box for _, box in placed],
            )
        )

    for i in range(n_terminal):
        img, placed = _render_scene(rng, width=width, height=height)
        halt_prompt = TERMINAL_PROMPTS[i % len(TERMINAL_PROMPTS)]
        samples.append(
            UISample(
                image=img,
                prompt=halt_prompt,
                x_norm=0.5,
                y_norm=0.5,
                move=0.0,
                left_click=0.0,
                right_click=0.0,
                done=1.0,
                source="synthetic_halt",
                bboxes=[box for _, box in placed],
            )
        )

    rng.shuffle(samples)
    return samples


def _query_hyprland_geometry() -> Tuple[Optional[Tuple[int, int, int, int]], List[Dict[str, Any]]]:
    """Return ``((min_x, min_y, total_w, total_h), visible_clients)`` from Hyprland IPC."""
    if not shutil.which("hyprctl"):
        return None, []
    try:
        monitors = json.loads(subprocess.check_output(["hyprctl", "monitors", "-j"], text=True, timeout=1.0))
        clients = json.loads(subprocess.check_output(["hyprctl", "clients", "-j"], text=True, timeout=1.0))
    except Exception:
        return None, []

    if not monitors:
        return None, []

    active_ws_ids = {
        int(m["activeWorkspace"]["id"])
        for m in monitors
        if isinstance(m.get("activeWorkspace"), dict) and "id" in m["activeWorkspace"]
    }
    min_x = min(int(m.get("x", 0)) for m in monitors)
    min_y = min(int(m.get("y", 0)) for m in monitors)
    max_x = max(int(m.get("x", 0)) + int(m.get("width", 0)) for m in monitors)
    max_y = max(int(m.get("y", 0)) + int(m.get("height", 0)) for m in monitors)
    total_w = max(1, max_x - min_x)
    total_h = max(1, max_y - min_y)

    visible = []
    for c in clients:
        if not c.get("mapped") or c.get("hidden"):
            continue
        ws_id = c.get("workspace", {}).get("id")
        if active_ws_ids and ws_id not in active_ws_ids:
            continue
        at = c.get("at", [0, 0])
        size = c.get("size", [0, 0])
        if size[0] >= 40 and size[1] >= 40:
            visible.append(c)

    return (min_x, min_y, total_w, total_h), visible


def harvest_hyprland_samples(
    n_frames: int = 24,
    max_width: int = 1280,
    terminal_ratio: float = 0.25,
    interval_s: float = 0.05,
) -> List[UISample]:
    """Capture live Wayland desktop screenshots via grim + exact Hyprland client bounding boxes."""
    if not shutil.which("grim"):
        return []

    bounds, clients = _query_hyprland_geometry()
    if bounds is None or not clients:
        return []

    min_x, min_y, total_w, total_h = bounds
    n_terminal = max(1, int(round(n_frames * terminal_ratio))) if n_frames >= 4 else 0
    n_active = n_frames - n_terminal

    samples: List[UISample] = []
    for idx in range(n_frames):
        try:
            proc = subprocess.run(
                ["grim", "-t", "jpeg", "-q", "85", "-"],
                capture_output=True,
                check=True,
                timeout=3.0,
            )
            img = Image.open(io.BytesIO(proc.stdout)).convert("RGB")
        except Exception:
            break

        if max_width and img.width > max_width:
            scale = max_width / float(img.width)
            img = img.resize((max_width, max(1, int(round(img.height * scale)))), Image.Resampling.BILINEAR)

        if idx < n_active:
            c = clients[idx % len(clients)]
            at_x, at_y = c["at"]
            cw, ch = c["size"]
            win_class = str(c.get("class") or "window").strip()

            if idx % 2 == 0:
                # Focus window center
                cx = (at_x - min_x + cw * 0.5) / float(total_w)
                cy = (at_y - min_y + ch * 0.5) / float(total_h)
                prompt = f"focus {win_class} window"
            else:
                # Top-right close corner of window
                cx = (at_x - min_x + max(10.0, cw - 18.0)) / float(total_w)
                cy = (at_y - min_y + min(18.0, ch * 0.1)) / float(total_h)
                prompt = f"click close button on {win_class}"

            samples.append(
                UISample(
                    image=img,
                    prompt=prompt,
                    x_norm=max(0.0, min(1.0, cx)),
                    y_norm=max(0.0, min(1.0, cy)),
                    move=1.0,
                    left_click=1.0,
                    right_click=0.0,
                    done=0.0,
                    source="hyprland_active",
                )
            )
        else:
            halt_prompt = TERMINAL_PROMPTS[(idx - n_active) % len(TERMINAL_PROMPTS)]
            samples.append(
                UISample(
                    image=img,
                    prompt=halt_prompt,
                    x_norm=0.5,
                    y_norm=0.5,
                    move=0.0,
                    left_click=0.0,
                    right_click=0.0,
                    done=1.0,
                    source="hyprland_halt",
                )
            )

        if interval_s > 0 and idx + 1 < n_frames:
            time.sleep(interval_s)

    return samples
