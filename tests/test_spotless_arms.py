"""End-to-end & unit verification tests for Spotless-Arms."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from PIL import Image

from spotless_arms.actuators.uinput_arm import LinuxVirtualArm
from spotless_arms.models.latent_core import load_spotless_adapter
from spotless_arms.models.motor_head import MotorHead, SpotlessVLA, decode_motor_vector
from spotless_arms.pipeline.pipelined_runner import AsyncScreenBuffer, PipelinedVLARunner
from spotless_arms.train_motor_head import (
    compute_motor_loss,
    parse_teacher_json_to_target,
    train_motor_head_from_cache,
)


class _DummyLLMModel(nn.Module):
    def __init__(self, d: int = 3584, vocab: int = 128):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, d)
        self.norm = nn.RMSNorm(d)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, inputs_embeds, past_key_values=None, use_cache=True):
        out = self.proj(inputs_embeds)
        pkv = (past_key_values or 0) + inputs_embeds.shape[1]
        return SimpleNamespace(last_hidden_state=out, past_key_values=pkv)


class _DummyVLM(nn.Module):
    def __init__(self, d: int = 3584, vocab: int = 128):
        super().__init__()
        self.llm = nn.Module()
        self.llm.model = _DummyLLMModel(d=d, vocab=vocab)
        self.llm.lm_head = nn.Linear(d, vocab, bias=False)
        self.d = d

    def get_vllm_embedding(self, data):
        device = self.llm.model.embed_tokens.weight.device
        dtype = self.llm.model.embed_tokens.weight.dtype
        return torch.randn(1, 16, self.d, device=device, dtype=dtype), None


class _DummyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return msgs[0]["content"]

    def decode(self, ids, skip_special_tokens=True):
        return '{"action": "click", "point": [500, 300]}'


class _DummyProcessor:
    def __init__(self):
        self.tokenizer = _DummyTokenizer()

    def __call__(self, prompts, images, max_slice_nums=1, return_tensors="pt"):
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "pixel_values": [torch.zeros(1, 3, 224, 224)],
        }


class SpotlessArmsTestSuite(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

    def test_adapter_loads_pretrained_weights_strictly(self) -> None:
        adapter, resolved = load_spotless_adapter(device=self.device, dtype=self.dtype, strict=True)
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.is_file())
        x = torch.randn(1, 1, 3584, device=self.device, dtype=self.dtype)
        y = adapter.verbalize(adapter.refine(x))
        self.assertEqual(tuple(y.shape), (1, 1, 3584))

    def test_motor_head_and_dual_mode_vla(self) -> None:
        adapter, _ = load_spotless_adapter(device=self.device, dtype=self.dtype)
        base_vlm = _DummyVLM().to(device=self.device, dtype=self.dtype)
        motor_head = MotorHead(hidden_size=3584, bottleneck=512, out_dim=6).to(
            device=self.device, dtype=self.dtype
        )
        vla = SpotlessVLA(
            base_vlm=base_vlm,
            adapter=adapter,
            motor_head=motor_head,
            processor=_DummyProcessor(),
        )
        img = Image.new("RGB", (640, 480), color="gray")

        # 1. forward_motor via (image, prompt) with adaptive k_steps
        with torch.no_grad():
            motor_vec = vla.forward_motor(image=img, prompt="Click the button", k_steps=4)
            motor_vec_k1 = vla.forward_motor(image=img, prompt="Click the button", k_steps=1)
        self.assertEqual(tuple(motor_vec.shape), (1, 6))
        self.assertEqual(tuple(motor_vec_k1.shape), (1, 6))
        self.assertTrue(0.0 <= float(motor_vec[0, 0].item()) <= 1.0)
        self.assertTrue(0.0 <= float(motor_vec[0, 1].item()) <= 1.0)

        # 2. forward_motor via (pixel_values, input_ids)
        motor_vec2 = vla.forward_motor(
            pixel_values=[torch.zeros(1, 3, 224, 224)],
            input_ids=torch.tensor([[1, 2, 3]]),
            K=4,
        )
        self.assertEqual(tuple(motor_vec2.shape), (1, 6))

        # 3. forward_text preserved for distillation
        res = vla.forward_text(image=img, prompt="Click the button", K=4, max_tokens=4, return_normed_h=True)
        self.assertEqual(res["action"], "click")
        self.assertEqual(res["point"], [500, 300])
        self.assertEqual(tuple(res["normed_h"].shape), (1, 1, 3584))

        tgt = parse_teacher_json_to_target(res)
        self.assertIsNotNone(tgt)
        self.assertAlmostEqual(float(tgt[0]), 0.5, places=4)
        self.assertAlmostEqual(float(tgt[1]), 0.3, places=4)
        self.assertEqual(float(tgt[2]), 1.0)
        self.assertEqual(float(tgt[3]), 1.0)

    def test_live_wayland_pipelined_runner_move_only(self) -> None:
        adapter, _ = load_spotless_adapter(device=self.device, dtype=self.dtype)
        base_vlm = _DummyVLM().to(device=self.device, dtype=self.dtype)
        vla = SpotlessVLA(
            base_vlm=base_vlm,
            adapter=adapter,
            processor=_DummyProcessor(),
        )
        with AsyncScreenBuffer(jpeg_quality=80) as sb, LinuxVirtualArm(dry_run=False) as arm:
            runner = PipelinedVLARunner(vla=vla, arm=arm, screen_buffer=sb)
            self.assertEqual(runner.k_initial, 2)
            self.assertEqual(runner.k_track, 1)
            telemetry = runner.run(
                prompt="Locate window center",
                max_ticks=4,
                done_logit_threshold=999.0,
                move_only=True,
                verbose=False,
            )
            self.assertEqual(len(telemetry), 4)
            # Adaptive-K default schedule: Tick 1 uses K=2, Ticks 2+ use K=1
            self.assertEqual([t["k_steps"] for t in telemetry], [2, 1, 1, 1])
            # Zero-wait overlapped fetch on subsequent ticks
            self.assertLess(telemetry[-1]["fetch_ms"], 5.0)
            # Sub-millisecond kernel actuation
            self.assertLess(telemetry[-1]["actuate_ms"], 1.0)
            self.assertEqual(arm.backend, "uinput")

    def test_auto_halt_on_single_click_safety(self) -> None:
        adapter, _ = load_spotless_adapter(device=self.device, dtype=self.dtype)
        base_vlm = _DummyVLM().to(device=self.device, dtype=self.dtype)
        motor_head = MotorHead(hidden_size=3584, bottleneck=512, out_dim=6).to(
            device=self.device, dtype=self.dtype
        )
        # Force strong positive move + left_click logits and negative done logit
        with torch.no_grad():
            motor_head.net[-1].weight.zero_()
            motor_head.net[-1].bias.copy_(
                torch.tensor([0.0, 0.0, 5.0, 5.0, -5.0, -5.0], device=self.device, dtype=self.dtype)
            )
        vla = SpotlessVLA(
            base_vlm=base_vlm,
            adapter=adapter,
            motor_head=motor_head,
            processor=_DummyProcessor(),
        )
        with AsyncScreenBuffer(jpeg_quality=80) as sb, LinuxVirtualArm(dry_run=True) as arm:
            runner = PipelinedVLARunner(
                vla=vla, arm=arm, screen_buffer=sb, k_initial=4, k_track=1, auto_halt_on_click=True
            )
            telemetry = runner.run(
                prompt="Click submit",
                max_ticks=10,
                move_only=False,
                verbose=False,
            )
            self.assertEqual(len(telemetry), 1)
            self.assertEqual(telemetry[0]["action"]["clicked"], "left")
            self.assertEqual(telemetry[0]["halted_reason"], "click:left")

    def test_balanced_done_training_separates_active_and_halt(self) -> None:
        torch.manual_seed(42)
        active_feats = torch.randn(12, 3584) + 1.0
        halt_feats = torch.randn(4, 3584) - 1.0
        features = torch.cat([active_feats, halt_feats], dim=0)
        targets = torch.cat(
            [
                torch.tensor([[0.25, 0.75, 1.0, 1.0, 0.0, 0.0]] * 12),
                torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0, 1.0]] * 4),
            ],
            dim=0,
        )
        mh = MotorHead(hidden_size=3584, bottleneck=512, out_dim=6)

        with torch.no_grad():
            loss_before, _ = compute_motor_loss(mh(features), targets)

        trained = train_motor_head_from_cache(
            mh, features, targets, epochs=25, batch_size=8, lr=1e-3, device=self.device
        )
        with torch.no_grad():
            preds = trained.cpu()(features)
            loss_after, _ = compute_motor_loss(preds, targets)

        self.assertLess(float(loss_after), float(loss_before) * 0.1)
        active_dec = decode_motor_vector(preds[0])
        halt_dec = decode_motor_vector(preds[-1])
        self.assertTrue(active_dec["should_move"])
        self.assertFalse(active_dec["done"])
        self.assertGreater(halt_dec["done_logit"], 0.0)
        self.assertTrue(halt_dec["done"])
        self.assertFalse(halt_dec["should_move"])

    def test_record_demos_cli_optional_prompt(self) -> None:
        from spotless_arms.record_demos import build_arg_parser

        parser = build_arg_parser()
        args = parser.parse_args([])
        self.assertEqual(args.prompt, "click target UI element")
        self.assertFalse(args.interactive_prompt)

        args_interactive = parser.parse_args(["--interactive-prompt", "--max-samples", "5"])
        self.assertTrue(args_interactive.interactive_prompt)
        self.assertEqual(args_interactive.max_samples, 5)

    def test_record_demos_hover_snap_fallback(self) -> None:
        import io
        from unittest.mock import patch
        from spotless_arms import record_demos

        buf = io.BytesIO()
        Image.new("RGB", (1920, 1080), color="navy").save(buf, format="JPEG")
        fake_jpeg = buf.getvalue()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with (
                patch.object(record_demos, "find_pointer_devices", return_value=[]),
                patch.object(record_demos, "capture_frame_jpeg", return_value=(fake_jpeg, 1920, 1080)),
                patch.object(record_demos, "get_cursor_position_wayland", return_value=(960, 540)),
                patch.object(record_demos, "get_display_dimensions_wayland", return_value=(1920, 1080)),
                patch("builtins.input", side_effect=["click test button", "q"]),
            ):
                manifest = record_demos.record_demonstrations(
                    output_dir=out_dir,
                    max_samples=3,
                )

            self.assertTrue(manifest.is_file())
            lines = [json.loads(l) for l in manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertEqual(len(lines), 1)
            rec = lines[0]
            self.assertEqual(rec["prompt"], "click test button")
            self.assertEqual(rec["button"], "left")
            self.assertAlmostEqual(rec["x_norm"], 0.5, places=4)
            self.assertAlmostEqual(rec["y_norm"], 0.5, places=4)
            self.assertTrue((out_dir / rec["image"]).is_file())

    def test_ema_damping_and_early_convergence_halt(self) -> None:
        from unittest.mock import patch

        adapter, _ = load_spotless_adapter(device=self.device, dtype=self.dtype)
        base_vlm = _DummyVLM().to(device=self.device, dtype=self.dtype)
        vla = SpotlessVLA(base_vlm=base_vlm, adapter=adapter, processor=_DummyProcessor())

        # Sequence: Tick 1 snaps to (0.2, 0.2); Tick 2 jumps to (0.6, 0.6) -> EMA(0.75) = (0.5, 0.5);
        # Ticks 3 & 4 stay at (0.501, 0.501) -> within tol=0.008 for patience=2 -> halts at Tick 4!
        seq = [
            torch.tensor([[0.200, 0.200, 5.0, -5.0, -5.0, -5.0]]),
            torch.tensor([[0.600, 0.600, 5.0, -5.0, -5.0, -5.0]]),
            torch.tensor([[0.501, 0.501, 5.0, -5.0, -5.0, -5.0]]),
            torch.tensor([[0.501, 0.501, 5.0, -5.0, -5.0, -5.0]]),
            torch.tensor([[0.501, 0.501, 5.0, -5.0, -5.0, -5.0]]),
        ]
        it = iter(seq)

        with AsyncScreenBuffer(jpeg_quality=80) as sb, LinuxVirtualArm(dry_run=True) as arm:
            runner = PipelinedVLARunner(
                vla=vla,
                arm=arm,
                screen_buffer=sb,
                k_initial=4,
                k_track=1,
                ema_alpha=0.75,
                convergence_tol=0.008,
                convergence_patience=2,
                halt_on_convergence=True,
            )
            with patch.object(vla, "forward_motor", side_effect=lambda **kw: next(it)):
                telemetry = runner.run(
                    prompt="Track stationary button",
                    max_ticks=8,
                    move_only=True,
                    verbose=False,
                )

        self.assertEqual(len(telemetry), 4)
        self.assertAlmostEqual(telemetry[0]["action"]["x_norm"], 0.200, places=3)
        self.assertAlmostEqual(telemetry[1]["action"]["x_norm"], 0.500, places=3)
        self.assertTrue(telemetry[-1]["converged"])
        self.assertEqual(telemetry[-1]["converged_tick"], 4)
        self.assertEqual(telemetry[-1]["halted_reason"], "converged")
        self.assertTrue(runner.last_summary["converged"])
        self.assertEqual(runner.last_summary["converged_tick"], 4)
        self.assertGreater(runner.last_summary["jitter_px"], 0.0)

    def test_uinput_keyboard_and_scroll_primitives(self) -> None:
        import time
        from evdev import ecodes
        from spotless_arms.actuators.uinput_arm import build_uinput_capabilities

        caps = build_uinput_capabilities()
        self.assertIn(ecodes.EV_REL, caps)
        self.assertIn(ecodes.REL_WHEEL, caps[ecodes.EV_REL])
        self.assertIn(ecodes.KEY_ENTER, caps[ecodes.EV_KEY])
        self.assertIn(ecodes.KEY_ESC, caps[ecodes.EV_KEY])
        self.assertIn(ecodes.KEY_A, caps[ecodes.EV_KEY])

        # Verify dry_run keyboard & scroll latency < 2ms/keystroke without touching real desktop
        with LinuxVirtualArm(dry_run=True) as arm:
            t0 = time.perf_counter()
            arm.scroll(-3)
            arm.press_key("enter")
            arm.press_key("esc")
            arm.press_key("tab")
            n = arm.type_text("Hello, Spotless-Arms 2026!", inter_key_delay_ms=1.0)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.assertEqual(n, len("Hello, Spotless-Arms 2026!"))
            self.assertLess(elapsed_ms / max(n, 1), 2.0)

        # Verify real /dev/uinput device registers extended capabilities cleanly (zero keystrokes emitted)
        with LinuxVirtualArm(dry_run=False) as live_arm:
            self.assertEqual(live_arm.backend, "uinput")

    def test_foveated_cropper_and_dual_view_prep(self) -> None:
        from spotless_arms.models.motor_head import FoveatedCropper

        cropper = FoveatedCropper(global_width=896, crop_size=336)
        src = Image.new("RGB", (3000, 1920), color="teal")
        g_view, f_crop = cropper.build_views(src, cursor_x=0.0, cursor_y=1.0)
        self.assertEqual(g_view.size, (896, 573))
        self.assertEqual(f_crop.size, (336, 336))

        adapter, _ = load_spotless_adapter(device=self.device, dtype=self.dtype)
        base_vlm = _DummyVLM().to(device=self.device, dtype=self.dtype)
        vla = SpotlessVLA(base_vlm=base_vlm, adapter=adapter, processor=_DummyProcessor())
        cpu_data = vla.prepare_multimodal_data_cpu(src, "Click icon", foveated=True, cursor_pos=(0.1, 0.2))
        self.assertIn("input_ids", cpu_data)
        self.assertIn("pixel_values", cpu_data)

    def test_synthetic_ui_bounds_non_overlap_and_auto_convergence(self) -> None:
        from spotless_arms.synthetic_ui import _boxes_overlap, generate_synthetic_ui_samples

        samples = generate_synthetic_ui_samples(n_samples=20, seed=123, terminal_ratio=0.25)
        self.assertEqual(len(samples), 20)
        n_halt = sum(1 for s in samples if s.done >= 0.5)
        n_active = sum(1 for s in samples if s.done < 0.5)
        self.assertEqual(n_halt, 5)
        self.assertEqual(n_active, 15)

        for s in samples:
            self.assertEqual(s.image.size, (1280, 819))
            self.assertTrue(0.0 <= s.x_norm <= 1.0)
            self.assertTrue(0.0 <= s.y_norm <= 1.0)
            self.assertIsNotNone(s.bboxes)
            boxes = s.bboxes or []
            self.assertGreaterEqual(len(boxes), 4)
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    self.assertFalse(_boxes_overlap(boxes[i], boxes[j], pad=0))

        # Verify auto-mode training pipeline converges cleanly on synthetic features
        torch.manual_seed(7)
        feats = torch.randn(20, 3584)
        tgts = torch.tensor([s.target_tensor_values() for s in samples], dtype=torch.float32)
        mh = MotorHead(hidden_size=3584, bottleneck=768, out_dim=6, dropout=0.15)
        with torch.no_grad():
            loss_before, _ = compute_motor_loss(mh(feats), tgts)
        trained = train_motor_head_from_cache(
            mh, feats, tgts, epochs=25, batch_size=8, lr=5e-4, device=self.device
        )
        with torch.no_grad():
            loss_after, _ = compute_motor_loss(trained.cpu()(feats), tgts)
        self.assertLess(float(loss_after), float(loss_before) * 0.15)

    def test_joint_unfreeze_adapter_grad_flow_and_resolution(self) -> None:
        from spotless_arms.models.latent_core import resolve_adapter_path
        from spotless_arms.synthetic_ui import generate_synthetic_ui_samples
        from spotless_arms.train_motor_head import train_joint_adapter_and_motor_head

        adapter, _ = load_spotless_adapter(device=self.device, dtype=torch.float32, freeze=False)
        base_vlm = _DummyVLM().to(device=self.device, dtype=torch.float32)
        for p in base_vlm.parameters():
            p.requires_grad = False
        mh = MotorHead(hidden_size=3584, bottleneck=128, out_dim=6, dropout=0.0).to(
            device=self.device, dtype=torch.float32
        )
        vla = SpotlessVLA(base_vlm=base_vlm, adapter=adapter, motor_head=mh, processor=_DummyProcessor())

        ref_before = vla.adapter.refiner[0].weight.detach().clone()
        verb_before = vla.adapter.verbalizer[0].weight.detach().clone()
        mh_before = vla.motor_head.net[0].weight.detach().clone()

        tr = generate_synthetic_ui_samples(n_samples=8, seed=11, terminal_ratio=0.25)
        va = generate_synthetic_ui_samples(n_samples=4, seed=22, terminal_ratio=0.25)
        res = train_joint_adapter_and_motor_head(
            vla=vla, train_samples=tr, val_samples=va, epochs=2, grad_accum_steps=4, k_steps=2
        )
        self.assertIn("cos_before", res)
        self.assertIn("cos_after", res)
        # Verify gradients updated BOTH SpotlessLatentAdapter (refiner + verbalizer) and MotorHead
        self.assertFalse(torch.allclose(ref_before, vla.adapter.refiner[0].weight.detach()))
        self.assertFalse(torch.allclose(verb_before, vla.adapter.verbalizer[0].weight.detach()))
        self.assertFalse(torch.allclose(mh_before, vla.motor_head.net[0].weight.detach()))

        # Verify prefer_motor_adapter resolves cleanly
        resolved = resolve_adapter_path(prefer_motor_adapter=True)
        self.assertIsNotNone(resolved)
        self.assertTrue(resolved.is_file())


if __name__ == "__main__":
    unittest.main()
