"""Continuous MotorHead and dual-mode SpotlessVLA wrapper."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor

from spotless_arms.models.latent_core import (
    DEFAULT_MODEL_ID,
    SpotlessLatentAdapter,
    load_minicpm_4bit,
    load_spotless_adapter,
    to_device,
)


MOTOR_DIM_NAMES = (
    "x_norm",
    "y_norm",
    "move_logit",
    "left_click_logit",
    "right_click_logit",
    "done_logit",
)

DEFAULT_MOTOR_HEAD_PATH = (
    Path(__file__).resolve().parents[2] / "checkpoints" / "best_motor_head.pt"
)
DEFAULT_MOTOR_HEAD_CANDIDATES = (
    Path("checkpoints") / "best_motor_head.pt",
    DEFAULT_MOTOR_HEAD_PATH,
    Path("/home/rutytoi/Documents/ChienGPT/spotless-arms/checkpoints/best_motor_head.pt"),
)
_AUTO_MOTOR_HEAD = "__auto__"


def resolve_motor_head_path(motor_head_path: Optional[str] = _AUTO_MOTOR_HEAD) -> Optional[Path]:
    """Resolve MotorHead checkpoint preferring local ./checkpoints/best_motor_head.pt."""
    if motor_head_path is None:
        return None
    if motor_head_path == _AUTO_MOTOR_HEAD:
        for candidate in DEFAULT_MOTOR_HEAD_CANDIDATES:
            p = candidate.expanduser()
            if p.is_file():
                return p
        return None
    p = Path(motor_head_path).expanduser()
    if p.is_file():
        return p
    raise FileNotFoundError(f"Requested MotorHead checkpoint not found: {motor_head_path}")


class MotorHead(nn.Module):
    """Lightweight MLP projecting RMSNorm(h_verb) directly to a 6-DoF motor vector.

    Output layout (shape ``[B, 6]``):
      - ``0``: ``x_norm`` (sigmoid -> ``[0.0, 1.0]`` normalized screen X)
      - ``1``: ``y_norm`` (sigmoid -> ``[0.0, 1.0]`` normalized screen Y)
      - ``2``: ``move_logit`` (raw logit: whether to move cursor)
      - ``3``: ``left_click_logit`` (raw logit: trigger left click)
      - ``4``: ``right_click_logit`` (raw logit: trigger right click)
      - ``5``: ``done_logit`` (raw logit: task complete / halt reflex loop)
    """

    def __init__(
        self,
        hidden_size: int = 3584,
        bottleneck: int = 512,
        out_dim: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bottleneck = bottleneck
        self.out_dim = out_dim

        self.net = nn.Sequential(
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.LayerNorm(bottleneck),
            nn.Linear(bottleneck, out_dim),
        )
        # Small init on output projection so initial coordinates start centered (~0.5)
        # and action logits start near 0.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward_raw(self, normed_h: torch.Tensor) -> torch.Tensor:
        """Return raw pre-sigmoid 6D motor tensor of shape ``[B, 6]``."""
        if normed_h.ndim == 3:
            normed_h = normed_h[:, -1, :]
        target_dtype = self.net[0].weight.dtype
        return self.net(normed_h.to(target_dtype))

    def forward(self, normed_h: torch.Tensor) -> torch.Tensor:
        """Return ``[B, 6]`` tensor with sigmoid applied to ``(x_norm, y_norm)``."""
        raw = self.forward_raw(normed_h)
        coords = torch.sigmoid(raw[:, :2])
        logits = raw[:, 2:]
        return torch.cat([coords, logits], dim=-1)


def decode_motor_vector(motor_vec: torch.Tensor, threshold: float = 0.5) -> Dict[str, Any]:
    """Decode a ``[1, 6]`` or ``[6]`` motor tensor into Python floats & booleans."""
    vec = motor_vec.detach().float().cpu().reshape(-1)
    if vec.numel() != 6:
        raise ValueError(f"Expected 6-element motor vector, got shape {tuple(motor_vec.shape)}")

    x_norm = float(vec[0].clamp(0.0, 1.0).item())
    y_norm = float(vec[1].clamp(0.0, 1.0).item())
    move_logit = float(vec[2].item())
    left_click_logit = float(vec[3].item())
    right_click_logit = float(vec[4].item())
    done_logit = float(vec[5].item())

    move_prob = float(torch.sigmoid(vec[2]).item())
    left_click_prob = float(torch.sigmoid(vec[3]).item())
    right_click_prob = float(torch.sigmoid(vec[4]).item())
    done_prob = float(torch.sigmoid(vec[5]).item())

    return {
        "x_norm": x_norm,
        "y_norm": y_norm,
        "move_logit": move_logit,
        "left_click_logit": left_click_logit,
        "right_click_logit": right_click_logit,
        "done_logit": done_logit,
        "move_prob": move_prob,
        "left_click_prob": left_click_prob,
        "right_click_prob": right_click_prob,
        "done_prob": done_prob,
        "should_move": move_prob >= threshold,
        "left_click": left_click_prob >= threshold,
        "right_click": right_click_prob >= threshold,
        "done": done_prob >= threshold,
    }


class SpotlessVLA(nn.Module):
    """Dual-mode Vision-Language-Action model built on MiniCPM-V-2_6 + K=4 SpotlessLatentAdapter.

    - ``forward_motor(...)``: Single-pass continuous control bypassing ``lm_head`` completely.
    - ``forward_text(...)``: Original autoregressive JSON generation for distillation/debugging.
    """

    def __init__(
        self,
        base_vlm: nn.Module,
        adapter: SpotlessLatentAdapter,
        motor_head: Optional[MotorHead] = None,
        processor: Optional[AutoProcessor] = None,
    ) -> None:
        super().__init__()
        self.vlm = base_vlm
        self.adapter = adapter
        self.processor = processor

        adapter_param = next(self.adapter.parameters())
        if motor_head is None:
            motor_head = MotorHead(hidden_size=adapter.d, bottleneck=512, out_dim=6).to(
                device=adapter_param.device,
                dtype=adapter_param.dtype,
            )
        self.motor_head = motor_head

        embed_weights = self.vlm.llm.model.embed_tokens.weight.data
        self.register_buffer("embed_norm", embed_weights.norm(p=2, dim=-1).mean())
        self._formatted_prompt_cache: Dict[str, str] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        adapter_path: Optional[str] = None,
        motor_head_path: Optional[str] = _AUTO_MOTOR_HEAD,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "SpotlessVLA":
        """Load 4-bit MiniCPM-V-2_6, pretrained K=4 adapter, and trained MotorHead weights.

        By default (``motor_head_path=_AUTO_MOTOR_HEAD``), automatically loads
        ``checkpoints/best_motor_head.pt`` if present. Pass ``motor_head_path=None``
        to force a freshly initialized ``MotorHead``.
        """
        base_vlm, processor = load_minicpm_4bit(model_id=model_id, device=device)
        adapter, resolved_adapter = load_spotless_adapter(
            adapter_path=adapter_path,
            device=device,
            dtype=dtype,
            freeze=True,
        )
        print(f"[SpotlessVLA] Loaded K=4 adapter from: {resolved_adapter}")

        motor_head = MotorHead(hidden_size=3584, bottleneck=512, out_dim=6).to(
            device=device,
            dtype=dtype,
        )
        resolved_mh = resolve_motor_head_path(motor_head_path)

        if resolved_mh is not None:
            sd = torch.load(resolved_mh, map_location="cpu", weights_only=True)
            motor_head.load_state_dict(sd, strict=True)
            motor_head = motor_head.to(device=device, dtype=dtype)
            print(f"[SpotlessVLA] Loaded MotorHead from: {resolved_mh}")
        else:
            print("[SpotlessVLA] Using freshly initialized MotorHead (no checkpoint loaded).")

        model = cls(base_vlm=base_vlm, adapter=adapter, motor_head=motor_head, processor=processor)
        model.eval()
        return model

    @property
    def device(self) -> torch.device:
        return next(self.adapter.parameters()).device

    def format_prompt(self, prompt: str) -> str:
        """Apply and cache the MiniCPM-V-2_6 chat template once per prompt string."""
        if self.processor is None:
            raise RuntimeError("SpotlessVLA.processor is required for format_prompt")
        cached = self._formatted_prompt_cache.get(prompt)
        if cached is not None:
            return cached
        copy_msgs = [{"role": "user", "content": f"(<image>./</image>)\n{prompt}"}]
        formatted = self.processor.tokenizer.apply_chat_template(
            copy_msgs, tokenize=False, add_generation_prompt=True
        )
        self._formatted_prompt_cache[prompt] = formatted
        return formatted

    def prepare_multimodal_data_cpu(
        self,
        image: Image.Image,
        prompt: str,
        max_slice_nums: int = 1,
    ) -> Dict[str, Any]:
        """Run CPU image slicing + tokenization (safe to overlap on a background thread)."""
        if self.processor is None:
            raise RuntimeError("SpotlessVLA.processor is required for prepare_multimodal_data_cpu")

        if image.mode != "RGB":
            image = image.convert("RGB")

        formatted_prompt = self.format_prompt(prompt)
        inputs = self.processor(
            [formatted_prompt],
            [[image]],
            max_slice_nums=max_slice_nums,
            return_tensors="pt",
        )
        return dict(inputs)

    def prepare_multimodal_data(
        self,
        image: Image.Image,
        prompt: str,
        max_slice_nums: int = 1,
    ) -> Dict[str, Any]:
        """Format image + prompt into MiniCPM-V-2_6 multimodal tensor dict on CUDA."""
        cpu_inputs = self.prepare_multimodal_data_cpu(
            image, prompt, max_slice_nums=max_slice_nums
        )
        return to_device(cpu_inputs, self.device)

    def extract_latent_state(
        self,
        data: Dict[str, Any],
        K: int = 4,
        k_steps: Optional[int] = None,
        return_pkv: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Any]]:
        """Run vision encoder + prefill + adaptive-K recurrent deliberation + verbalizer + RMSNorm.

        Returns ``(normed_h, pkv_or_none)`` where ``normed_h`` has shape ``[B, 1, 3584]``.
        All operations remain purely on CUDA with zero host syncs inside the latent loop.
        """
        steps = K if k_steps is None else k_steps
        inputs_embeds, _ = self.vlm.get_vllm_embedding(data)
        model_dtype = self.vlm.llm.model.embed_tokens.weight.dtype

        out = self.vlm.llm.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )
        pkv = out.past_key_values
        h = out.last_hidden_state[:, -1:, :]
        del inputs_embeds, out

        # Adaptive-K Recurrent Deliberation Loop with hypersphere norm alignment
        for _ in range(steps):
            h_ref = self.adapter.refine(h)
            norm_val = h_ref.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
            e_lat = ((h_ref / norm_val) * self.embed_norm).to(model_dtype)

            out = self.vlm.llm.model(
                inputs_embeds=e_lat,
                past_key_values=pkv,
                use_cache=True,
            )
            pkv = out.past_key_values
            h = out.last_hidden_state[:, -1:, :]
            del h_ref, norm_val, e_lat, out

        # Verbalizer + RMSNorm (final representation before head)
        h_verb = self.adapter.verbalize(h)
        norm_dtype = self.vlm.llm.model.norm.weight.dtype
        normed_h = self.vlm.llm.model.norm(h_verb.to(norm_dtype))
        del h, h_verb

        if not return_pkv:
            del pkv
            return normed_h, None
        return normed_h, pkv

    def forward_motor(
        self,
        pixel_values: Optional[Any] = None,
        input_ids: Optional[torch.Tensor] = None,
        *,
        data: Optional[Dict[str, Any]] = None,
        image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        K: int = 4,
        k_steps: Optional[int] = None,
        max_slice_nums: int = 1,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Single-pass continuous motor forward (zero text detokenization, skips lm_head).

        Supports adaptive latent cycles via ``k_steps`` (defaults to ``K=4`` for backward compatibility).

        Returns:
            ``motor_tensor`` of shape ``[B, 6]`` (``[1, 6]`` for single frame).
        """
        if data is None:
            if image is not None and prompt is not None:
                data = self.prepare_multimodal_data(image, prompt, max_slice_nums=max_slice_nums)
            elif pixel_values is not None and input_ids is not None:
                data = {"pixel_values": pixel_values, "input_ids": input_ids}
                if extra_inputs:
                    data.update(extra_inputs)
                data = to_device(data, self.device)
            else:
                raise ValueError(
                    "Provide either (pixel_values, input_ids), data=dict, or (image, prompt)."
                )

        steps = K if k_steps is None else k_steps
        normed_h, _ = self.extract_latent_state(data, k_steps=steps, return_pkv=False)
        motor_tensor = self.motor_head(normed_h)
        return motor_tensor

    def forward_text(
        self,
        image: Optional[Image.Image] = None,
        prompt: Optional[str] = None,
        *,
        data: Optional[Dict[str, Any]] = None,
        K: int = 4,
        max_tokens: int = 32,
        max_slice_nums: int = 1,
        return_normed_h: bool = False,
    ) -> Dict[str, Any]:
        """Preserved autoregressive lm_head JSON path for teacher distillation & debugging."""
        if self.processor is None:
            raise RuntimeError("SpotlessVLA.processor is required for forward_text")

        t0 = time.perf_counter()
        if data is None:
            if image is None or prompt is None:
                raise ValueError("Provide either data=dict or (image, prompt).")
            data = self.prepare_multimodal_data(image, prompt, max_slice_nums=max_slice_nums)

        generated_ids = []
        normed_h_out: Optional[torch.Tensor] = None
        eos_ids = {self.processor.tokenizer.eos_token_id, 151645}

        with torch.inference_mode():
            normed_h, pkv = self.extract_latent_state(data, K=K, return_pkv=True)
            if return_normed_h:
                normed_h_out = normed_h.detach().clone()

            logits = self.vlm.llm.lm_head(normed_h)
            token_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_ids.append(int(token_id.item()))
            del logits

            for _ in range(max_tokens - 1):
                if int(token_id.item()) in eos_ids:
                    break
                token_embeds = self.vlm.llm.model.embed_tokens(token_id)
                out = self.vlm.llm.model(
                    inputs_embeds=token_embeds,
                    past_key_values=pkv,
                    use_cache=True,
                )
                pkv = out.past_key_values
                normed = self.vlm.llm.model.norm(out.last_hidden_state[:, -1:, :])
                next_logits = self.vlm.llm.lm_head(normed)
                token_id = torch.argmax(next_logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids.append(int(token_id.item()))
                del token_embeds, out, normed, next_logits

            del pkv, normed_h, token_id

        latency_ms = (time.perf_counter() - t0) * 1000.0
        response = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        result: Dict[str, Any] = {
            "action": "unknown",
            "point": [0, 0],
            "text": None,
            "raw_output": response,
            "num_tokens": len(generated_ids),
            "latency_ms": latency_ms,
        }
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                if "action" in parsed:
                    result["action"] = parsed["action"]
                if "point" in parsed and isinstance(parsed["point"], list) and len(parsed["point"]) == 2:
                    result["point"] = parsed["point"]
                if "text" in parsed:
                    result["text"] = parsed["text"]
        except json.JSONDecodeError:
            pass

        if return_normed_h:
            result["normed_h"] = normed_h_out

        return result
