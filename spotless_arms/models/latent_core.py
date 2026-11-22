"""Spotless K=4 Latent Adapter and 4-bit MiniCPM-V-2_6 loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import transformers.modeling_utils
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig, PreTrainedModel


DEFAULT_MODEL_ID = "openbmb/MiniCPM-V-2_6"
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOTOR_ADAPTER_PATH = _REPO_ROOT / "checkpoints" / "best_motor_adapter.pt"
DEFAULT_ADAPTER_CANDIDATES = (
    str(Path("checkpoints") / "best_vlm_adapter.pt"),
    str(_REPO_ROOT / "checkpoints" / "best_vlm_adapter.pt"),
    "/home/rutytoi/Documents/ChienGPT/zelyr/core/vision_engine/best_vlm_adapter.pt",
    "/home/rutytoi/qwen_experiments/checkpoints/best_vlm_adapter.pt",
)
DEFAULT_MOTOR_ADAPTER_CANDIDATES = (
    str(Path("checkpoints") / "best_motor_adapter.pt"),
    str(DEFAULT_MOTOR_ADAPTER_PATH),
    *DEFAULT_ADAPTER_CANDIDATES,
)


class _TiedKeysDescriptor:
    """Compatibility shim for MiniCPM-V-2_6 with newer transformers versions."""

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        val = obj.__dict__.get("_all_tied_weights_keys", None)
        if val is None:
            val = getattr(obj, "_tied_weights_keys", None)
        return val if val is not None else {}

    def __set__(self, obj: Any, value: Any) -> None:
        obj.__dict__["_all_tied_weights_keys"] = value if value is not None else {}


_SHIMS_APPLIED = False


def apply_transformers_compat_shims() -> None:
    """Apply MiniCPM-V-2_6 & 8GB VRAM compatibility patches once."""
    global _SHIMS_APPLIED
    if _SHIMS_APPLIED:
        return
    PreTrainedModel.all_tied_weights_keys = _TiedKeysDescriptor()
    # Disable caching_allocator_warmup to prevent false-positive CUDA OOM on 8GB GPUs
    transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
    _SHIMS_APPLIED = True


def resolve_adapter_path(
    adapter_path: Optional[str] = None,
    prefer_motor_adapter: bool = False,
) -> Optional[Path]:
    """Resolve adapter checkpoint.

    When ``prefer_motor_adapter=True``, checks ``checkpoints/best_motor_adapter.pt`` first,
    falling back to ``checkpoints/best_vlm_adapter.pt``.
    """
    candidates = []
    if adapter_path:
        candidates.append(adapter_path)
    env_path = os.environ.get("VLM_ADAPTER_PATH")
    if env_path:
        candidates.append(env_path)
    candidates.extend(
        DEFAULT_MOTOR_ADAPTER_CANDIDATES if prefer_motor_adapter else DEFAULT_ADAPTER_CANDIDATES
    )

    for candidate in candidates:
        p = Path(candidate).expanduser()
        if p.is_file():
            return p
    return None


def to_device(obj: Any, device: torch.device | str) -> Any:
    """Recursively moves nested tensors (lists/tuples/dicts of tensors) to target device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, list):
        return [to_device(item, device) for item in obj]
    if isinstance(obj, tuple):
        return tuple(to_device(item, device) for item in obj)
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    return obj


class SpotlessLatentAdapter(nn.Module):
    """K=4 recurrent latent refiner + verbalizer residual MLP adapter.

    Matches the exact state_dict schema of ``best_vlm_adapter.pt``:
      - refiner.{0,2}.{weight,bias}
      - verbalizer.{0,2}.{weight,bias}
    """

    def __init__(self, d: int = 3584, bottleneck: int = 448) -> None:
        super().__init__()
        self.d = d
        self.bottleneck = bottleneck
        self.refiner = nn.Sequential(
            nn.Linear(d, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d),
        )
        self.verbalizer = nn.Sequential(
            nn.Linear(d, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, d),
        )
        for module in (self.refiner, self.verbalizer):
            nn.init.normal_(module[0].weight, std=0.01)
            nn.init.zeros_(module[0].bias)
            nn.init.normal_(module[2].weight, std=0.01)
            nn.init.zeros_(module[2].bias)

    def refine(self, h: torch.Tensor) -> torch.Tensor:
        target_dtype = self.refiner[0].weight.dtype
        h_cast = h.to(target_dtype)
        return h_cast + self.refiner(h_cast)

    def verbalize(self, h: torch.Tensor) -> torch.Tensor:
        target_dtype = self.verbalizer[0].weight.dtype
        h_cast = h.to(target_dtype)
        return h_cast + self.verbalizer(h_cast)


def load_spotless_adapter(
    adapter_path: Optional[str] = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    freeze: bool = True,
    strict: bool = True,
    prefer_motor_adapter: bool = False,
) -> Tuple[SpotlessLatentAdapter, Optional[Path]]:
    """Instantiate SpotlessLatentAdapter and load pretrained weights if found."""
    adapter = SpotlessLatentAdapter(d=3584, bottleneck=448)
    resolved = resolve_adapter_path(adapter_path, prefer_motor_adapter=prefer_motor_adapter)
    if resolved is not None:
        state_dict = torch.load(resolved, map_location="cpu", weights_only=True)
        adapter.load_state_dict(state_dict, strict=strict)
    elif adapter_path is not None:
        raise FileNotFoundError(f"Requested adapter checkpoint not found: {adapter_path}")

    adapter = adapter.to(device=device, dtype=dtype)
    if freeze:
        for param in adapter.parameters():
            param.requires_grad = False
        adapter.eval()
    return adapter, resolved


def load_minicpm_4bit(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | torch.device = "cuda",
) -> Tuple[nn.Module, AutoProcessor]:
    """Load MiniCPM-V-2_6 in 4-bit NF4 double-quantized bfloat16 for 8GB VRAM GPUs."""
    apply_transformers_compat_shims()

    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["vpm", "resampler"],
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    device_map = {"": str(device)} if isinstance(device, (str, torch.device)) else device
    try:
        base_vlm = AutoModel.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    except TypeError:
        base_vlm = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )

    for param in base_vlm.parameters():
        param.requires_grad = False
    base_vlm.eval()

    return base_vlm, processor
