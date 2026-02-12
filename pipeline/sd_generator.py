"""
SD Generator — Stable Diffusion image generation with LoRA support
===================================================================
Uses HuggingFace ``diffusers`` to load an SDXL (or SD-1.5) checkpoint
and generate panel images.

LoRA Integration
----------------
**Drop-in design**: place ``.safetensors`` files into the configured folders
(``loras/character/`` and ``loras/style/`` by default) and they will be
loaded automatically — no code changes required.

Usage::

    gen = SDGenerator.from_config(cfg["sd"])
    images = gen.generate_panels(panel_prompts)
"""

import logging
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from PIL import Image

logger = logging.getLogger(__name__)

# ── Lazy imports for heavy ML libs (fail fast with clear message) ────
_DIFFUSERS_AVAILABLE = False
try:
    import torch
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionPipeline,
        DPMSolverMultistepScheduler,
    )
    _DIFFUSERS_AVAILABLE = True
except ImportError:
    pass


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA adapter."""
    name: str           # human-readable label ("character", "style", …)
    path: str           # path to .safetensors file
    weight: float = 0.8 # adapter strength


class SDGenerator:
    """
    Generate images using Stable Diffusion + LoRA.

    Parameters
    ----------
    model_path : str
        HuggingFace model ID or local path to diffusion checkpoint.
    lora_configs : list[LoRAConfig]
        LoRA adapters to load.  Discovered automatically by ``from_config``.
    guidance_scale : float
    num_inference_steps : int
    default_width, default_height : int
        Default panel resolution.
    seed : int | None
        Global seed for reproducibility.  ``None`` = random.
    """

    def __init__(
        self,
        model_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_configs: Optional[List[LoRAConfig]] = None,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        default_width: int = 768,
        default_height: int = 512,
        seed: Optional[int] = None,
    ):
        self.model_path = model_path
        self.lora_configs = lora_configs or []
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.default_width = default_width
        self.default_height = default_height
        self.seed = seed

        self._pipe = None  # lazy-loaded

    # ── Factory from config dict ─────────────────────────────────────
    @classmethod
    def from_config(cls, sd_cfg: Dict[str, Any]) -> "SDGenerator":
        """
        Build an ``SDGenerator`` from the ``sd:`` section of config.yaml.

        LoRA weights are **auto-discovered** — just drop .safetensors files
        into the configured directories and they'll be picked up.
        """
        lora_cfgs: List[LoRAConfig] = []

        lora_section = sd_cfg.get("lora", {})
        for adapter_name in ("character", "style"):
            adapter = lora_section.get(adapter_name, {})
            lora_dir = adapter.get("dir")
            weight = adapter.get("weight", 0.8)

            if lora_dir:
                found = cls._discover_safetensors(lora_dir)
                for p in found:
                    lora_cfgs.append(LoRAConfig(
                        name=f"{adapter_name}/{Path(p).stem}",
                        path=p,
                        weight=weight,
                    ))
                if not found:
                    logger.info(
                        f"No .safetensors found in {lora_dir!r} for "
                        f"{adapter_name} LoRA — skipping."
                    )

        panel_size = sd_cfg.get("panel_size", sd_cfg.get("layout", {}).get("panel_size", {}))

        return cls(
            model_path=sd_cfg.get("model_path",
                                  "stabilityai/stable-diffusion-xl-base-1.0"),
            lora_configs=lora_cfgs,
            guidance_scale=sd_cfg.get("guidance_scale", 7.5),
            num_inference_steps=sd_cfg.get("num_inference_steps", 30),
            default_width=panel_size.get("width", 768) if isinstance(panel_size, dict) else 768,
            default_height=panel_size.get("height", 512) if isinstance(panel_size, dict) else 512,
            seed=sd_cfg.get("seed"),
        )

    # ── LoRA auto-discovery ──────────────────────────────────────────
    @staticmethod
    def _discover_safetensors(directory: str) -> List[str]:
        """
        Find all ``.safetensors`` files in *directory*.

        This is the core of the "drop-in" design: users simply place their
        LoRA weights into the folder and the pipeline picks them up.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            logger.debug(f"LoRA directory {directory!r} does not exist (yet).")
            return []
        files = sorted(str(p) for p in dir_path.glob("*.safetensors"))
        if files:
            logger.info(f"Discovered {len(files)} LoRA file(s) in {directory}: "
                        f"{[Path(f).name for f in files]}")
        return files

    # ── Pipeline loading ─────────────────────────────────────────────
    def _load_pipeline(self):
        """Lazy-load the diffusion pipeline and attach LoRA adapters."""
        if self._pipe is not None:
            return

        if not _DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers / torch not installed. "
                "Run: pip install -r requirements.txt"
            )

        logger.info(f"Loading SD pipeline from {self.model_path!r} …")

        # Detect SDXL vs SD-1.5 by model path heuristic
        is_xl = "xl" in self.model_path.lower() or "sdxl" in self.model_path.lower()
        PipeClass = StableDiffusionXLPipeline if is_xl else StableDiffusionPipeline

        # Determine device & dtype
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

        self._pipe = PipeClass.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16" if dtype == torch.float16 else None,
        )

        # Use DPM++ 2M scheduler for speed
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config
        )

        # ── Load LoRA adapters ───────────────────────────────────────
        adapter_names = []
        adapter_weights = []

        for lora in self.lora_configs:
            adapter_name = lora.name.replace("/", "_")
            logger.info(f"Loading LoRA: {lora.path!r} "
                        f"(adapter={adapter_name!r}, weight={lora.weight})")
            self._pipe.load_lora_weights(
                lora.path,
                adapter_name=adapter_name,
            )
            adapter_names.append(adapter_name)
            adapter_weights.append(lora.weight)

        if adapter_names:
            self._pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            logger.info(f"Active LoRA adapters: {list(zip(adapter_names, adapter_weights))}")

        self._pipe.to(device)
        logger.info(f"Pipeline ready on {device} ({dtype})")

    # ── Image generation ─────────────────────────────────────────────
    def generate_panel(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """
        Generate a single panel image.

        Parameters
        ----------
        prompt : str
            Positive prompt (with LoRA tags already injected).
        negative_prompt : str
        width, height : int | None
            Override default panel size.
        seed : int | None
            Per-panel seed override.

        Returns
        -------
        PIL.Image.Image
        """
        self._load_pipeline()

        w = width or self.default_width
        h = height or self.default_height
        s = seed if seed is not None else self.seed

        generator = None
        if s is not None:
            device = self._pipe.device
            generator = torch.Generator(device=device).manual_seed(s)
            logger.info(f"Using seed {s}")

        logger.info(f"Generating image ({w}×{h}) …")
        logger.debug(f"  Prompt: {prompt[:200]}")

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=w,
            height=h,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        )

        image = result.images[0]
        logger.info("Panel generated successfully.")
        return image

    def generate_panels(
        self,
        panel_prompts: list,
        seed_offset: int = 0,
    ) -> List[Image.Image]:
        """
        Generate images for all panels.

        Parameters
        ----------
        panel_prompts : list[PanelPrompt]
            Objects with ``final_prompt`` and ``negative_prompt`` attributes.
        seed_offset : int
            Added to self.seed for per-panel variation.

        Returns
        -------
        list[PIL.Image.Image]
        """
        images: List[Image.Image] = []
        for i, pp in enumerate(panel_prompts):
            panel_seed = None
            if self.seed is not None:
                panel_seed = self.seed + seed_offset + i

            logger.info(f"── Panel {pp.panel_number} ({pp.camera_angle}) ──")
            img = self.generate_panel(
                prompt=pp.final_prompt,
                negative_prompt=pp.negative_prompt,
                seed=panel_seed,
            )
            images.append(img)

        return images
