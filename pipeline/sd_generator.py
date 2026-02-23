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

from PIL import Image, ImageDraw
from diffusers.image_processor import IPAdapterMaskProcessor

logger = logging.getLogger(__name__)

# ── Lazy imports for heavy ML libs (fail fast with clear message) ────
_DIFFUSERS_AVAILABLE = False
try:
    import torch
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionImg2ImgPipeline,
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
    ip_adapter : Dict[str, Any] | None
        Configuration for IP-Adapter (model_id, subfolder, weight_name, scale).
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
        ip_adapter: Optional[Dict[str, Any]] = None,
    ):
        self.model_path = model_path
        self.lora_configs = lora_configs or []
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.default_width = default_width
        self.default_height = default_height
        self.seed = seed
        self.ip_adapter = ip_adapter

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
            ip_adapter=sd_cfg.get("ip_adapter"),
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

        # Use DPM++ 2M Karras scheduler for better convergence
        # We explicitly set algorithm_type="dpmsolver++" to avoid conflict with "deis"
        # which can be present in some community models (like DreamShaper).
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
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

        # ── Load IP-Adapter ──────────────────────────────────────────
        if self.ip_adapter and self.ip_adapter.get("enable", False):
            ip_id = self.ip_adapter.get("model_id", "h94/IP-Adapter")
            sub_f = self.ip_adapter.get("subfolder", "sdxl_models")
            w_name = self.ip_adapter.get("weight_name", "ip-adapter_sdxl.bin")
            
            logger.info(f"Loading standard IP-Adapter weights...")
            
            if "vit-h" in w_name.lower():
                logger.info("Detected ViT-H IP-Adapter for SDXL. Explicitly loading ViT-H image encoder...")
                from transformers import CLIPVisionModelWithProjection
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    ip_id,
                    subfolder="models/image_encoder",
                    torch_dtype=dtype,
                ).to(device)
                self._pipe.image_encoder = image_encoder
            
            self._pipe.load_ip_adapter(
                ip_id, 
                subfolder=sub_f, 
                weight_name=w_name
            )
            
            # Disable by default to prevent unwanted injection
            self._pipe.set_ip_adapter_scale(0.0)
            logger.info("IP-Adapter loaded successfully (scale initialized to 0.0)")



        # Optimization for VRAM
        if device == "cuda":
            self._pipe.enable_model_cpu_offload()
            torch.cuda.empty_cache()
            logger.info("Pipeline VRAM optimization enabled: model_cpu_offload")
        else:
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
        ip_adapter_image: Optional[Image.Image] = None,
        layouts: Optional[List[Dict[str, Any]]] = None,
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
        
        # 1. Base Multiplier Fix (Must be multiple of 8)
        w = round(w / 8) * 8
        h = round(h / 8) * 8

        # 2. Auto-upscale for SDXL stability (Minimum ~1024 primary edge)
        # SDXL latent space degrades structurally at low resolutions (e.g., 768x512)
        if "xl" in self.model_path.lower():
            primary_edge = max(w, h)
            if primary_edge < 1024:
                logger.warning(f"Resolution {w}x{h} is too low for stable SDXL generation. Upscaling primary edge to 1024.")
                ratio = 1024 / primary_edge
                w = round((w * ratio) / 8) * 8
                h = round((h * ratio) / 8) * 8

        s = seed if seed is not None else self.seed
        generator = None
        if s is not None:
            generator = torch.Generator(device=self._pipe.device).manual_seed(s)

        logger.info(f"Generating image ({w}×{h}) …")

        logger.info("Generating via Text2Img")
        
        # Determine if IP-Adapter is active on the UNet
        has_ip = getattr(self._pipe, "unet", None) and getattr(self._pipe.unet, "encoder_hid_proj", None) is not None

        t2i_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "width": w,
            "height": h,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "generator": generator,
        }

        if has_ip and ip_adapter_image is not None:
            t2i_kwargs["ip_adapter_image"] = ip_adapter_image
            
            # Use the configured scale (default 0.7) for a 1-stage generation
            scale = self.ip_adapter.get("scale", 0.7) if self.ip_adapter else 0.7
            self._pipe.set_ip_adapter_scale(scale)
            logger.info(f"Text2Img with IP-Adapter (scale={scale})")
            
        final_image = self._pipe(**t2i_kwargs).images[0]
        logger.info("Panel generated successfully.")
        return final_image

    def generate_panels(
        self,
        panel_prompts: list,
        seed_offset: int = 0,
        ip_adapter_image: Optional[Image.Image] = None,
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
                ip_adapter_image=ip_adapter_image,
                layouts=getattr(pp, "layouts", []),
            )
            images.append(img)

        return images
