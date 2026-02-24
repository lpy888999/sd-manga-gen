"""
SD Generator — Stable Diffusion image generation with LoRA + Two-Stage support
===============================================================================
Uses HuggingFace ``diffusers`` to load an SDXL (or SD-1.5) checkpoint
and generate panel images.

Generation Modes
----------------
The pipeline selects a mode automatically based on configuration:

* **LoRA only** (no IP-Adapter or no reference image)
  → Single-stage Text2Img.  LoRA drives style and character anatomy.

* **IP-Adapter only** (no LoRAs loaded)
  → Single-stage Text2Img with IP-Adapter at configured scale.

* **LoRA + IP-Adapter** (both enabled, ``two_stage.enabled: true``)
  → **Two-stage** generation:
  1. Stage 1 — pure LoRA Text2Img (no IP influence) for clean composition.
  2. Stage 2 — ControlNet Canny Img2Img + low-scale IP-Adapter to nudge
     character identity toward the reference without breaking layout/pose.

  Set ``ip_adapter.two_stage.enabled: false`` to revert to old single-stage
  behaviour even when both are active.

LoRA Integration
----------------
**Drop-in design**: place ``.safetensors`` files into the configured folders
(``loras/character/`` and ``loras/style/`` by default) and they will be
loaded automatically — no code changes required.

Usage::

    gen = SDGenerator.from_config(cfg["sd"])
    images = gen.generate_panels(panel_prompts, ip_adapter_image=ref_img)
"""

import logging
import glob
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)

# ── Lazy imports for heavy ML libs (fail fast with clear message) ────
_DIFFUSERS_AVAILABLE = False
_CONTROLNET_AVAILABLE = False

try:
    import torch
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionXLControlNetImg2ImgPipeline,
        ControlNetModel,
        DPMSolverMultistepScheduler,
    )
    _DIFFUSERS_AVAILABLE = True
    _CONTROLNET_AVAILABLE = True
except ImportError:
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

# Optional: cv2 for Canny; we fall back to PIL if not available
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


@dataclass
class LoRAConfig:
    """Configuration for a single LoRA adapter."""
    name: str           # human-readable label ("character", "style", …)
    path: str           # path to .safetensors file
    weight: float = 0.8 # adapter strength


class SDGenerator:
    """
    Generate images using Stable Diffusion + LoRA, with optional two-stage
    ControlNet Canny + IP-Adapter refinement.

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
        Configuration for IP-Adapter (model_id, subfolder, weight_name,
        scale, two_stage sub-dict).
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

        # Extract two-stage config once for convenience
        self._two_stage_cfg: Dict[str, Any] = {}
        self._face_restoration_cfg: Dict[str, Any] = {}
        if ip_adapter and ip_adapter.get("enable", False):
            self._two_stage_cfg = ip_adapter.get("two_stage", {})
            self._face_restoration_cfg = ip_adapter.get("face_restoration", {})

        self._pipe = None        # lazy-loaded primary t2i pipeline
        self._cn_pipe = None     # lazy-loaded ControlNet i2i pipeline
        self._restorer = None    # lazy-loaded FaceRestorer

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

    # ── Canny edge helper ────────────────────────────────────────────
    @staticmethod
    def _make_canny(
        image: Image.Image,
        low_threshold: int = 50,
        high_threshold: int = 150,
    ) -> Image.Image:
        """
        Extract Canny edges from *image* and return an RGB PIL image.

        Uses cv2 if available, falls back to a Laplacian-approximation
        via PIL's edge-detection filter otherwise.
        """
        if _CV2_AVAILABLE:
            gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, low_threshold, high_threshold)
            # Replicate single channel to RGB for ControlNet
            canny_rgb = np.stack([edges, edges, edges], axis=-1)
            return Image.fromarray(canny_rgb)
        else:
            logger.warning(
                "cv2 not available — falling back to PIL FIND_EDGES for Canny. "
                "Install opencv-python for better edge quality."
            )
            gray = image.convert("L")
            edges = gray.filter(ImageFilter.FIND_EDGES)
            return edges.convert("RGB")

    # ── Mode resolution ──────────────────────────────────────────────
    def _resolve_mode(self, ip_adapter_image: Optional[Image.Image]) -> str:
        """
        Determine the generation mode based on active configuration.

        Returns one of:
          "two_stage"  — LoRA + IP-Adapter + two_stage.enabled + ref image present
          "ip_only"    — IP-Adapter active, no LoRAs (or two_stage disabled)
          "lora_only"  — LoRAs active, IP-Adapter disabled or no ref image
          "plain"      — neither LoRA nor IP-Adapter active
        """
        has_loras = bool(self.lora_configs)
        ip_cfg = self.ip_adapter or {}
        has_ip_enabled = ip_cfg.get("enable", False)
        has_ref = ip_adapter_image is not None
        has_ip = has_ip_enabled and has_ref
        ts_enabled = self._two_stage_cfg.get("enabled", False)

        if has_loras and has_ip and ts_enabled and _CONTROLNET_AVAILABLE:
            return "two_stage"
        elif has_ip:
            return "ip_only"
        elif has_loras:
            return "lora_only"
        else:
            return "plain"

    # ── Pipeline loading ─────────────────────────────────────────────
    def _load_pipeline(self) -> Tuple[str, Any]:
        """
        Lazy-load the primary diffusion pipeline and attach LoRA adapters.

        Returns (device, dtype) for downstream use.
        """
        if self._pipe is not None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
            return device, dtype

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

            logger.info("Loading standard IP-Adapter weights...")

            if "vit-h" in w_name.lower():
                logger.info("Detected ViT-H IP-Adapter — loading matching ViT-H image encoder.")
                from transformers import CLIPVisionModelWithProjection
                image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                    ip_id,
                    subfolder="models/image_encoder",
                    torch_dtype=dtype,
                ).to(device)
                self._pipe.image_encoder = image_encoder

            self._pipe.load_ip_adapter(ip_id, subfolder=sub_f, weight_name=w_name)

            # Disable by default — activated per-call in generate_panel
            self._pipe.set_ip_adapter_scale(0.0)
            logger.info("IP-Adapter loaded (scale initialised to 0.0)")

        # ── VRAM optimisation ─────────────────────────────────────────
        if device == "cuda":
            self._pipe.enable_model_cpu_offload()
            torch.cuda.empty_cache()
            logger.info("Pipeline VRAM optimisation enabled: model_cpu_offload")
        else:
            self._pipe.to(device)

        logger.info(f"Primary pipeline ready on {device} ({dtype})")
        return device, dtype

    # ── ControlNet pipeline (lazy, reuses primary weights) ──────────
    def _load_controlnet_pipeline(self, device: str, dtype: Any):
        """
        Lazy-load the ControlNet Canny + IP-Adapter Img2Img pipeline.

        Reuses UNet, VAE, tokenizers, and text encoders from ``self._pipe``
        to avoid doubling VRAM usage.
        """
        if self._cn_pipe is not None:
            return

        if not _CONTROLNET_AVAILABLE:
            raise RuntimeError(
                "StableDiffusionXLControlNetImg2ImgPipeline not available. "
                "Please upgrade diffusers: pip install -U diffusers"
            )

        cn_id = self._two_stage_cfg.get(
            "controlnet_id", "diffusers/controlnet-canny-sdxl-1.0"
        )
        logger.info(f"Loading ControlNet model: {cn_id!r} …")
        controlnet = ControlNetModel.from_pretrained(cn_id, torch_dtype=dtype)

        # Build ControlNet i2i pipeline from the already-loaded primary pipe
        # components to share weights and save VRAM.
        logger.info("Building ControlNet Img2Img pipeline (shared weights) …")
        self._cn_pipe = StableDiffusionXLControlNetImg2ImgPipeline(
            vae=self._pipe.vae,
            text_encoder=self._pipe.text_encoder,
            text_encoder_2=self._pipe.text_encoder_2,
            tokenizer=self._pipe.tokenizer,
            tokenizer_2=self._pipe.tokenizer_2,
            unet=self._pipe.unet,
            scheduler=self._pipe.scheduler,
            controlnet=controlnet,
            image_encoder=getattr(self._pipe, "image_encoder", None),
            feature_extractor=getattr(self._pipe, "feature_extractor", None),
        )

        if device == "cuda":
            self._cn_pipe.enable_model_cpu_offload()
            torch.cuda.empty_cache()
        else:
            self._cn_pipe.to(device)

        logger.info("ControlNet Img2Img pipeline ready.")

    # ── Image generation ─────────────────────────────────────────────
    def generate_panel(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        seed: Optional[int] = None,
        ip_adapter_image: Optional[Image.Image] = None,
    ) -> Image.Image:
        """
        Generate a single panel image.

        Mode is chosen automatically (see module docstring):
        - ``two_stage``  : LoRA t2i → Canny ControlNet i2i + low IP-Adapter
        - ``ip_only``    : t2i with IP-Adapter at configured scale
        - ``lora_only``  : t2i with LoRA only
        - ``plain``      : plain t2i

        Parameters
        ----------
        prompt, negative_prompt : str
        width, height : int | None  — override default panel size
        seed : int | None           — per-panel seed override
        ip_adapter_image : PIL.Image | None — character reference image
        """
        device, dtype = self._load_pipeline()

        # ── Resolve resolution ───────────────────────────────────────
        w = width or self.default_width
        h = height or self.default_height

        w = round(w / 8) * 8
        h = round(h / 8) * 8

        # Auto-upscale for SDXL stability (primary edge ≥ 1024)
        if "xl" in self.model_path.lower():
            primary_edge = max(w, h)
            if primary_edge < 1024:
                logger.warning(
                    f"Resolution {w}×{h} is below SDXL minimum. "
                    f"Upscaling primary edge to 1024."
                )
                ratio = 1024 / primary_edge
                w = round((w * ratio) / 8) * 8
                h = round((h * ratio) / 8) * 8

        # ── Resolve seed / generator ────────────────────────────────
        s = seed if seed is not None else self.seed
        generator = None
        if s is not None:
            generator = torch.Generator(device=self._pipe.device).manual_seed(s)

        logger.info(f"Generating panel ({w}×{h}) …")

        # ── Determine mode ───────────────────────────────────────────
        mode = self._resolve_mode(ip_adapter_image)
        logger.info(f"Generation mode: {mode}")

        # ── Shared t2i kwargs ────────────────────────────────────────
        base_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            width=w,
            height=h,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            generator=generator,
        )

        # ────────────────────────────────────────────────────────────
        #  Mode: TWO-STAGE
        #  Stage 1 → LoRA Text2Img (clean composition, no IP influence)
        #  Stage 2 → ControlNet Canny Img2Img + low IP-Adapter scale
        # ────────────────────────────────────────────────────────────
        if mode == "two_stage":
            ts = self._two_stage_cfg

            # ── STAGE 1: pure LoRA t2i ──────────────────────────────
            logger.info("Stage 1 — LoRA Text2Img (composition pass, IP scale=0.0)")
            self._pipe.set_ip_adapter_scale(0.0)
            
            # Pass ip_adapter_image even if scale=0 to satisfy UNet conditioning
            stage1_kwargs = base_kwargs.copy()
            if getattr(self._pipe, "unet", None) and getattr(self._pipe.unet, "encoder_hid_proj", None) is not None:
                stage1_kwargs["ip_adapter_image"] = ip_adapter_image
                
            stage1_img = self._pipe(**stage1_kwargs).images[0]
            logger.info("Stage 1 complete.")

            # ── Extract Canny edges ─────────────────────────────────
            canny_low  = ts.get("canny_low_threshold",  50)
            canny_high = ts.get("canny_high_threshold", 150)
            canny_img = self._make_canny(stage1_img, canny_low, canny_high)
            logger.info(f"Canny edges extracted (low={canny_low}, high={canny_high}).")

            # ── STAGE 2: ControlNet Canny i2i + low IP-Adapter ─────
            self._load_controlnet_pipeline(device, dtype)

            strength       = ts.get("stage2_strength",  0.45)
            ip_scale_s2    = ts.get("stage2_ip_scale",  0.35)
            cn_scale       = ts.get("controlnet_scale", 0.6)

            logger.info(
                f"Stage 2 — ControlNet Canny Img2Img "
                f"(strength={strength}, ip_scale={ip_scale_s2}, cn_scale={cn_scale})"
            )
            self._cn_pipe.set_ip_adapter_scale(ip_scale_s2)

            # Stage 2 generator: increment seed by 1 to add slight variation
            s2_generator = None
            if s is not None:
                s2_generator = torch.Generator(
                    device=self._cn_pipe.device
                ).manual_seed(s + 1)

            final_image = self._cn_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                image=stage1_img,
                control_image=canny_img,
                ip_adapter_image=ip_adapter_image,
                strength=strength,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                controlnet_conditioning_scale=cn_scale,
                generator=s2_generator,
            ).images[0]

            logger.info("Stage 2 complete. Panel generated (two-stage).")
            
            # STAGE 3: Face Restoration
            if self._face_restoration_cfg.get("enabled", False):
                if self._restorer is None:
                    from .utils.face_restorer import FaceRestorer
                    self._restorer = FaceRestorer(self._face_restoration_cfg)
                final_image = self._restorer.restore(
                    final_image, self._pipe, prompt, negative_prompt, ip_adapter_image=ip_adapter_image
                )
                
            return final_image

        # ────────────────────────────────────────────────────────────
        #  Mode: IP_ONLY  — single-stage with IP-Adapter at full scale
        # ────────────────────────────────────────────────────────────
        elif mode == "ip_only":
            scale = (self.ip_adapter or {}).get("scale", 0.6)
            self._pipe.set_ip_adapter_scale(scale)
            base_kwargs["ip_adapter_image"] = ip_adapter_image
            logger.info(f"Single-stage Text2Img with IP-Adapter (scale={scale})")
            final_image = self._pipe(**base_kwargs).images[0]
            logger.info(f"Panel generated ({mode}).")
            
            # STAGE 3: Face Restoration
            if self._face_restoration_cfg.get("enabled", False):
                if self._restorer is None:
                    from .utils.face_restorer import FaceRestorer
                    self._restorer = FaceRestorer(self._face_restoration_cfg)
                final_image = self._restorer.restore(
                    final_image, self._pipe, prompt, negative_prompt, ip_adapter_image=ip_adapter_image
                )
                
            return final_image

        # ────────────────────────────────────────────────────────────
        #  Mode: LORA_ONLY or PLAIN — straightforward Text2Img
        # ────────────────────────────────────────────────────────────
        else:
            # Ensure IP-Adapter is inactive if it happens to be loaded
            if self.ip_adapter and self.ip_adapter.get("enable", False):
                self._pipe.set_ip_adapter_scale(0.0)
            logger.info(f"Single-stage Text2Img ({mode})")
            final_image = self._pipe(**base_kwargs).images[0]
            logger.info(f"Panel generated ({mode}).")
            
            # STAGE 3: Face Restoration
            if self._face_restoration_cfg.get("enabled", False):
                if self._restorer is None:
                    from .utils.face_restorer import FaceRestorer
                    self._restorer = FaceRestorer(self._face_restoration_cfg)
                final_image = self._restorer.restore(
                    final_image, self._pipe, prompt, negative_prompt
                )
                
            return final_image

    # ── Batch generation ─────────────────────────────────────────────
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
        ip_adapter_image : PIL.Image | None
            Character reference image forwarded to every panel.

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
            )
            images.append(img)

        return images
