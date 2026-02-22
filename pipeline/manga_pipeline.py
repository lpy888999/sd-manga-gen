"""
Manga Pipeline — End-to-end orchestration
===========================================
Chains all steps:

1. Extract character features from reference image (vision LLM)
2. Expand user prompt into panel narratives (LLM Step 1)
3. Convert narratives to SD prompts (LLM Step 2)
4. Generate panel images (Stable Diffusion + LoRA)
5. Compose final comic layout (PIL)

Usage::

    from pipeline.manga_pipeline import MangaPipeline

    pipe = MangaPipeline.from_config("config.yaml")
    output = pipe.run(
        reference_image="ref.png",
        user_prompt="A samurai fighting a robot in the rain",
    )
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
from PIL import Image

from pipeline.story_expander import StoryExpander
from pipeline.prompt_engineer import PromptEngineer, PanelPrompt
from pipeline.sd_generator import SDGenerator, LoRAConfig
from pipeline.layout_composer import LayoutComposer
from pipeline.script_generator import ScriptGenerator
from pipeline.audio_engine import AudioEngine

logger = logging.getLogger(__name__)


class MangaPipeline:
    """
    End-to-end manga generation pipeline.

    Parameters
    ----------
    config : dict
        Parsed config.yaml contents.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        llm_cfg = config.get("llm", {})
        sd_cfg = config.get("sd", {})
        layout_cfg = config.get("layout", {})
        prompt_cfg = config.get("prompt", {})
        panel_cfg = config.get("panels", {})
        tts_cfg = config.get("tts", {})

        # ── Build sub-components ─────────────────────────────────────
        self.story_expander = StoryExpander(
            model_name=llm_cfg.get("model_name", "qwen3-coder-next:cloud"),
            temperature=llm_cfg.get("temperature", 0.7),
        )

        # Build LoRA tag strings for prompt injection
        lora_section = sd_cfg.get("lora", {})
        lora_tags = self._build_lora_tags(lora_section)
        trigger_words = lora_section.get("trigger_words", [])

        self.prompt_engineer = PromptEngineer(
            model_name=llm_cfg.get("model_name", "qwen3-coder-next:cloud"),
            temperature=0.4,
            quality_suffix=prompt_cfg.get(
                "quality_suffix",
                "masterpiece, best quality, high resolution, comic style, thick lineart",
            ),
            negative_prompt=prompt_cfg.get(
                "negative_prompt",
                "low quality, blurry, distorted face, extra fingers, bad anatomy",
            ),
            lora_tags=lora_tags,
            trigger_words=trigger_words,
        )

        self.sd_generator = SDGenerator.from_config(sd_cfg)

        self.layout_composer = LayoutComposer(
            border_width=layout_cfg.get("border_width", 6),
            gutter=layout_cfg.get("gutter", 12),
            background_color=layout_cfg.get("background_color", "white"),
        )

        # ── TTS modules (optional) ───────────────────────────────────
        self.tts_enabled = tts_cfg.get("enabled", False)
        if self.tts_enabled:
            # Script LLM can be separate from main LLM
            script_model = (
                tts_cfg.get("script_model_name")
                or llm_cfg.get("model_name", "qwen3-coder-next:cloud")
            )
            self.script_generator = ScriptGenerator(
                model_name=script_model,
                temperature=tts_cfg.get("script_temperature", 0.5),
            )
            self.audio_engine = AudioEngine.from_config(tts_cfg)
        else:
            self.script_generator = None
            self.audio_engine = None

        self.auto_threshold = panel_cfg.get("auto_threshold", 30)
        self.default_panel_count = panel_cfg.get("default_count", 4)
        self._log_fh = None  # track our dedicated FileHandler

    # ── Factory ──────────────────────────────────────────────────────
    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "MangaPipeline":
        """Load pipeline from a YAML config file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded config from {path}")
        return cls(config)

    # ── Main entry point ─────────────────────────────────────────────
    def run(
        self,
        reference_image: Optional[str] = None,
        user_prompt: str = "",
        panel_count: Optional[int] = None,
        output_path: str = "output/comic.png",
        seed: Optional[int] = None,
        enable_audio: Optional[bool] = None,
        audio_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full manga generation pipeline.

        Parameters
        ----------
        reference_image : str | None
            Path to character reference image for IP-Adapter consistency.
        user_prompt : str
            Story concept.
        panel_count : int | None
            4 or 6; None = auto-detect.
        output_path : str
            Where to save the final comic.
        seed : int | None
            Override seed from config.
        enable_audio : bool | None
            Override tts.enabled from config.  None = use config.
        audio_dir : str | None
            Override tts.output_dir from config for audio files.

        Returns
        -------
        dict
            ``{"comic_path": str, "audio": dict | None}``
        """
        t0 = time.time()
        do_audio = enable_audio if enable_audio is not None else self.tts_enabled

        # Guarantee all logs (pipeline.*, models.*, root) are written to a file
        # in the output directory.  diffusers/transformers model loading can
        # silently remove root-logger handlers, so we attach our own.
        self._ensure_file_logging(output_path)

        # ── Step 1: Story expansion ──────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1 · Story Expansion")
        logger.info("=" * 60)
        use_ref = reference_image is not None and Path(reference_image).exists()
        
        panels = self.story_expander.expand(
            user_prompt=user_prompt,
            panel_count=panel_count,
            auto_threshold=self.auto_threshold,
            use_reference=use_ref,
        )
        logger.info(f"Generated {len(panels)} panel descriptions:")
        for i, p in enumerate(panels):
            logger.info(f"  Panel {i+1}: {p}")

        # ── Step 2: SD prompt engineering ─────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 2 · Prompt Engineering")
        logger.info("=" * 60)
        panel_prompts = self.prompt_engineer.generate(
            panels=panels,
            use_reference=use_ref,
        )
        for pp in panel_prompts:
            logger.info(f"  Panel {pp.panel_number}: {pp.final_prompt}")

        # ── Step 2.5: Script extraction (TTS) ────────────────────────
        audio_result = None
        if do_audio and self.script_generator:
            logger.info("=" * 60)
            logger.info("STEP 2.5 · Script Extraction (TTS)")
            logger.info("=" * 60)
            scripts = self.script_generator.generate(panels)
            for s in scripts:
                for line in s.lines:
                    logger.info(f"  Panel {s.panel} [{line.role}]: {line.text}")

            # ── Step 2.6: Audio synthesis ─────────────────────────────
            if self.audio_engine:
                logger.info("=" * 60)
                logger.info("STEP 2.6 · Audio Synthesis (Coqui TTS)")
                logger.info("=" * 60)
                audio_result = self.audio_engine.synthesize(
                    scripts, output_dir=audio_dir
                )

        # ── Step 3: Image generation ─────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3 · Image Generation (SD + LoRA + IP-Adapter)")
        logger.info("=" * 60)
        if seed is not None:
            self.sd_generator.seed = seed

        ip_image = None
        face_embeds = None
        if reference_image and Path(reference_image).exists():
            logger.info(f"Loading reference image for IP-Adapter: {reference_image}")
            ip_image = Image.open(reference_image).convert("RGB")
            
            # --- InsightFace Extraction for FaceID Plus V2 ---
            ip_cfg = self.config.get("sd", {}).get("ip_adapter", {})
            if ip_cfg.get("enable", False) and ip_cfg.get("type") == "faceid_plus_v2":
                logger.info("FaceID Plus V2 enabled. Initializing InsightFace to extract features...")
                import cv2
                import numpy as np
                from insightface.app import FaceAnalysis
                
                # app initialized here to avoid heavy global import overhead if not used
                app = FaceAnalysis(name="antelopev2", root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                app.prepare(ctx_id=0, det_size=(640, 640))
                
                # InsightFace expects BGR cv2 image
                cv_img = cv2.cvtColor(np.array(ip_image), cv2.COLOR_RGB2BGR)
                faces = app.get(cv_img)
                
                if not faces:
                    logger.warning("No face detected in reference image by InsightFace! IP-Adapter may fail to inject character.")
                else:
                    logger.info(f"Detected {len(faces)} face(s). Using the most prominent one.")
                    # Sort by bounding box size to get the main subject
                    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                    import torch
                    face_embeds = torch.from_pretrained(faces[0].normed_embedding).unsqueeze(0)
                    
        images = self.sd_generator.generate_panels(
            panel_prompts,
            ip_adapter_image=ip_image,
            ip_adapter_face_embeds=face_embeds,
        )

        # ── Step 4: Layout composition ───────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 4 · Comic Layout Composition")
        logger.info("=" * 60)
        comic = self.layout_composer.compose(images, output_path=output_path)

        elapsed = time.time() - t0
        logger.info(f"✅ Pipeline complete in {elapsed:.1f}s — saved to {output_path}")

        return {
            "comic_path": output_path,
            "audio": audio_result,
        }

    # ── logging helpers ──────────────────────────────────────────────
    def _ensure_file_logging(self, output_path: str):
        """
        Attach a FileHandler to the *root* logger so that every logger
        (pipeline.*, models.*, main, etc.) writes to ``pipeline.log``
        in the output directory, regardless of what diffusers/transformers
        do to the logging configuration during model loading.
        """
        log_dir = Path(output_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "pipeline.log"

        fmt = logging.Formatter(
            "%(asctime)s │ %(name)-30s │ %(levelname)-5s │ %(message)s",
            datefmt="%H:%M:%S",
        )

        root = logging.getLogger()
        root.setLevel(logging.DEBUG)

        # ── Remove stale / duplicate file handlers ────────────────────
        resolved = log_file.resolve()
        for h in root.handlers[:]:
            if isinstance(h, logging.FileHandler):
                try:
                    if Path(h.baseFilename).resolve() == resolved:
                        root.removeHandler(h)
                        h.close()
                except Exception:
                    pass

        # ── Attach a fresh FileHandler ────────────────────────────────
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)
        self._log_fh = fh

        # ── Ensure stdout handler exists (for shell redirect capture) ─
        has_stream = any(
            isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
            for h in root.handlers
        )
        if not has_stream:
            sh = logging.StreamHandler(sys.stdout)
            sh.setLevel(logging.DEBUG)
            sh.setFormatter(fmt)
            root.addHandler(sh)

        # ── Silence noisy third-party loggers ─────────────────────────
        for lib in ("httpx", "httpcore", "urllib3", "diffusers",
                    "transformers", "accelerate"):
            logging.getLogger(lib).setLevel(logging.WARNING)

        logger.debug(f"Pipeline file logging attached → {log_file}")

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _build_lora_tags(lora_section: Dict[str, Any]) -> List[str]:
        """
        Build LoRA tag strings from config for prompt injection.

        These are the ``<lora:Name:Weight>`` strings appended to each
        SD prompt by the backend (not by the LLM).
        """
        tags: List[str] = []
        for adapter_name in ("character", "style"):
            adapter = lora_section.get(adapter_name, {})
            lora_dir = adapter.get("dir")
            weight = adapter.get("weight", 0.8)

            if not lora_dir:
                continue

            dir_path = Path(lora_dir)
            if not dir_path.is_dir():
                continue

            for sf in sorted(dir_path.glob("*.safetensors")):
                tag = f"<lora:{sf.stem}:{weight}>"
                tags.append(tag)

        return tags
