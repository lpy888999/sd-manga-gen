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
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
from PIL import Image

from pipeline.character_extractor import CharacterExtractor
from pipeline.story_expander import StoryExpander
from pipeline.prompt_engineer import PromptEngineer, PanelPrompt
from pipeline.sd_generator import SDGenerator, LoRAConfig
from pipeline.layout_composer import LayoutComposer

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

        # ── Build sub-components ─────────────────────────────────────
        self.extractor = CharacterExtractor(
            vision_model_name=llm_cfg.get("vision_model_name", "gemma3:12b"),
            temperature=llm_cfg.get("temperature", 0.3),
        )

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

        self.auto_threshold = panel_cfg.get("auto_threshold", 30)
        self.default_panel_count = panel_cfg.get("default_count", 4)

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
        character_tags: Optional[str] = None,
        user_prompt: str = "",
        panel_count: Optional[int] = None,
        output_path: str = "output/comic.png",
        seed: Optional[int] = None,
    ) -> str:
        """
        Run the full manga generation pipeline.

        Parameters
        ----------
        reference_image : str | None
            Path to character reference image.
        character_tags : str | None
            Manual character tags (skips vision extraction).
        user_prompt : str
            Story concept.
        panel_count : int | None
            4 or 6; None = auto-detect.
        output_path : str
            Where to save the final comic.
        seed : int | None
            Override seed from config.

        Returns
        -------
        str
            Path to the saved comic image.
        """
        t0 = time.time()

        # ── Step 0: Character feature extraction ─────────────────────
        if character_tags:
            logger.info(f"Using manual character tags: {character_tags}")
            features = character_tags
        elif reference_image:
            logger.info("=" * 60)
            logger.info("STEP 0 · Character Feature Extraction")
            logger.info("=" * 60)
            features = self.extractor.extract(reference_image)
        else:
            features = ""
            logger.warning("No reference image or character tags provided. "
                           "Character consistency will be limited.")

        # ── Step 1: Story expansion ──────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 1 · Story Expansion")
        logger.info("=" * 60)
        panels = self.story_expander.expand(
            user_prompt=user_prompt,
            panel_count=panel_count,
            auto_threshold=self.auto_threshold,
        )
        logger.info(f"Generated {len(panels)} panel descriptions.")

        # ── Step 2: SD prompt engineering ─────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 2 · Prompt Engineering")
        logger.info("=" * 60)
        panel_prompts = self.prompt_engineer.generate(
            panels=panels,
            character_features=features,
        )
        for pp in panel_prompts:
            logger.info(f"  Panel {pp.panel_number}: {pp.final_prompt[:120]}…")

        # ── Step 3: Image generation ─────────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 3 · Image Generation (SD + LoRA)")
        logger.info("=" * 60)
        if seed is not None:
            self.sd_generator.seed = seed

        images = self.sd_generator.generate_panels(panel_prompts)

        # ── Step 4: Layout composition ───────────────────────────────
        logger.info("=" * 60)
        logger.info("STEP 4 · Comic Layout Composition")
        logger.info("=" * 60)
        comic = self.layout_composer.compose(images, output_path=output_path)

        elapsed = time.time() - t0
        logger.info(f"✅ Pipeline complete in {elapsed:.1f}s — saved to {output_path}")

        return output_path

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
