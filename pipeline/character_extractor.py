"""
Character Feature Extractor
============================
Extracts visual character features from a reference image using a
vision-capable LLM (multimodal model via Ollama), or accepts manual tags.

Output: comma-separated SD-compatible tags, e.g.
    "1boy, silver hair, round glasses, black trench coat"
"""

import base64
import logging
from pathlib import Path
from typing import Optional

from models.ollama_model import OllamaChatModel

logger = logging.getLogger(__name__)

# ── Prompt sent to the vision model ──────────────────────────────────
EXTRACTION_PROMPT = """\
You are an expert anime / manga character descriptor. 
Analyse the provided reference image and output a **comma-separated list** of 
Stable Diffusion tags that capture the character's visual features.

## Rules
1. Start with gender & count tag: `1boy`, `1girl`, `1person`, etc.
2. Cover: hair (colour, length, style), eyes (colour, shape), clothing, 
   accessories, body type, distinguishing marks.
3. Use **only** SD-standard tags (lowercase, comma-separated).
4. Do NOT include background, pose, or action descriptions.
5. Output ONLY the tag list — no explanations, no markdown.

## Example Output
1girl, long silver hair, ponytail, blue eyes, white techwear jacket, \
black leggings, combat boots, scar on left cheek
"""


class CharacterExtractor:
    """Extract character visual features from a reference image."""

    def __init__(self, vision_model_name: str = "gemma3:12b",
                 temperature: float = 0.3):
        self.llm = OllamaChatModel(
            model_name=vision_model_name,
            temperature=temperature,
        )
        self.llm.set_step_name("Character Extraction")

    # ── public API ───────────────────────────────────────────────────
    def extract(self, image_path: str) -> str:
        """
        Send the reference image to a vision LLM and return SD tags.

        Parameters
        ----------
        image_path : str
            Path to the character reference image (PNG / JPG / WEBP).

        Returns
        -------
        str
            Comma-separated SD tags describing the character.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Reference image not found: {image_path}")

        # Encode image to base64
        image_data = base64.b64encode(path.read_bytes()).decode("utf-8")
        mime = self._guess_mime(path)

        # Build multimodal message (OpenAI vision format)
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": EXTRACTION_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime};base64,{image_data}",
                    },
                },
            ],
        }

        logger.info("Sending reference image to vision LLM for feature extraction …")
        result = self.llm.invoke([message])
        tags = result.content.strip()
        logger.info(f"Extracted character tags: {tags}")
        return tags

    # ── helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _guess_mime(path: Path) -> str:
        suffix = path.suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }.get(suffix, "image/png")
