"""
Prompt Engineer (LLM Step 2)
==============================
Converts narrative panel descriptions into optimized Stable Diffusion
tag-based prompts.  Appends LoRA syntax and quality tags in post-processing
(not by the LLM) to guarantee correct syntax and weights.

Output: list of ``PanelPrompt`` dataclass instances.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from models.ollama_model import OllamaChatModel

logger = logging.getLogger(__name__)

# ── Dataclass for a single panel prompt ──────────────────────────────
@dataclass
class PanelPrompt:
    panel_number: int
    camera_angle: str
    panel_design: str                    # LLM's spatial/layout reasoning
    sd_prompt: str                       # raw tags from LLM
    final_prompt: str = ""               # after LoRA / quality injection
    negative_prompt: str = ""
    layouts: List[Dict[str, Any]] = field(default_factory=list) # Bounding boxes for IP-Adapter masking


# ── System prompt — the "SD Engineer" ────────────────────────────────
SYSTEM_PROMPT = """\
## Role
You are an expert Prompt Engineer for Stable Diffusion. You specialize in \
converting natural language scenes into high-quality, tag-based prompts.

## Task
Convert the provided narrative panels into technical SD prompts. You must \
incorporate consistent character visual traits into every panel where the \
protagonist appears.

## Constraints
1. **Strict Brevity (77-Token Limit)**: SDXL's CLIP encoder only processes the first 77 tokens. You MUST extract ONLY the most essential visual keywords. Maximum 30 tags per prompt. Remove all filler words (e.g., "a", "the", "in", "with").
2. **Prioritized Weighting**: Apply higher weight to the most critical subjects and actions using SD syntax, e.g., `(1boy:1.2)`, `(swinging sword:1.3)`. Keep backgrounds and secondary elements unweighted to save space.
3. **Explicit Layout & Bounding Boxes**: Declare character positions using a `layout` array. For each character, provide a `label` (MUST use "protagonist" for the main character), a `box` with normalized bounding box coordinates `[x_min, y_min, x_max, y_max]` from 0.0 to 1.0 (e.g., `[0.1, 0.2, 0.4, 0.9]`), and a brief `prompt` for that specific character.
4. **NO Comic Formatting**: Do NOT include words like "comic", "manga", "speech bubble", "panel", "text", or "borders" in your `sd_prompt`, as this causes Structural image corruption.
5. **Tag Format**: Use comma-separated phrases. Order MUST be: [Weighted Subject], [Weighted Action], [Environment], [Lighting/Effect].
6. **Protagonist Appearance**: Do NOT describe the protagonist's physical appearance (e.g. hair color, clothing style, eye color, etc.). Their appearance is completely controlled by a reference image adapter. Just use generic tags like `1boy` or `1girl`, `protagonist`, and describe their action/emotion. You CAN describe the appearance of other secondary characters.
7. **Output**: Strictly valid JSON — no markdown fences, no commentary.

## Output JSON Format
{{
  "comic_output": [
    {{
      "panel_number": 1,
      "camera_angle": "Wide Shot / Close-up / etc.",
      "panel_design": "Explain the exact scene layout, number of characters, and character positioning.",
      "layout": [
        {{"label": "protagonist", "box": [0.1, 0.2, 0.4, 0.9], "prompt": "1man, protagonist, lunging forward"}},
        {{"label": "villain", "box": [0.6, 0.2, 0.9, 0.9], "prompt": "1man, wizard, black robe"}}
      ],
      "sd_prompt": "tags, go, here, masterpiece, high quality"
    }}
  ]
}}

## Few-Shot Example
**Input (Panel Description)**: The samurai lunges forward, a flash of steel \
cutting through the raindrops. Sparks fly as the blade clangs against the \
robot's metallic armor.
**Output**:
{{
  "comic_output": [
    {{
      "panel_number": 3,
      "camera_angle": "Action Shot",
      "panel_design": "2 characters. The protagonist is on the left lunging forward, the giant robot is on the right. Heavy rain environment.",
      "layout": [
        {{"label": "protagonist", "box": [0.05, 0.2, 0.45, 0.9], "prompt": "1man, protagonist, fighting stance"}},
        {{"label": "robot", "box": [0.55, 0.1, 0.95, 0.9], "prompt": "giant robot, metallic armor, blocking"}}
      ],
      "sd_prompt": "(1man:1.2), protagonist, (lunging forward:1.3), swinging katana, sparks flying, fighting giant robot, heavy rain, neon rim light"
    }}
  ]
}}
"""


class PromptEngineer:
    """Convert panel narratives into SD-optimised prompts with LoRA injection."""

    def __init__(
        self,
        model_name: str = "qwen3-coder-next:cloud",
        temperature: float = 0.4,
        quality_suffix: str = "masterpiece, comic_style",
        negative_prompt: str = "low quality, blurry, distorted, extra fingers",
        lora_tags: Optional[List[str]] = None,
        trigger_words: Optional[List[str]] = None,
    ):
        self.llm = OllamaChatModel(
            model_name=model_name,
            temperature=temperature,
        )
        self.llm.set_step_name("Prompt Engineering")

        self.quality_suffix = quality_suffix
        self.negative_prompt = negative_prompt
        self.lora_tags = lora_tags or []          # e.g. ["<lora:MyChar:0.8>"]
        self.trigger_words = trigger_words or []  # e.g. ["manga style"]

    # ── public API ───────────────────────────────────────────────────
    def generate(
        self,
        panels: List[str],
    ) -> List[PanelPrompt]:
        """
        Convert narrative panel descriptions into SD prompts.

        Parameters
        ----------
        panels : list[str]
            Panel narrative texts from StoryExpander.

        Returns
        -------
        list[PanelPrompt]
        """
        system = SYSTEM_PROMPT

        # Build the user message with all panels
        panel_text = "\n".join(
            f"Panel {i+1}: {desc}" for i, desc in enumerate(panels)
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": panel_text},
        ]

        logger.info("Converting panel narratives to SD prompts …")
        result = self.llm.invoke(messages)
        raw = result.content.strip()
        logger.debug(f"Raw SD prompt JSON:\n{raw}")

        # Parse JSON from LLM output
        parsed = self._parse_json(raw)
        panel_prompts = self._build_panel_prompts(parsed, len(panels))
        
        # Log textual reasoning cleanly
        for p in panel_prompts:
            logger.info(f"Panel {p.panel_number} Design: {p.panel_design}")
            
        return panel_prompts

    # ── JSON parsing ─────────────────────────────────────────────────
    @staticmethod
    def _parse_json(text: str) -> List[Dict[str, Any]]:
        """Extract the comic_output list from LLM JSON response."""
        # Strip wrapping markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", text)
        cleaned = cleaned.strip().rstrip("`")

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                logger.error(f"Failed to parse JSON from LLM output:\n{text}")
                raise ValueError("LLM did not return valid JSON for SD prompts.")

        if isinstance(data, dict) and "comic_output" in data:
            return data["comic_output"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected JSON structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")

    def _build_panel_prompts(
        self, raw_panels: List[Dict[str, Any]], expected_count: int
    ) -> List[PanelPrompt]:
        """Build PanelPrompt objects and inject LoRA / quality tags."""
        prompts: List[PanelPrompt] = []

        for item in raw_panels:
            sd_prompt = item.get("sd_prompt", "")

            # ── Backend prompt assembly ──────────────────────────────
            # Template: [Trigger Words], [Panel SD Prompt], [LoRA Tags], [Quality Suffix]
            parts: List[str] = []

            if self.trigger_words:
                parts.append(", ".join(self.trigger_words))

            parts.append(sd_prompt)

            if self.lora_tags:
                parts.append(", ".join(self.lora_tags))

            parts.append(self.quality_suffix)

            final_prompt = ", ".join(parts)

            prompts.append(PanelPrompt(
                panel_number=item.get("panel_number", len(prompts) + 1),
                camera_angle=item.get("camera_angle", "Medium Shot"),
                panel_design=item.get("panel_design", "No design provided."),
                sd_prompt=sd_prompt,
                final_prompt=final_prompt,
                negative_prompt=self.negative_prompt,
                layouts=item.get("layout", []),
            ))

        if len(prompts) != expected_count:
            logger.warning(
                f"Expected {expected_count} panel prompts but got {len(prompts)}"
            )

        return prompts
