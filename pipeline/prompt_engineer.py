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
    sd_prompt: str                       # raw tags from LLM
    final_prompt: str = ""               # after LoRA / quality injection
    negative_prompt: str = ""


# ── System prompt — the "SD Engineer" ────────────────────────────────
SYSTEM_PROMPT = """\
## Role
You are an expert Prompt Engineer for Stable Diffusion. You specialize in \
converting natural language scenes into high-quality, tag-based prompts.

## Task
Convert the provided narrative panels into technical SD prompts. You must \
incorporate the **Character Consistency Tags** provided below into every \
panel where the character appears.

## Constraints
1. **Tag Format**: Use comma-separated tags.  \
   Order: [Subject], [Action], [Environment], [Shot Type], [Lighting/Effect].
2. **Character Consistency**: You MUST include the Character Features in \
   every prompt where the character appears.
3. **Consistency**: Use "masterpiece, high quality, comic style" as global suffixes.
4. **Output**: Strictly valid JSON — no markdown fences, no commentary.

## Fixed Character Features (Reference)
{character_features}

## Output JSON Format
{{
  "comic_output": [
    {{
      "panel_number": 1,
      "camera_angle": "Wide Shot / Close-up / etc.",
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
      "sd_prompt": "1man, samurai, silver hair, black hakama, lunging forward, \
swinging katana, sword trail, sparks flying, fighting giant robot, heavy rain, \
splash, neon rim lighting, motion blur, cinematic composition, masterpiece, \
high quality, comic style"
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
        quality_suffix: str = "masterpiece, best quality, high resolution, comic style, thick lineart",
        negative_prompt: str = "low quality, blurry, distorted face, extra fingers, bad anatomy, watermark, text, signature",
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
        character_features: str,
    ) -> List[PanelPrompt]:
        """
        Convert narrative panel descriptions into SD prompts.

        Parameters
        ----------
        panels : list[str]
            Panel narrative texts from StoryExpander.
        character_features : str
            Comma-separated character tags from CharacterExtractor.

        Returns
        -------
        list[PanelPrompt]
        """
        system = SYSTEM_PROMPT.format(character_features=character_features)

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
        logger.info(f"Raw SD prompt JSON:\n{raw}")

        # Parse JSON from LLM output
        parsed = self._parse_json(raw)
        panel_prompts = self._build_panel_prompts(parsed, len(panels))
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
                sd_prompt=sd_prompt,
                final_prompt=final_prompt,
                negative_prompt=self.negative_prompt,
            ))

        if len(prompts) != expected_count:
            logger.warning(
                f"Expected {expected_count} panel prompts but got {len(prompts)}"
            )

        return prompts
