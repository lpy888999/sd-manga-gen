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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "panel_number": self.panel_number,
            "camera_angle": self.camera_angle,
            "panel_design": self.panel_design,
            "sd_prompt": self.sd_prompt,
            "final_prompt": self.final_prompt,
            "negative_prompt": self.negative_prompt,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PanelPrompt":
        return cls(
            panel_number=data.get("panel_number", 1),
            camera_angle=data.get("camera_angle", ""),
            panel_design=data.get("panel_design", ""),
            sd_prompt=data.get("sd_prompt", ""),
            final_prompt=data.get("final_prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
        )


# ── System prompt — the "SD Engineer" ────────────────────────────────
SYSTEM_PROMPT = """\
## Role
You are an expert Prompt Engineer for Stable Diffusion XL (SDXL). You specialize in \
converting narrative panels into highly effective, natural language prompts.

## Task
Convert the provided narrative panels into SDXL prompts. SDXL has excellent \
natural language understanding, so you should write cohesive, descriptive sentences \
rather than disconnected tags. Ensure consistent character visual traits across \
panels where the protagonist appears.

## Constraints
1. **Length Limit (77-Token Limit)**: SDXL's CLIP encoder prioritizing the first \
77 tokens. Keep your description to 1-2 concise, visually dense sentences. \
Remove non-essential narrative fluff, but keep it as natural English.
2. **Prioritized Weighting**: Apply higher weight to the most critical subjects \
and actions using SD syntax if necessary, e.g., `(swinging sword:1.2)`, but \
rely mostly on clear natural language.
3. **NO Comic Formatting**: Do NOT include words like "comic", "manga", "speech \
bubble", "panel", "text", or "borders" in your `sd_prompt`, as this causes structural \
image corruption.
4. **Character Appearance & Feature Decoupling**:
   {reference_instruction}
5. **Output**: Strictly valid JSON — no markdown fences, no commentary.

## Output JSON Format
{{
  "comic_output": [
    {{
      "panel_number": 1,
      "camera_angle": "Wide Shot / Close-up / Action Shot / etc.",
      "panel_design": "Explain the exact scene layout, number of characters, and visual focus.",
      "sd_prompt": "A natural language description of the scene goes here."
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
      "panel_design": "The protagonist and a giant robot in heavy rain. The focus is on the collision of the katana and the metallic armor.",
      "sd_prompt": "The protagonist lunges forward with a fighting stance, swinging a katana. Sparks are flying as the sword hits the giant robot's metallic armor in heavy rain, neon trim light."
    }}
  ]
}}
"""

# Instruction for Feature Decoupling when using IP-Adapter reference images
REFERENCE_INSTRUCTION_ACTIVE = """\
[IP-ADAPTER ACTIVE] A reference image is provided. To avoid 'Prompt Competition', \
you MUST NOT describe ANY facial anatomy, skin color, or specific clothing colors \
(e.g., 'sharp jawline', 'blue eyes', 'pale skin', 'red jacket'). Focus ONLY on:
- Global traits (e.g., '1man', 'protagonist')
- Action (e.g., 'lunging forward', 'running', 'fighting stance')
- Location/Environment (e.g., 'in a dark alley', 'heavy rain')
- Atmosphere and Expression (e.g., 'angry expression', 'cinematic lighting', 'action shot')\
"""

REFERENCE_INSTRUCTION_INACTIVE = """\
You MUST include strict visual descriptions of the characters as specified in \
the panel descriptions. Do not lose these distinctive visual traits, they are \
critical for maintaining character consistency across panels.\
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
        use_reference: bool = False,
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
        ref_instr = REFERENCE_INSTRUCTION_ACTIVE if use_reference else REFERENCE_INSTRUCTION_INACTIVE
        system = SYSTEM_PROMPT.format(reference_instruction=ref_instr)

        # Build the user message with all panels
        panel_text = "\n".join(
            f"Panel {i+1}: {desc}" for i, desc in enumerate(panels)
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": panel_text},
        ]

        if use_reference:
            logger.info("IP-Adapter mode active: Prompt engineering will decouple facial features.")

        logger.info("Converting panel narratives to SD prompts …")
        result = self.llm.invoke(messages)
        raw = result.content.strip()
        logger.info(f"Raw SD prompt JSON:\n{raw}")

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
            ))

        if len(prompts) != expected_count:
            logger.warning(
                f"Expected {expected_count} panel prompts but got {len(prompts)}"
            )

        return prompts
