"""
Story Expander (LLM Step 1)
============================
Transforms a brief user concept into a structured, multi-panel narrative
with cinematic detail.

Output: a list of panel descriptions (plain text).
"""

import logging
import re
from typing import List, Optional

from models.ollama_model import OllamaChatModel

logger = logging.getLogger(__name__)

# ── System prompt — the "Narrative Architect" ────────────────────────
SYSTEM_PROMPT = """\
## Role
You are a professional Comic Scriptwriter. Your goal is to transform a brief \
user concept into a detailed, multi-panel narrative.

## Task
1. **Expand the Story**: Add sensory details (lighting, weather, textures) \
and emotional depth.
2. **Breakdown**: Structure the story into exactly {panel_count} sequential panels.
3. **Visual Focus**: Ensure each panel has one clear, dramatic action or focal point.

## Output Format
You MUST output exactly {panel_count} panels in this format — nothing else:

Panel 1: [Detailed visual description in 2-3 sentences]
Panel 2: [Detailed visual description in 2-3 sentences]
... (up to Panel {panel_count})

## Few-Shot Example
**User Input**: A samurai fighting a robot in the rain.
**Output**:
Panel 1: A lone samurai stands in a neon-lit alleyway during a heavy downpour. \
Rain splashes off his straw hat, and his hand grips the hilt of a katana.
Panel 2: A massive, rusted combat droid emerges from the shadows, its single red \
eye glowing through the mist and steam.
Panel 3: The samurai lunges forward, a flash of steel cutting through the raindrops. \
Sparks fly as the blade clangs against the robot's metallic armor.
Panel 4: The robot falls, its circuits sparking in a puddle. The samurai sheathes \
his sword, his silhouette reflecting in the wet pavement as the neon lights flicker.
"""


class StoryExpander:
    """Expand a short user idea into detailed panel-by-panel narrative."""

    def __init__(self, model_name: str = "qwen3-coder-next:cloud",
                 temperature: float = 0.7):
        self.llm = OllamaChatModel(
            model_name=model_name,
            temperature=temperature,
        )
        self.llm.set_step_name("Story Expansion")

    # ── public API ───────────────────────────────────────────────────
    def expand(
        self,
        user_prompt: str,
        panel_count: Optional[int] = None,
        auto_threshold: int = 30,
    ) -> List[str]:
        """
        Expand *user_prompt* into *panel_count* panel descriptions.

        Parameters
        ----------
        user_prompt : str
            Brief story concept from the user.
        panel_count : int | None
            4 or 6.  If None, auto-detect from word count.
        auto_threshold : int
            Word-count boundary for auto panel selection.

        Returns
        -------
        list[str]
            List of panel narrative descriptions.
        """
        if panel_count is None:
            word_count = len(user_prompt.split())
            panel_count = 4 if word_count <= auto_threshold else 6
            logger.info(f"Auto-detected panel count: {panel_count} "
                        f"(word count={word_count}, threshold={auto_threshold})")

        if panel_count not in (4, 6):
            raise ValueError(f"panel_count must be 4 or 6, got {panel_count}")

        system = SYSTEM_PROMPT.format(panel_count=panel_count)
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_prompt},
        ]

        logger.info(f"Expanding story into {panel_count} panels …")
        result = self.llm.invoke(messages)
        raw = result.content.strip()
        logger.info(f"Raw story expansion:\n{raw}")

        panels = self._parse_panels(raw, panel_count)
        return panels

    # ── parsing ──────────────────────────────────────────────────────
    @staticmethod
    def _parse_panels(text: str, expected: int) -> List[str]:
        """Parse 'Panel N: …' formatted text into a list of descriptions."""
        # Match lines like "Panel 1: …"
        pattern = re.compile(r"Panel\s*\d+\s*:\s*(.+?)(?=Panel\s*\d+\s*:|$)",
                             re.DOTALL | re.IGNORECASE)
        matches = pattern.findall(text)
        panels = [m.strip() for m in matches if m.strip()]

        if len(panels) != expected:
            logger.warning(
                f"Expected {expected} panels but parsed {len(panels)}. "
                f"Falling back to line-based split."
            )
            # Fallback: split by blank lines or numbered lines
            lines = [l.strip() for l in text.split("\n") if l.strip()]
            # Try to group into expected count
            if len(lines) >= expected:
                panels = lines[:expected]
            else:
                panels = lines

        return panels
