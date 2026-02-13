"""
Script Generator (LLM Step 2.5)
=================================
Converts verbose panel narratives from :class:`StoryExpander` into concise
dialogue / narration lines, with role and gender tags for TTS.

Output: ``List[PanelScript]`` — one entry per panel, each containing
structured ``ScriptLine`` items.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from models.ollama_model import OllamaChatModel

logger = logging.getLogger(__name__)

# ── System prompt — the "Script Extractor" ───────────────────────────
SYSTEM_PROMPT = """\
## Role
You are a professional manga script editor.  Your job is to convert detailed \
panel narratives into SHORT, punchy dialogue and narration lines suitable for \
speech synthesis (TTS) voice-over.

## Rules
1. Each line of text MUST be ≤ 20 words.  Shorter is better.
2. Every panel MUST have at least one line (narration or dialogue).
3. Distinguish **Narrator** (scene-setting, inner monologue) from named \
**Characters** (e.g. "Samurai", "Girl", "Robot").
4. Every character and narrator MUST have a `gender` field: "male" or "female".
5. The Narrator gender is "male" by default unless context suggests otherwise.
6. Keep dialogue dramatic, concise, and suitable for voice acting.

## Output
Return **ONLY** valid JSON — no markdown fences, no explanation.

```json
[
  {{
    "panel": 1,
    "lines": [
      {{"role": "Narrator", "text": "A rainy night in the city.", "gender": "male"}},
      {{"role": "Samurai", "text": "Come out, tin beast!", "gender": "male"}}
    ]
  }}
]
```

## Few-Shot Example

**Input panels**:
Panel 1: A lone samurai stands in a neon-lit alleyway during a heavy downpour.
Panel 2: A massive combat droid emerges, its red eye glowing.

**Output**:
[
  {{
    "panel": 1,
    "lines": [
      {{"role": "Narrator", "text": "Rain falls on neon-lit streets.", "gender": "male"}},
      {{"role": "Samurai", "text": "Tonight, we settle this.", "gender": "male"}}
    ]
  }},
  {{
    "panel": 2,
    "lines": [
      {{"role": "Narrator", "text": "A shadow stirs in the alley.", "gender": "male"}},
      {{"role": "Robot", "text": "Target acquired.", "gender": "male"}}
    ]
  }}
]
"""


# ── Data structures ──────────────────────────────────────────────────
@dataclass
class ScriptLine:
    """A single line of dialogue or narration."""
    role: str       # "Narrator" or character name
    text: str       # the spoken text (≤ 20 words)
    gender: str     # "male" or "female"


@dataclass
class PanelScript:
    """All dialogue lines for one panel."""
    panel: int
    lines: List[ScriptLine] = field(default_factory=list)


class ScriptGenerator:
    """
    Convert panel narratives → concise TTS-ready dialogue script.

    Parameters
    ----------
    model_name : str
        Ollama model to use (can differ from main pipeline LLM).
    temperature : float
    """

    def __init__(
        self,
        model_name: str = "qwen3-coder-next:cloud",
        temperature: float = 0.5,
    ):
        self.llm = OllamaChatModel(
            model_name=model_name,
            temperature=temperature,
        )
        self.llm.set_step_name("Script Generation")

    # ── public API ───────────────────────────────────────────────────
    def generate(self, panels: List[str]) -> List[PanelScript]:
        """
        Extract dialogue / narration from panel narratives.

        Parameters
        ----------
        panels : list[str]
            Panel descriptions from :class:`StoryExpander`.

        Returns
        -------
        list[PanelScript]
        """
        panel_text = "\n".join(
            f"Panel {i + 1}: {desc}" for i, desc in enumerate(panels)
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": panel_text},
        ]

        logger.info(f"Extracting script for {len(panels)} panels …")
        result = self.llm.invoke(messages)
        raw = result.content.strip()
        logger.info(f"Raw script JSON:\n{raw}")

        return self._parse_script(raw, len(panels))

    # ── parsing ──────────────────────────────────────────────────────
    @staticmethod
    def _parse_script(raw: str, expected_panels: int) -> List[PanelScript]:
        """Parse LLM JSON output into PanelScript objects."""
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse script JSON: {e}")
            logger.error(f"Raw content: {cleaned[:500]}")
            # Fallback: one narrator line per panel
            return [
                PanelScript(
                    panel=i + 1,
                    lines=[ScriptLine(role="Narrator", text=f"Panel {i + 1}.", gender="male")],
                )
                for i in range(expected_panels)
            ]

        scripts: List[PanelScript] = []
        for entry in data:
            panel_num = entry.get("panel", len(scripts) + 1)
            lines = []
            for line_data in entry.get("lines", []):
                lines.append(ScriptLine(
                    role=line_data.get("role", "Narrator"),
                    text=line_data.get("text", ""),
                    gender=line_data.get("gender", "male"),
                ))
            scripts.append(PanelScript(panel=panel_num, lines=lines))

        if len(scripts) != expected_panels:
            logger.warning(
                f"Expected {expected_panels} panel scripts but got {len(scripts)}"
            )

        return scripts
