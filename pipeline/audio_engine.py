"""
Audio Engine — Coqui TTS speech synthesis
==========================================
Generates per-panel ``.wav`` files from :class:`ScriptGenerator` output.

Voice Assignment
----------------
- ``Narrator`` → fixed speaker from config
- Named characters → automatically assigned from gender-matched voice pool
- Same character always gets the same voice across all panels

Supports two TTS backends:

1. **VCTK** (default): ``tts_models/en/vctk/vits`` — fast, multi-speaker
2. **XTTS**: ``tts_models/multilingual/multi-dataset/xtts_v2`` — multilingual

Usage::

    engine = AudioEngine.from_config(cfg["tts"])
    engine.synthesize(scripts, output_dir="output/audio")
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

from pipeline.script_generator import PanelScript, ScriptLine

logger = logging.getLogger(__name__)

# ── Lazy import for TTS (heavy dependency) ───────────────────────────
_TTS_AVAILABLE = False
try:
    from TTS.api import TTS as CoquiTTS
    _TTS_AVAILABLE = True
except ImportError:
    pass


class AudioEngine:
    """
    Generate speech audio from panel scripts.

    Parameters
    ----------
    model_name : str
        Coqui TTS model identifier.
    narrator_voice : str
        Speaker ID for the narrator.
    male_pool : list[str]
        Pool of male speaker IDs for character assignment.
    female_pool : list[str]
        Pool of female speaker IDs for character assignment.
    output_dir : str
        Directory for output ``.wav`` files.
    """

    def __init__(
        self,
        model_name: str = "tts_models/en/vctk/vits",
        narrator_voice: str = "p230",
        male_pool: Optional[List[str]] = None,
        female_pool: Optional[List[str]] = None,
        output_dir: str = "output/audio",
    ):
        self.model_name = model_name
        self.narrator_voice = narrator_voice
        self.male_pool = list(male_pool or ["p232", "p243", "p245", "p246"])
        self.female_pool = list(female_pool or ["p229", "p231", "p234", "p236"])
        self.output_dir = output_dir

        # Character → speaker_id cache (ensures consistency across panels)
        self._voice_map: Dict[str, str] = {}
        self._tts = None  # lazy-loaded

    # ── Factory from config ──────────────────────────────────────────
    @classmethod
    def from_config(cls, tts_cfg: Dict[str, Any]) -> "AudioEngine":
        """Build from the ``tts:`` section of config.yaml."""
        voices = tts_cfg.get("voices", {})
        return cls(
            model_name=tts_cfg.get("model", "tts_models/en/vctk/vits"),
            narrator_voice=voices.get("narrator", "p230"),
            male_pool=voices.get("male_pool", ["p232", "p243", "p245", "p246"]),
            female_pool=voices.get("female_pool", ["p229", "p231", "p234", "p236"]),
            output_dir=tts_cfg.get("output_dir", "output/audio"),
        )

    # ── TTS model loading ────────────────────────────────────────────
    def _load_tts(self):
        """Lazy-load the Coqui TTS model."""
        if self._tts is not None:
            return

        if not _TTS_AVAILABLE:
            raise ImportError(
                "Coqui TTS not installed. Run: pip install TTS>=0.22.0"
            )

        logger.info(f"Loading TTS model: {self.model_name} …")
        self._tts = CoquiTTS(self.model_name)
        logger.info("TTS model loaded.")

    # ── Voice assignment ─────────────────────────────────────────────
    def _get_voice(self, role: str, gender: str) -> str:
        """
        Get a consistent speaker ID for a role.

        - Narrator → always the configured narrator voice.
        - Known character → return cached voice.
        - New character → pick from the gender-appropriate pool.
        """
        if role.lower() == "narrator":
            return self.narrator_voice

        # Return cached voice if already assigned
        if role in self._voice_map:
            return self._voice_map[role]

        # Pick from pool
        pool = self.male_pool if gender == "male" else self.female_pool
        if pool:
            voice = random.choice(pool)
        else:
            # Pool exhausted — fall back to narrator voice
            logger.warning(
                f"No voices left in {gender} pool for '{role}', "
                f"falling back to narrator voice."
            )
            voice = self.narrator_voice

        self._voice_map[role] = voice
        logger.info(f"Assigned voice '{voice}' to character '{role}' ({gender})")
        return voice

    # ── Main synthesis ───────────────────────────────────────────────
    def synthesize(
        self,
        scripts: List[PanelScript],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate ``.wav`` files for all script lines.

        Parameters
        ----------
        scripts : list[PanelScript]
            Output from :class:`ScriptGenerator`.
        output_dir : str | None
            Override output directory.

        Returns
        -------
        dict
            ``{"audio_dir": str, "files": list[str], "script": list[dict]}``
        """
        self._load_tts()

        out = Path(output_dir or self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_files: List[str] = []
        script_json: List[dict] = []

        for ps in scripts:
            panel_entry = {"panel": ps.panel, "lines": []}

            for j, line in enumerate(ps.lines):
                voice = self._get_voice(line.role, line.gender)
                filename = f"panel_{ps.panel:02d}_line_{j + 1:02d}.wav"
                filepath = out / filename

                logger.info(
                    f"TTS: Panel {ps.panel}, Line {j + 1} "
                    f"[{line.role}/{voice}]: \"{line.text}\""
                )

                try:
                    self._tts.tts_to_file(
                        text=line.text,
                        speaker=voice,
                        file_path=str(filepath),
                    )
                    all_files.append(str(filepath))
                except Exception as e:
                    logger.error(f"TTS failed for '{line.text}': {e}")
                    # Continue with remaining lines
                    continue

                panel_entry["lines"].append({
                    "role": line.role,
                    "text": line.text,
                    "gender": line.gender,
                    "voice": voice,
                    "file": filename,
                })

            script_json.append(panel_entry)

        # Write full script JSON
        script_path = out / "full_script.json"
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script_json, f, ensure_ascii=False, indent=2)
        logger.info(f"Script saved to {script_path}")

        result = {
            "audio_dir": str(out),
            "files": all_files,
            "script": script_json,
        }

        logger.info(
            f"✅ Generated {len(all_files)} audio files in {out}"
        )
        return result

    # ── Utility ──────────────────────────────────────────────────────
    def get_voice_map(self) -> Dict[str, str]:
        """Return the current character → voice mapping."""
        return dict(self._voice_map)
