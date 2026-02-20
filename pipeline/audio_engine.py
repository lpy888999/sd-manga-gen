"""
Audio Engine — Edge TTS speech synthesis
==========================================
Generates per-panel ``.wav`` files from :class:`ScriptGenerator` output using Microsoft Edge TTS API.

Voice Assignment
----------------
- ``Narrator`` → fixed speaker from config
- Named characters → automatically assigned from gender-matched voice pool
- Same character always gets the same voice across all panels

Usage::

    engine = AudioEngine.from_config(cfg["tts"])
    engine.synthesize(scripts, output_dir="output/audio")
"""

import json
import logging
import random
import re
import wave
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any

import edge_tts
from pydub import AudioSegment

from pipeline.script_generator import PanelScript, ScriptLine

logger = logging.getLogger(__name__)

class AudioEngine:
    """
    Generate speech audio from panel scripts using edge-tts.

    Parameters
    ----------
    model_name : str
        TTS engine marker (e.g., "edge-tts").
    narrator_voice : str
        Speaker ID for the narrator (e.g., "en-US-ChristopherNeural").
    male_pool : list[str]
        Pool of male speaker IDs for character assignment.
    female_pool : list[str]
        Pool of female speaker IDs for character assignment.
    output_dir : str
        Directory for output files.
    """

    def __init__(
        self,
        model_name: str = "edge-tts",
        narrator_voice: str = "en-US-ChristopherNeural",
        male_pool: Optional[List[str]] = None,
        female_pool: Optional[List[str]] = None,
        output_dir: str = "output/audio",
    ):
        self.model_name = model_name
        self.narrator_voice = narrator_voice
        self.male_pool = list(male_pool or ["en-US-GuyNeural", "en-US-EricNeural", "en-GB-RyanNeural"])
        self.female_pool = list(female_pool or ["en-US-AriaNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"])
        self.output_dir = output_dir

        self._voice_map: Dict[str, str] = {}

    @classmethod
    def from_config(cls, tts_cfg: Dict[str, Any]) -> "AudioEngine":
        voices = tts_cfg.get("voices", {})
        return cls(
            model_name=tts_cfg.get("model", "edge-tts"),
            narrator_voice=voices.get("narrator", "en-US-ChristopherNeural"),
            male_pool=voices.get("male_pool", ["en-US-GuyNeural", "en-US-EricNeural", "en-GB-RyanNeural"]),
            female_pool=voices.get("female_pool", ["en-US-AriaNeural", "en-US-JennyNeural", "en-GB-SoniaNeural"]),
            output_dir=tts_cfg.get("output_dir", "output/audio"),
        )

    def _get_voice(self, role: str, gender: str) -> str:
        role_norm = str(role).strip().lower()
        if role_norm == "narrator":
            return self.narrator_voice

        if role in self._voice_map:
            return self._voice_map[role]

        gender_norm = str(gender).strip().lower()
        pool = self.male_pool if gender_norm == "male" else self.female_pool
        
        if pool:
            voice = random.choice(pool)
        else:
            logger.warning(f"No voices left in {gender} pool for '{role}', falling back to narrator.")
            voice = self.narrator_voice

        self._voice_map[role] = voice
        logger.info(f"Assigned voice '{voice}' to character '{role}' ({gender})")
        return voice

    async _synthesize_async(self, text: str, voice: str, filepath: Path):
        communicate = edge_tts.Communicate(text, voice)
        # Edge-TTS natively saves as mp3, we save as tmp.mp3 then convert to wav
        tmp_mp3 = filepath.with_suffix(".mp3")
        await communicate.save(str(tmp_mp3))
        
        # Convert mp3 to wav (24kHz, mono) for standard concatenation processing later
        audio = AudioSegment.from_mp3(str(tmp_mp3))
        audio = audio.set_frame_rate(24000).set_channels(1)
        audio.export(str(filepath), format="wav")
        # Clean up mp3
        tmp_mp3.unlink(missing_ok=True)

    def synthesize(
        self,
        scripts: List[PanelScript],
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate ``.wav`` files for all script lines using asyncio edge-tts.
        """
        out = Path(output_dir or self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        all_files: List[str] = []
        script_json: List[dict] = []

        # Run the async loop to process lines synchronously in order
        async def run_all():
            for ps in scripts:
                panel_entry = {"panel": ps.panel, "lines": []}
                for j, line in enumerate(ps.lines):
                    voice = self._get_voice(line.role, line.gender)
                    filename = f"panel_{ps.panel:02d}_line_{j + 1:02d}.wav"
                    filepath = out / filename

                    text = self._clean_text(line.text)
                    if len(text) < 3:
                        logger.warning(f"TTS: Skipping Panel {ps.panel} Line {j+1} — text too short: {text!r}")
                        continue

                    logger.info(f"TTS: Panel {ps.panel}, Line {j + 1} [{line.role}/{voice}]: \"{text}\"")
                    
                    try:
                        await self._synthesize_async(text, voice, filepath)
                        all_files.append(str(filepath))
                    except Exception as e:
                        logger.error(f"TTS failed for '{text}': {e}")
                        continue

                    panel_entry["lines"].append({
                        "role": line.role,
                        "text": line.text,
                        "gender": line.gender,
                        "voice": voice,
                        "file": filename,
                    })
                script_json.append(panel_entry)

        # Execute async TTS loop
        asyncio.run(run_all())

        script_path = out / "full_script.json"
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script_json, f, ensure_ascii=False, indent=2)
        logger.info(f"Script saved to {script_path}")

        merged_path = self._merge_wavs(all_files, out / "story.wav")

        result = {
            "audio_dir": str(out),
            "files": all_files,
            "merged": str(merged_path) if merged_path else None,
            "script": script_json,
        }

        logger.info(f"✅ Generated {len(all_files)} audio files in {out}")
        return result

    @staticmethod
    def _merge_wavs(wav_paths: List[str], out_path: Path) -> Optional[Path]:
        if not wav_paths:
            return None

        valid = [p for p in wav_paths if Path(p).exists()]
        if not valid:
            return None

        try:
            with wave.open(valid[0], "rb") as first:
                params = first.getparams()

            with wave.open(str(out_path), "wb") as out_wav:
                out_wav.setparams(params)
                for path in valid:
                    with wave.open(path, "rb") as src:
                        out_wav.writeframes(src.readframes(src.getnframes()))

            return out_path
        except Exception as exc:
            logging.getLogger(__name__).warning(f"WAV merge failed: {exc}")
            return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalize text for TTS. Edge TTS is much more robust to punctuation 
        than VITS, but we still clean LLM artifacts.
        """
        import unicodedata
        if not text:
            return ""

        text = unicodedata.normalize("NFKC", text)
        text = re.sub(r"[\u200B-\u200F\u2028\u2029\uFEFF\u00AD]", "", text)
        text = re.sub(r"\s", " ", text)
        text = re.sub(r"^[A-Za-z][A-Za-z\s]{0,20}:\s*", "", text)
        text = re.sub(r"[\[\]\(\)]", "", text)
        text = re.sub(r"[\u2014\u2013]+", " ", text)
        text = text.replace("--", " ")
        text = re.sub(r"[\x00-\x1F\x7F]", "", text)
        return re.sub(r" +", " ", text).strip()

    def get_voice_map(self) -> Dict[str, str]:
        return dict(self._voice_map)
