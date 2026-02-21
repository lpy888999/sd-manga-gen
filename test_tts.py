#!/usr/bin/env python3
import sys
import logging
import argparse
import asyncio
import edge_tts
from pydub import AudioSegment
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tts-test")

async def test_edge_tts(text: str, speaker: str, out_file: str):
    log.info(f"Synthesizing text: '{text}' with speaker {speaker} to {out_file}")
    
    communicate = edge_tts.Communicate(text, speaker)
    
    # Edge-TTS saves as MP3 natively
    tmp_mp3 = Path(out_file).with_suffix(".mp3")
    await communicate.save(str(tmp_mp3))
    
    # Convert to WAV to match pipeline expectations
    audio = AudioSegment.from_mp3(str(tmp_mp3))
    audio = audio.set_frame_rate(24000).set_channels(1)
    audio.export(out_file, format="wav")
    
    # Clean up
    tmp_mp3.unlink(missing_ok=True)
    log.info(f"Successfully generated {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Test Edge TTS with specific parameters.")
    parser.add_argument("--text", type=str, default="Storms birth power.", help="Text to synthesize")
    parser.add_argument("--speaker", type=str, default="en-US-ChristopherNeural", help="Speaker ID")
    parser.add_argument("--out", type=str, default="debug.wav", help="Output file")
    args = parser.parse_args()

    # We do a Unicode debug print to verify text cleanliness just in case
    def debug_unicode(text: str):
        import unicodedata
        log.info(f"RAW TEXT: {repr(text)}")
        log.info(f"LENGTH: {len(text)}")
        log.info("CODEPOINTS:")
        for i, ch in enumerate(text):
            log.info(f"{i:02d}  {ch!r}  U+{ord(ch):04X}  {unicodedata.name(ch, 'UNKNOWN')}")

    log.info("--- TEXT DEBUG INFO ---")
    debug_unicode(args.text)
    log.info("-----------------------")

    # Run the single Edge-TTS test
    asyncio.run(test_edge_tts(args.text, args.speaker, args.out))

    # Also test an alternate speaker to ensure parameter passing works
    alt_speaker = "en-US-AriaNeural"
    alt_out = str(Path(args.out).with_name("debug_alt.wav"))
    log.info(f"--- Running alternate speaker test ({alt_speaker}) ---")
    asyncio.run(test_edge_tts(args.text, alt_speaker, alt_out))

    log.info("Done!")

if __name__ == "__main__":
    main()
