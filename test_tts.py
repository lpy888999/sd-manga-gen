#!/usr/bin/env python3
import sys
import logging
import argparse
from TTS.api import TTS

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("tts-test")

def main():
    parser = argparse.ArgumentParser(description="Test Coqui TTS with specific parameters.")
    parser.add_argument("--text", type=str, default="Storms birth power.", help="Text to synthesize")
    parser.add_argument("--speaker", type=str, default="p230", help="Speaker ID")
    parser.add_argument("--out", type=str, default="debug.wav", help="Output file")
    args = parser.parse_args()

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

    model_name = "tts_models/en/vctk/vits"
    log.info(f"Loading {model_name} on GPU...")
    # Using gpu=True as VITS is known to have numerical instabilities on CPU
    tts = TTS(model_name, gpu=True)

    speakers = getattr(tts, "speakers", [])
    log.info(f"Available speakers: {len(speakers)}")
    if args.speaker not in speakers and speakers:
        log.warning(f"Speaker '{args.speaker}' not found. Using {speakers[0]}")
        args.speaker = speakers[0]

    # Test 1: Current Settings
    log.info("Test 1: Current settings (noise=0.33, w=0.6)")
    tts.tts_to_file(
        text=args.text,
        file_path="test1_current.wav",
        speaker=args.speaker,
        noise_scale=0.33,
        noise_scale_w=0.6,
        length_scale=1.2
    )

    # Test 2: Ultra low noise (Robotic but should be clean)
    log.info("Test 2: Ultra low noise (noise=0.01, w=0.01)")
    tts.tts_to_file(
        text=args.text,
        file_path="test2_zero_noise.wav",
        speaker=args.speaker,
        noise_scale=0.01,
        noise_scale_w=0.01,
        length_scale=1.2
    )

    # Test 3: Different Speaker (Checking if p230 is the culprit)
    alt_speaker = "p225" if "p225" in speakers else (speakers[1] if len(speakers) > 1 else args.speaker)
    log.info(f"Test 3: Different speaker ({alt_speaker})")
    tts.tts_to_file(
        text=args.text,
        file_path="test3_alt_speaker.wav",
        speaker=alt_speaker,
        noise_scale=0.33,
        noise_scale_w=0.6,
        length_scale=1.2
    )

    # Test 4: No commas (Checking if punctuation is the trigger)
    clean_text = args.text.replace(",", " ")
    log.info("Test 4: No commas")
    tts.tts_to_file(
        text=clean_text,
        file_path="test4_no_commas.wav",
        speaker=args.speaker,
        noise_scale=0.33,
        noise_scale_w=0.6,
        length_scale=1.2
    )

    log.info("All 4 test variants generated!")

if __name__ == "__main__":
    main()
