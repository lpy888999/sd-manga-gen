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

    model_name = "tts_models/en/vctk/vits"
    log.info(f"Loading {model_name}...")
    tts = TTS(model_name, gpu=False)

    speakers = getattr(tts, "speakers", [])
    log.info(f"Available speakers: {len(speakers)}")
    if args.speaker not in speakers and speakers:
        log.warning(f"Speaker '{args.speaker}' not found. Using {speakers[0]}")
        args.speaker = speakers[0]

    log.info(f"Synthesizing text: '{args.text}' with speaker {args.speaker} to {args.out}")
    tts.tts_to_file(
        text=args.text,
        file_path=args.out,
        speaker=args.speaker,
        noise_scale=0.33,
        noise_scale_w=0.6,
        length_scale=1.2
    )

    log.info("Done!")

if __name__ == "__main__":
    main()
