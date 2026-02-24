#!/usr/bin/env python3
"""
SDXL Manga Generation Pipeline — CLI Entry Point
==================================================

Usage examples::

    # Full pipeline with reference image
    python main.py -r ref.png -p "A samurai fighting a robot in the rain"

    # Manual character tags (skip vision extraction)
    python main.py --character-tags "1boy, silver hair, black coat" \\
                   -p "A detective solving a mystery in a dark library" \\
                   --panels 6

    # Custom config and output path
    python main.py -r ref.png -p "..." -c my_config.yaml -o output/my_comic.png

    # With fixed seed for reproducibility
    python main.py -r ref.png -p "..." --seed 42
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDXL Manga Generation Pipeline — "
                    "Generate fixed-layout comics from a text prompt",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python main.py -r character.png -p "A samurai duels a robot in the rain"
  python main.py -p "校园日常" --panels 6
  python main.py -r ref.png -p "宇宙探险" --seed 42 -o output/space_comic.png
""",
    )

    # ── Required ─────────────────────────────────────────────────────
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        required=True,
        help="Story concept / idea to expand into a comic",
    )

    # ── Character source ─────────────────────────────────────────────
    parser.add_argument(
        "-r", "--reference",
        type=str,
        default=None,
        help="Path to character reference image (PNG/JPG/WEBP)",
    )

    # ── Optional ─────────────────────────────────────────────────────
    parser.add_argument(
        "--panels",
        type=int,
        choices=[4, 6],
        default=None,
        help="Number of panels (4 or 6). Default: auto-detect from prompt length",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output/comic.png",
        help="Output file path (default: output/comic.png)",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="config.yaml",
        help="Path to config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )
    parser.add_argument(
        "--audio",
        action="store_true",
        help="Enable TTS audio generation (requires Coqui TTS)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory for TTS audio output (overrides config tts.output_dir)",
    )
    parser.add_argument(
        "--prompt-cache",
        type=str,
        default=None,
        help="Path to JSON file to save/load generated prompts to eliminate LLM variance",
    )

    return parser.parse_args()


def setup_logging(verbose: bool = False, log_file: str = None):
    """Configure logging for the pipeline."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(name)-30s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
        force=True, # MUST FORCE OVERRIDE to prevent diffusers/transformers hijacking
        handlers=handlers
    )
    
    # Explicitly set the root logger level
    logging.root.setLevel(level)
    # Silence noisy libraries
    for lib in ("httpx", "httpcore", "urllib3", "diffusers", "transformers"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Disable diffusers progress bars to keep logs clean
    try:
        from diffusers.utils.logging import disable_progress_bar
        disable_progress_bar()
    except ImportError:
        pass


def main():
    args = parse_args()
    
    # Determine output directory and setup logging to file
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"
    
    setup_logging(args.verbose, str(log_file))

    log = logging.getLogger("main")
    log.info("=" * 60)
    log.info("  SDXL Manga Generation Pipeline")
    log.info("=" * 60)
    log.info(f"  Prompt:     {args.prompt}")
    log.info(f"  Reference:  {args.reference or '(none)'}")
    log.info(f"  Panels:     {args.panels or 'auto'}")
    log.info(f"  Seed:       {args.seed or 'random'}")
    log.info(f"  Output:     {args.output}")
    log.info(f"  Audio dir:  {args.audio_dir or '(config default)'}")
    log.info("=" * 60)

    # Validate reference image exists
    if args.reference and not Path(args.reference).exists():
        log.error(f"Reference image not found: {args.reference}")
        sys.exit(1)

    # Lazy import to avoid loading heavy deps (torch, PIL, etc.) for --help
    from pipeline.manga_pipeline import MangaPipeline

    # Re-setup logging after importing heavy ML libs (diffusers/transformers)
    # which may hijack or clear root logger handlers during import.
    setup_logging(args.verbose, str(log_file))

    # Build and run pipeline
    pipeline = MangaPipeline.from_config(args.config)

    output = pipeline.run(
        reference_image=args.reference,
        user_prompt=args.prompt,
        panel_count=args.panels,
        output_path=args.output,
        seed=args.seed,
        enable_audio=args.audio if args.audio else None,
        audio_dir=args.audio_dir,
        prompt_cache=args.prompt_cache,
    )

    log.info(f"\u2705 Comic saved to: {output['comic_path']}")
    if output.get("audio"):
        log.info(f"\U0001f50a Audio files: {output['audio']['audio_dir']}")


if __name__ == "__main__":
    main()
