#!/usr/bin/env python3
"""
SDXL Manga Generation Pipeline — Gradio Frontend
===================================================
Designed for deployment on HuggingFace Spaces.

Launch locally::

    python app.py

HuggingFace Spaces will auto-detect ``app.py`` as the entry point.
"""

import logging
import tempfile
import os
import sys
from pathlib import Path

import gradio as gr
import yaml

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.manga_pipeline import MangaPipeline

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-28s │ %(levelname)-5s │ %(message)s",
    datefmt="%H:%M:%S",
)
for lib in ("httpx", "httpcore", "urllib3", "diffusers", "transformers"):
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger("app")

# ── Load config & build pipeline (once at startup) ───────────────────
CONFIG_PATH = os.getenv("MANGA_CONFIG", "config.yaml")

_pipeline = None

def get_pipeline() -> MangaPipeline:
    """Lazy-load pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        logger.info(f"Loading pipeline from {CONFIG_PATH} …")
        _pipeline = MangaPipeline.from_config(CONFIG_PATH)
    return _pipeline


# ── Core generation function ─────────────────────────────────────────
def generate_comic(
    reference_image: str | None,
    character_tags: str,
    prompt: str,
    panel_count: str,
    seed: int,
    enable_audio: bool,
):
    """
    Gradio callback — runs the full manga generation pipeline.

    Returns
    -------
    PIL.Image.Image | None
        The generated comic page, or None on error.
    str
        Status / log message.
    list | None
        Audio file paths if audio enabled, else None.
    """
    if not prompt or not prompt.strip():
        return None, "⚠️ Please enter a story prompt.", None

    # Parse panel count
    panels = None  # auto
    if panel_count == "4 Panels":
        panels = 4
    elif panel_count == "6 Panels":
        panels = 6

    # Seed: 0 or -1 = random
    actual_seed = seed if seed > 0 else None

    # Character tags (empty string → None → use vision extraction)
    tags = character_tags.strip() if character_tags else None

    # Output to a temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "comic.png")

        try:
            pipe = get_pipeline()
            result = pipe.run(
                reference_image=reference_image,
                character_tags=tags,
                user_prompt=prompt.strip(),
                panel_count=panels,
                output_path=output_path,
                seed=actual_seed,
                enable_audio=enable_audio,
            )

            from PIL import Image
            comic_path = result["comic_path"]
            comic = Image.open(comic_path).copy()
            status = f"✅ Generated {panels or 'auto'}-panel comic"
            if actual_seed:
                status += f" (seed={actual_seed})"

            # Collect audio files
            audio_files = None
            audio_data = result.get("audio")
            if audio_data and audio_data.get("files"):
                audio_files = audio_data["files"]
                status += f" + {len(audio_files)} audio clips"

            return comic, status, audio_files

        except Exception as e:
            logger.exception("Pipeline error")
            return None, f"❌ Error: {e}", None


# ── Gradio UI ────────────────────────────────────────────────────────
TITLE = "🎨 SDXL Manga Generator"
DESCRIPTION = """\
**Reference Image → Story Expansion → SD Prompt Engineering → Stable Diffusion + LoRA → Comic Layout**

Upload a character reference image (or provide manual tags), enter a story concept, and generate a multi-panel manga page.
Enable 🔊 Audio to add per-panel voice-over (Coqui TTS).
"""

EXAMPLES = [
    [None, "1boy, silver hair, round glasses, black trench coat",
     "A samurai fighting a robot in the rain", "4 Panels", 42],
    [None, "1girl, long blue hair, white uniform, katana",
     "A girl discovers an ancient temple hidden in a bamboo forest", "6 Panels", 0],
    [None, "1boy, spiky black hair, red scarf, mechanical arm",
     "太空探险家在废弃空间站发现了神秘信号", "4 Panels", 0],
]

CSS = """
.gradio-container { max-width: 1100px !important; }
#comic-output { min-height: 400px; }
.gr-button-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; }
"""

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title=TITLE,
        css=CSS,
        theme=gr.themes.Soft(
            primary_hue="purple",
            secondary_hue="blue",
        ),
    ) as demo:
        gr.Markdown(f"# {TITLE}")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            # ── Left column: Inputs ──────────────────────────────────
            with gr.Column(scale=1):
                reference_image = gr.Image(
                    label="📷 Character Reference Image (optional)",
                    type="filepath",
                    height=200,
                )
                character_tags = gr.Textbox(
                    label="🏷️ Character Tags (optional, overrides image extraction)",
                    placeholder="e.g. 1boy, silver hair, round glasses, black trench coat",
                    lines=2,
                )
                prompt = gr.Textbox(
                    label="📝 Story Prompt",
                    placeholder="e.g. A samurai fighting a robot in the rain",
                    lines=3,
                )

                with gr.Row():
                    panel_count = gr.Radio(
                        ["Auto", "4 Panels", "6 Panels"],
                        value="Auto",
                        label="📐 Panel Count",
                    )
                    seed = gr.Number(
                        label="🎲 Seed (0 = random)",
                        value=0,
                        precision=0,
                    )

                generate_btn = gr.Button(
                    "🚀 Generate Comic",
                    variant="primary",
                    size="lg",
                )

                enable_audio = gr.Checkbox(
                    label="🔊 Enable Audio (TTS voice-over)",
                    value=False,
                )

            # ── Right column: Output ─────────────────────────────────
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="🖼️ Generated Comic",
                    elem_id="comic-output",
                    type="pil",
                    height=500,
                )
                status = gr.Textbox(
                    label="Status",
                    interactive=False,
                )
                audio_output = gr.File(
                    label="🔊 Audio Files",
                    file_count="multiple",
                    visible=True,
                )

        # ── Examples ─────────────────────────────────────────────────
        gr.Examples(
            examples=EXAMPLES,
            inputs=[reference_image, character_tags, prompt, panel_count, seed],
            label="💡 Example Prompts",
        )

        # ── Wire up ─────────────────────────────────────────────────
        generate_btn.click(
            fn=generate_comic,
            inputs=[reference_image, character_tags, prompt, panel_count, seed, enable_audio],
            outputs=[output_image, status, audio_output],
        )

    return demo


# ── Launch ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
