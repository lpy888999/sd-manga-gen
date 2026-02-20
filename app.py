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
import shutil
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
    str | None
        Path to merged story.wav if audio enabled, else None.
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

    # Use a persistent temp dir so Gradio can serve the files after return
    tmpdir = tempfile.mkdtemp(prefix="manga_")
    output_path = os.path.join(tmpdir, "comic.png")
    audio_dir = os.path.join(tmpdir, "audio") if enable_audio else None

    try:
        pipe = get_pipeline()
        result = pipe.run(
            reference_image=reference_image,
            user_prompt=prompt.strip(),
            panel_count=panels,
            output_path=output_path,
            seed=actual_seed,
            enable_audio=enable_audio,
            audio_dir=audio_dir,
        )

        from PIL import Image
        comic_path = result["comic_path"]
        comic = Image.open(comic_path).copy()
        status = f"✅ Generated {panels or 'auto'}-panel comic"
        if actual_seed:
            status += f" (seed={actual_seed})"

        # Return merged audio path for gr.Audio player
        merged_wav = None
        audio_data = result.get("audio")
        if audio_data:
            merged_wav = audio_data.get("merged")
            n_clips = len(audio_data.get("files", []))
            if merged_wav:
                status += f" + {n_clips} audio clips → story.wav"

        return comic, status, merged_wav

    except Exception as e:
        logger.exception("Pipeline error")
        return None, f"❌ Error: {e}", None


# ── Gradio UI ────────────────────────────────────────────────────────
TITLE = "🎨 SDXL Manga Generator"
DESCRIPTION = """\
**Reference Image → Story Expansion → SD Prompt Engineering → Stable Diffusion + LoRA + IP-Adapter → Comic Layout**

Upload a character reference image, enter a story concept, and generate a multi-panel manga page.
Enable 🔊 Audio to add per-panel voice-over (Coqui TTS) merged into a single playable track.
"""

EXAMPLES = [
    [None, "A samurai fighting a robot in the rain", "4 Panels", 42],
    [None, "A girl discovers an ancient temple hidden in a bamboo forest", "6 Panels", 0],
    [None, "太空探险家在废弃空间站发现了神秘信号", "4 Panels", 0],
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
                    label="📷 Character Reference Image (Optional - IP-Adapter)",
                    type="filepath",
                    height=200,
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

                enable_audio = gr.Checkbox(
                    label="🔊 Enable Audio (TTS voice-over)",
                    value=False,
                )

                generate_btn = gr.Button(
                    "🚀 Generate Comic",
                    variant="primary",
                    size="lg",
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
                audio_output = gr.Audio(
                    label="🔊 Story Audio (merged)",
                    type="filepath",
                    visible=True,
                )

        # ── Examples ─────────────────────────────────────────────────
        gr.Examples(
            examples=EXAMPLES,
            inputs=[reference_image, prompt, panel_count, seed],
            label="💡 Example Prompts",
        )

        # ── Wire up ─────────────────────────────────────────────────
        generate_btn.click(
            fn=generate_comic,
            inputs=[reference_image, prompt, panel_count, seed, enable_audio],
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
