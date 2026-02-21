#!/usr/bin/env python3
"""
LLM Pipeline Test — Validate all text stages before diffusion
===============================================================
Tests Steps 0–2 (Character Extraction → Story Expansion → Prompt Engineering)
WITHOUT running Stable Diffusion.  Requires only Ollama running locally.

Optional ``--test-tts`` flag also tests Step 2.5 (Script Extraction).

Usage::

    python tests/test_llm_pipeline.py                    # default settings
    python tests/test_llm_pipeline.py --panels 6         # force 6-panel
    python tests/test_llm_pipeline.py --skip-vision      # skip character extraction
    python tests/test_llm_pipeline.py --config config.yaml

Test assets:
    tests/fixtures/reference_character.png   — anime character reference image
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# ── Setup project path ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from pipeline.story_expander import StoryExpander
from pipeline.prompt_engineer import PromptEngineer

# ── Constants ────────────────────────────────────────────────────────
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

# Fixed test prompts — deliberately short (→ 4 panels) and long (→ 6 panels)
TEST_PROMPTS = {
    "short": "This character fights a giant mech robot in a neon-lit city during heavy rain",
    "long": (
        "This character is a bounty hunter who receives a mysterious contract "
        "to retrieve an ancient relic from an abandoned space station. "
        "Along the way he is ambushed by alien parasites and fights them in zero gravity, "
        "only to discover that the relic is actually a sentient sword."
    ),
}


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(name)-28s │ %(levelname)-5s │ %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def divider(title: str):
    width = 70
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}\n")


def sub_divider(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")



#  STEP 1 · Story Expansion
# ═══════════════════════════════════════════════════════════════════
def test_story_expansion(config: dict, prompt: str, panel_count: int | None) -> list[str]:
    divider("STEP 1 · Story Expansion (Narrative Architect)")

    llm_cfg = config.get("llm", {})
    panel_cfg = config.get("panels", {})

    expander = StoryExpander(
        model_name=llm_cfg.get("model_name", "qwen3-coder-next:cloud"),
        temperature=llm_cfg.get("temperature", 0.7),
    )

    print(f"  💬 Input prompt: {prompt}")
    print(f"  🤖 Model:        {llm_cfg.get('model_name', 'qwen3-coder-next:cloud')}")
    print()

    t0 = time.time()
    panels = expander.expand(
        user_prompt=prompt,
        panel_count=panel_count,
        auto_threshold=panel_cfg.get("auto_threshold", 30),
    )
    elapsed = time.time() - t0

    print(f"  ✅ Generated {len(panels)} panels ({elapsed:.1f}s):")
    for i, desc in enumerate(panels):
        print(f"\n  ┌─ Panel {i + 1} ────────────────────────────────────")
        # Wrap long text
        for line in _wrap(desc, 60):
            print(f"  │ {line}")
        print(f"  └──────────────────────────────────────────────────")
    return panels


# ═══════════════════════════════════════════════════════════════════
#  STEP 2 · Prompt Engineering
# ═══════════════════════════════════════════════════════════════════
def test_prompt_engineering(config: dict, panels: list[str]):
    divider("STEP 2 · Prompt Engineering (SD Engineer)")

    llm_cfg = config.get("llm", {})
    prompt_cfg = config.get("prompt", {})
    lora_section = config.get("sd", {}).get("lora", {})

    # Build LoRA tags the same way the pipeline does
    lora_tags = []
    for adapter_name in ("character", "style"):
        adapter = lora_section.get(adapter_name, {})
        lora_dir = adapter.get("dir")
        weight = adapter.get("weight", 0.8)
        if lora_dir:
            dir_path = Path(lora_dir)
            if dir_path.is_dir():
                for sf in sorted(dir_path.glob("*.safetensors")):
                    lora_tags.append(f"<lora:{sf.stem}:{weight}>")

    trigger_words = lora_section.get("trigger_words", [])

    engineer = PromptEngineer(
        model_name=llm_cfg.get("model_name", "qwen3-coder-next:cloud"),
        temperature=0.4,
        quality_suffix=prompt_cfg.get("quality_suffix",
                                      "masterpiece, best quality, high resolution, comic style, thick lineart"),
        negative_prompt=prompt_cfg.get("negative_prompt",
                                       "low quality, blurry, distorted face, extra fingers, bad anatomy"),
        lora_tags=lora_tags,
        trigger_words=trigger_words,
    )

    print(f"  🤖 Model:          {llm_cfg.get('model_name', 'qwen3-coder-next:cloud')}")
    print(f"  🔗 LoRA tags:      {lora_tags or '(none — no .safetensors found)'}")
    print(f"  📎 Trigger words:  {trigger_words or '(none)'}")
    print()

    t0 = time.time()
    panel_prompts = engineer.generate(
        panels=panels,
    )
    elapsed = time.time() - t0

    print(f"  ✅ Generated {len(panel_prompts)} SD prompts ({elapsed:.1f}s):")
    for pp in panel_prompts:
        print(f"\n  ┌─ Panel {pp.panel_number} ({pp.camera_angle}) ─────────────")
        sub_divider("Raw SD Prompt (from LLM)")
        for line in _wrap(pp.sd_prompt, 65):
            print(f"  │ {line}")
        sub_divider("Final Prompt (with LoRA + quality)")
        for line in _wrap(pp.final_prompt, 65):
            print(f"  │ {line}")
        sub_divider("Negative Prompt")
        for line in _wrap(pp.negative_prompt, 65):
            print(f"  │ {line}")
        print(f"  └──────────────────────────────────────────────────")

    # Also dump as JSON for easy inspection
    divider("FULL JSON OUTPUT (machine-readable)")
    json_out = [
        {
            "panel_number": pp.panel_number,
            "camera_angle": pp.camera_angle,
            "sd_prompt": pp.sd_prompt,
            "final_prompt": pp.final_prompt,
            "negative_prompt": pp.negative_prompt,
        }
        for pp in panel_prompts
    ]
    print(json.dumps(json_out, ensure_ascii=False, indent=2))


# ═══════════════════════════════════════════════════════════════════
#  STEP 2.5 · Script Extraction (TTS)
# ═══════════════════════════════════════════════════════════════════
def test_script_extraction(config: dict, panels: list[str]):
    divider("STEP 2.5 · Script Extraction (TTS)")

    llm_cfg = config.get("llm", {})
    tts_cfg = config.get("tts", {})

    from pipeline.script_generator import ScriptGenerator

    # Use TTS-specific model if configured, else fallback to main LLM
    script_model = (
        tts_cfg.get("script_model_name")
        or llm_cfg.get("model_name", "qwen3-coder-next:cloud")
    )
    script_temp = tts_cfg.get("script_temperature", 0.5)

    generator = ScriptGenerator(
        model_name=script_model,
        temperature=script_temp,
    )

    print(f"  🤖 Script model: {script_model} (temp={script_temp})")
    print(f"  📄 Panels: {len(panels)}")
    print()

    t0 = time.time()
    scripts = generator.generate(panels)
    elapsed = time.time() - t0

    print(f"  ✅ Extracted scripts for {len(scripts)} panels ({elapsed:.1f}s):")
    for ps in scripts:
        print(f"\n  ┌─ Panel {ps.panel} ────────────────────────────────────")
        for line in ps.lines:
            icon = "🎙️" if line.role.lower() == "narrator" else "💬"
            print(f"  │ {icon} [{line.role}/{line.gender}]: \"{line.text}\"")
        print(f"  └──────────────────────────────────────────────────")

    # JSON dump
    divider("SCRIPT JSON (machine-readable)")
    json_out = [
        {
            "panel": ps.panel,
            "lines": [
                {"role": l.role, "text": l.text, "gender": l.gender}
                for l in ps.lines
            ],
        }
        for ps in scripts
    ]
    print(json.dumps(json_out, ensure_ascii=False, indent=2))


# ── Utility ──────────────────────────────────────────────────────────
def _wrap(text: str, width: int) -> list[str]:
    """Simple word-wrapping for terminal display."""
    words = text.split()
    lines, current = [], ""
    for w in words:
        if current and len(current) + len(w) + 1 > width:
            lines.append(current)
            current = w
        else:
            current = f"{current} {w}" if current else w
    if current:
        lines.append(current)
    return lines or [""]


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Test LLM pipeline stages (no GPU / diffusion required)",
    )
    parser.add_argument("-c", "--config", default="config.yaml", help="Config path")
    parser.add_argument("--panels", type=int, choices=[4, 6], default=None,
                        help="Force panel count")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override test prompt")
    parser.add_argument("--prompt-key", choices=["short", "long"], default="short",
                        help="Use built-in test prompt (short=4 panels, long=6)")
    parser.add_argument("--skip-vision", action="store_true",
                        help="Skip vision extraction, use hardcoded tags")
    parser.add_argument("--reference", type=str, default=None,
                        help="Custom reference image path")
    parser.add_argument("--test-tts", action="store_true",
                        help="Also test TTS script extraction (Step 2.5)")
    args = parser.parse_args()

    setup_logging()

    config = load_config(args.config)
    prompt = args.prompt or TEST_PROMPTS[args.prompt_key]
    ref_image = args.reference or str(DEFAULT_REF_IMAGE)

    print()
    print("🎨 SDXL Manga Generator — LLM Pipeline Test")
    print("=" * 50)
    print(f"  Config:    {args.config}")
    print(f"  Prompt:    {prompt}")
    print(f"  Panels:    {args.panels or 'auto'}")
    print(f"  TTS test:  {'yes' if args.test_tts else 'no'}")
    print("=" * 50)

    total_t0 = time.time()

    # ── Step 1: Story expansion ──────────────────────────────────
    panels = test_story_expansion(config, prompt, args.panels)

    # ── Step 2: Prompt engineering ───────────────────────────────
    test_prompt_engineering(config, panels)

    # ── Step 2.5: TTS script extraction (optional) ──────────────
    if args.test_tts:
        test_script_extraction(config, panels)

    # ── Summary ──────────────────────────────────────────────────
    elapsed = time.time() - total_t0
    divider(f"ALL DONE — {elapsed:.1f}s total")
    print("  Next step: feed the final_prompt values into SD generator")
    print("  (requires GPU + diffusion models)")
    if args.test_tts:
        print("  TTS script was extracted — review the dialogue above.")
    print()


if __name__ == "__main__":
    main()
