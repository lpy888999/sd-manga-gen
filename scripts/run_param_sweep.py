#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
from PIL import Image, ImageDraw
from pipeline.sd_generator import SDGenerator

# Configure logging to pass through
logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-5s │ %(message)s")
logger = logging.getLogger("sweep")

def make_grid(images, labels, cols, rows):
    """Stitches images into a grid and overlays labels."""
    if not images:
        return None
    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h), 'white')
    draw = ImageDraw.Draw(grid)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        r = i // cols
        c = i % cols
        x, y = c * w, r * h
        grid.paste(img, (x, y))
        
        # Add a black background box for the text label
        draw.rectangle([x, y, x + 350, y + 40], fill="black")
        draw.text((x + 15, y + 15), label, fill="white")
        
    return grid

def main():
    config_path = PROJECT_ROOT / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 1. Initialize SD Generator once (saves VRAM & time)
    logger.info("Initializing SDGenerator...")
    sd = SDGenerator.from_config(config["sd"])

    # 2. Setup inputs
    ref_path = PROJECT_ROOT / "tests/fixtures/luffy.jpg"
    if not ref_path.exists():
        logger.error(f"Reference image not found: {ref_path}")
        sys.exit(1)
        
    ref_img = Image.open(ref_path).convert("RGB")
    
    # Prompt for testing
    prompt = "A young man stands on a roof at night. There is a full moon glowing brightly behind him. A ninja in dark clothing jumps down towards him. He holds a sword."
    neg_prompt = config["prompt"].get("negative_prompt", "")
    seed = 42
    
    # ─── 3. Define the parameter grid to sweep ───
    # X-axis: ControlNet Scale (structural constraint)
    cn_scales = [0.25, 0.35, 0.45] 
    # Y-axis: Stage 2 Strength (denoising / freedom)
    stage2_strengths = [0.35, 0.45, 0.55]
    fixed_ip_scale = 0.50
    
    logger.info(f"Starting sweep: {len(cn_scales)} CN scales × {len(stage2_strengths)} Strengths = {len(cn_scales) * len(stage2_strengths)} panels")

    images = []
    labels = []

    # Make sure output directory exists
    output_dir = PROJECT_ROOT / "output" / "sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 1
    # Iterate dynamically
    for strength in stage2_strengths:      # Rows
        for cn_scale in cn_scales:         # Cols
            logger.info(f"--- Panel {count} | CN: {cn_scale} | Strength: {strength} ---")
            
            # Hot-swap the two-stage configuration dynamically
            sd._two_stage_cfg["controlnet_scale"] = cn_scale
            sd._two_stage_cfg["stage2_strength"] = strength
            sd._two_stage_cfg["stage2_ip_scale"] = fixed_ip_scale
            
            # Generate exactly one panel (skips LLM layout entirely)
            img = sd.generate_panel(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=1024,
                height=768,
                seed=seed, # fix seed so everything else is identical
                ip_adapter_image=ref_img
            )
            
            images.append(img)
            labels.append(f"CN Scale: {cn_scale} | Denoise: {strength} | IP: {fixed_ip_scale}")
            count += 1

    # 4. Stitch and save
    logger.info("Stitching matrix image...")
    grid = make_grid(images, labels, cols=len(cn_scales), rows=len(stage2_strengths))
    
    out_file = output_dir / "parameter_sweep_matrix.jpg"
    grid.save(out_file, quality=90)
    logger.info(f"✅ Sweep complete! Matrix saved to: {out_file}")

if __name__ == "__main__":
    main()
