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
    
    # Updated pirate prompt for testing
    prompt = "A young pirate with a straw hat and red vest stands on the deck of a wooden ship at night. A huge moon is in the background. High quality manga style."
    neg_prompt = config["prompt"].get("negative_prompt", "")
    seed = 42
    
    # Make sure output directory exists
    output_dir = PROJECT_ROOT / "output" / "sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── 3. PRE-RUN: Capture Stage 1 and Canny for debug ───
    logger.info("Generating Stage 1 debug images...")
    # Temporarily force IP scale to 0 to get stable Stage 1
    sd._pipe.set_ip_adapter_scale(0.0)
    
    # Need to pass ip_adapter_image even for scale 0 to avoid the unet crash
    stage1_img = sd._pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        width=1024,
        height=768,
        generator=torch.Generator(device=sd.device).manual_seed(seed),
        ip_adapter_image=ref_img
    ).images[0]
    
    stage1_img.save(output_dir / "debug_stage1_composition.jpg")
    
    # Extract Canny
    ts = sd._two_stage_cfg
    canny_img = sd._make_canny(
        stage1_img, 
        ts.get("canny_low_threshold", 75), 
        ts.get("canny_high_threshold", 175)
    )
    canny_img.save(output_dir / "debug_stage1_canny.jpg")
    logger.info("✅ Stage 1 and Canny debug images saved to output/sweep/")

    # ─── 4. Define the parameter grid to sweep ───
    # X-axis: ControlNet Scale (structural constraint)
    cn_scales = [0.25, 0.35, 0.45] 
    # Y-axis: Stage 2 Strength (denoising / freedom)
    stage2_strengths = [0.35, 0.45, 0.55]
    fixed_ip_scale = 0.50
    
    logger.info(f"Starting sweep: {len(cn_scales)} CN scales × {len(stage2_strengths)} Strengths = {len(cn_scales) * len(stage2_strengths)} panels")

    images = []
    labels = []

    count = 1
    # Iterate dynamically
    for strength in stage2_strengths:      # Rows
        for cn_scale in cn_scales:         # Cols
            logger.info(f"--- Panel {count} | CN: {cn_scale} | Strength: {strength} ---")
            
            # Hot-swap the two-stage configuration dynamically
            sd._two_stage_cfg["controlnet_scale"] = cn_scale
            sd._two_stage_cfg["stage2_strength"] = strength
            sd._two_stage_cfg["stage2_ip_scale"] = fixed_ip_scale
            
            # Generate exactly one panel
            img = sd.generate_panel(
                prompt=prompt,
                negative_prompt=neg_prompt,
                width=1024,
                height=768,
                seed=seed,
                ip_adapter_image=ref_img
            )
            
            images.append(img)
            labels.append(f"CN: {cn_scale} | Str: {strength}")
            count += 1

    # 5. Stitch and save
    logger.info("Stitching matrix image...")
    grid = make_grid(images, labels, cols=len(cn_scales), rows=len(stage2_strengths))
    
    out_file = output_dir / "parameter_sweep_matrix.jpg"
    grid.save(out_file, quality=90)
    logger.info(f"✅ Sweep complete! Matrix saved to: {out_file}")

if __name__ == "__main__":
    main()
