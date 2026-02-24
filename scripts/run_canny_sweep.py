#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-5s │ %(message)s")
logger = logging.getLogger("canny-sweep")

def make_canny(image: Image.Image, low_threshold: int, high_threshold: int) -> Image.Image:
    """Helper to cleanly extract Canny edges from a PIL image using cv2."""
    img_np = np.array(image.convert("RGB"))
    
    # Optional blur to reduce noise before edge detection
    # img_np = cv2.GaussianBlur(img_np, (5, 5), 0)
    
    canny_np = cv2.Canny(img_np, low_threshold, high_threshold)
    canny_np = canny_np[:, :, None]
    canny_np = np.concatenate([canny_np, canny_np, canny_np], axis=2)
    return Image.fromarray(canny_np)

def make_grid(images, labels, cols, rows):
    """Stitches images into an annotated grid."""
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
    input_path = PROJECT_ROOT / "output" / "sweep" / "debug_stage1_composition.jpg"
    
    if not input_path.exists():
        logger.error(f"Input image not found: {input_path}")
        logger.info("Please run `sbatch run_sweep.sh` first to generate the debug composition image.")
        sys.exit(1)
        
    logger.info(f"Loading base image: {input_path}")
    base_img = Image.open(input_path).convert("RGB")
    
    # ─── Define the parameter grid to sweep ───
    # X-axis: High Threshold (controls strong edges)
    # OpenCV recommendations: high should be ~2-3x low
    high_thresholds = [100, 150, 200, 250] 
    
    # Y-axis: Low Threshold (controls weak/fine edges attached to strong ones)
    low_thresholds = [25, 50, 75, 100]
    
    logger.info(f"Starting sweep: {len(high_thresholds)} High × {len(low_thresholds)} Low = {len(high_thresholds) * len(low_thresholds)} panels")

    images = []
    labels = []

    count = 1
    for low in low_thresholds:         # Rows
        for high in high_thresholds:   # Cols
            logger.info(f"--- Panel {count} | Low: {low} | High: {high} ---")
            
            # Generate the Canny map
            canny_img = make_canny(base_img, low, high)
            
            images.append(canny_img)
            labels.append(f"Low: {low} | High: {high}")
            count += 1

    # Stitch and save
    logger.info("Stitching canny matrix image...")
    grid = make_grid(images, labels, cols=len(high_thresholds), rows=len(low_thresholds))
    
    out_file = input_path.parent / "canny_threshold_matrix.jpg"
    grid.save(out_file, quality=90)
    logger.info(f"✅ Sweep complete! Matrix saved to: {out_file}")

if __name__ == "__main__":
    main()
