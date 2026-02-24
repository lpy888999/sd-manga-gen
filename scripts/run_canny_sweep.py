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
    
    # --- Edge Extractor 1: Canny (OpenCV) ---
    logger.info("Extracting Canny...")
    img_np = np.array(base_img)
    canny_np = cv2.Canny(img_np, 75, 175)
    canny_np = canny_np[:, :, None]
    canny_img = Image.fromarray(np.concatenate([canny_np, canny_np, canny_np], axis=2))
    images.append(canny_img)
    labels.append("Canny (cv2)")

    # --- Edge Extractor 2: Sobel (OpenCV) ---
    logger.info("Extracting Sobel...")
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx**2 + sobely**2)
    sobel_mag = np.uint8(255 * sobel_mag / np.max(sobel_mag))
    # Thresholding slightly to clean it up like a lineart
    _, sobel_thresh = cv2.threshold(sobel_mag, 50, 255, cv2.THRESH_BINARY)
    sobel_thresh = sobel_thresh[:, :, None]
    sobel_img = Image.fromarray(np.concatenate([sobel_thresh, sobel_thresh, sobel_thresh], axis=2))
    images.append(sobel_img)
    labels.append("Sobel Filter")

    # --- Edge Extractor 3: HED (controlnet_aux) ---
    logger.info("Extracting HED...")
    try:
        from controlnet_aux import HEDdetector
        hed = HEDdetector.from_pretrained("lllyasviel/Annotators")
        hed_img = hed(base_img)
        images.append(hed_img)
        labels.append("HED (controlnet_aux)")
    except ImportError:
        logger.error("controlnet_aux missing! Skipping HED.")
        images.append(Image.new('RGB', base_img.size, 'black'))
        labels.append("HED (Missing Package)")

    # --- Edge Extractor 4: Lineart Anime (controlnet_aux) ---
    logger.info("Extracting Lineart Anime (AnimeLineart)...")
    try:
        from controlnet_aux import LineartAnimeDetector
        lineart = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
        lineart_img = lineart(base_img)
        images.append(lineart_img)
        labels.append("Lineart Anime (controlnet_aux)")
    except ImportError:
        logger.error("controlnet_aux missing! Skipping Lineart.")
        images.append(Image.new('RGB', base_img.size, 'black'))
        labels.append("Lineart Anime (Missing Package)")

    # Stitch and save
    logger.info("Stitching 2x2 comparison matrix image...")
    grid = make_grid(images, labels, cols=2, rows=2)
    
    out_file = input_path.parent / "edge_detectors_comparison.jpg"
    grid.save(out_file, quality=90)
    logger.info(f"✅ Comparison complete! Matrix saved to: {out_file}")

if __name__ == "__main__":
    main()
