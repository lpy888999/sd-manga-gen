import logging
import cv2
import numpy as np
import torch
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CASCADE_PATH = Path("/vol/bitbucket/jl10525/lbpcascade_animeface.xml")

logger = logging.getLogger("face-restorer")

class FaceRestorer:
    """
    Handles Stage 3 Anime Face Restoration.
    Uses `lbpcascade_animeface` to find bounding boxes, then uses a localized
    SD/Img2Img inpainting or upscaler run to fix IP-Adapter distortions.
    """
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.enabled = cfg.get("enabled", False)
        self.upscale_factor = cfg.get("upscale_factor", 1.0)
        self.strength = cfg.get("strength", 0.5)
        
        self.detector = None
        self._load_detector()

    def _load_detector(self):
        if not self.enabled:
            return
            
        if not CASCADE_PATH.exists():
            logger.error(f"Cascade xml not found at {CASCADE_PATH}!")
            logger.error("Please ensure you downloaded lbpcascade_animeface.xml")
            self.enabled = False
            return
            
        logger.info("Loading lbpcascade_animeface...")
        self.detector = cv2.CascadeClassifier(str(CASCADE_PATH))
            
    def _get_face_boxes(self, img_np: np.ndarray):
        """Returns a list of [xmin, ymin, xmax, ymax] for detected faces"""
        if not self.detector:
            return []
            
        # OpenCV Cascade expects grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Equalize histogram to improve contrast and detection rate
        gray = cv2.equalizeHist(gray)
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )
        
        boxes = []
        for (x, y, w, h) in faces:
            # Convert x, y, w, h -> xmin, ymin, xmax, ymax
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
                
        return boxes

    def restore(self, image: Image.Image, pipe, prompt: str, negative_prompt: str, ip_adapter_image: Image.Image = None) -> Image.Image:
        """
        Detects faces in the PIL image, crops them, runs them through a localized
        img2img pass using the provided pipeline, and seamlessly blends them back.
        """
        if not self.enabled or self.detector is None:
            return image
            
        img_np = np.array(image)
        boxes = self._get_face_boxes(img_np)
        
        if not boxes:
            logger.info("Stage 3: No faces detected to restore.")
            return image
            
        logger.info(f"Stage 3: Found {len(boxes)} face(s) for restoration.")
        
        # We need an Img2Img pipeline. We instantiate it dynamically from the T2I pipeline
        # components to share VRAM seamlessly.
        from diffusers import StableDiffusionXLImg2ImgPipeline
        i2i_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        i2i_pipe.set_progress_bar_config(disable=True)
        
        result_img = image.copy()
        
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            # Expand the bounding box slightly for context (padding)
            w, h = xmax - xmin, ymax - ymin
            pad_x = int(w * 0.3)
            pad_y = int(h * 0.3)
            
            x1 = max(0, xmin - pad_x)
            y1 = max(0, ymin - pad_y)
            x2 = min(image.width, xmax + pad_x)
            y2 = min(image.height, ymax + pad_y)
            
            # Crop the face region
            face_crop = result_img.crop((x1, y1, x2, y2))
            crop_w, crop_h = face_crop.size
            
            # SDXL needs dimensions to be multiples of 64
            # We scale up the crop to at least 512x512 for SDXL to have enough resolution to "paint" a face
            target_size = 512
            resize_ratio = target_size / max(crop_w, crop_h)
            if resize_ratio > 1:
                # Face is small, enlarge it for the model
                new_w = int(crop_w * resize_ratio)
                new_h = int(crop_h * resize_ratio)
                # Force strictly multiple of 64
                new_w = (new_w // 64) * 64
                new_h = (new_h // 64) * 64
                if new_w == 0: new_w = 64
                if new_h == 0: new_h = 64
                
                model_input_face = face_crop.resize((new_w, new_h), Image.LANCZOS)
            else:
                # Face is already large enough, just snap to 64
                new_w = (crop_w // 64) * 64
                new_h = (crop_h // 64) * 64
                if new_w == 0: new_w = 64
                if new_h == 0: new_h = 64
                model_input_face = face_crop.resize((new_w, new_h), Image.LANCZOS)
                
            logger.info(f"  Restoring face {i+1}/{len(boxes)} (Crop: {crop_w}x{crop_h} -> Input: {new_w}x{new_h})")
            # Turn OFF IP-Adapter for the face restoration so it can draw a clean anime face
            # without the reference image corrupting it again.
            enc_type = ""
            if hasattr(i2i_pipe, "unet") and hasattr(i2i_pipe.unet, "config"):
                # Use str() and or "" to safely handle None returned by FrozenDict.get
                enc_type = str(dict(i2i_pipe.unet.config).get("encoder_hid_dim_type", "") or "")
            
            safe_ip_image = ip_adapter_image
            if "ip_image_proj" in enc_type:
                i2i_pipe.set_ip_adapter_scale(0.0)
                if safe_ip_image is None:
                    safe_ip_image = Image.new("RGB", (224, 224), (255, 255, 255))
            
            # Run localized img2img
            # We use a lower strength (e.g. 0.3 - 0.5) to keep the original pose and lighting,
            # but completely redraw the messy pixels.
            with torch.no_grad():
                call_args = {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "image": model_input_face,
                    "strength": self.strength,
                    "num_inference_steps": 30, # Sufficient for low strength
                    "guidance_scale": 5.0,
                    "output_type": "pil"
                }
                
                # Only pass ip_adapter_image if the pipeline expects it
                if "ip_image_proj" in enc_type:
                    call_args["ip_adapter_image"] = safe_ip_image
                    
                restored_face = i2i_pipe(**call_args).images[0]

            # Resize the restored high-res face back to the original crop size map
            restored_face_resized = restored_face.resize((crop_w, crop_h), Image.LANCZOS)
            
            # Smooth blending (feathering)
            # Create a simple gaussian mask centered on the bounding box
            mask = Image.new("L", (crop_w, crop_h), 0)
            draw = ImageDraw.Draw(mask)
            # Draw a white ellipse in the middle (the actual face) leaving the padded area soft
            ellipse_pad_x = pad_x * 0.5
            ellipse_pad_y = pad_y * 0.5
            draw.ellipse([ellipse_pad_x, ellipse_pad_y, crop_w - ellipse_pad_x, crop_h - ellipse_pad_y], fill=255)
            
            # Blur the mask heavily for a seamless blend
            mask = mask.filter(ImageFilter.GaussianBlur(radius=min(pad_x, pad_y) * 0.5))
            
            # Paste it back
            result_img.paste(restored_face_resized, (x1, y1), mask)
            
        # Clean up temporary img2img pipeline to prevent memory leaks
        del i2i_pipe
        torch.cuda.empty_cache()
            
        logger.info("Stage 3: Face Restoration complete.")
        return result_img
