import logging
import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("face-restorer")

class FaceRestorer:
    """
    Handles Stage 3 Anime Face Restoration.
    Uses `anime-face-detector` to find bounding boxes, then uses a localized
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
            
        try:
            from anime_face_detector import create_detector
            # create_detector will automatically download weights from HF if missing
            logger.info("Loading anime-face-detector...")
            # We use yolov3 for fast, reliable anime face detection
            self.detector = create_detector('yolov3') 
        except ImportError:
            logger.error("anime-face-detector is not installed! Face restoration will be skipped.")
            logger.error("Please run: pip install anime-face-detector")
            self.enabled = False
            
    def _get_face_boxes(self, img_np: np.ndarray):
        """Returns a list of [xmin, ymin, xmax, ymax] for detected faces"""
        if not self.detector:
            return []
            
        # anime_face_detector expects BGR numpy array
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        preds = self.detector(img_bgr)
        
        boxes = []
        for pred in preds:
            # pred is a dict with 'bbox' and 'keypoints'
            bbox = pred['bbox']
            # bbox is [xmin, ymin, xmax, ymax, confidence]
            confidence = bbox[4]
            if confidence > 0.5: # Hardcoded threshold to avoid false positives
                boxes.append([int(x) for x in bbox[:4]])
                
        return boxes

    def restore(self, image: Image.Image, pipe, prompt: str, negative_prompt: str) -> Image.Image:
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
        
        # We need a standard img2img pipeline, not the controlnet one, 
        # so we extract the necessary components from the provided pipeline
        # (Assuming the caller passes sd._pipe which is a StableDiffusionXLPipeline)
        
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
            old_ip_scale = getattr(pipe, "ip_adapter_scale", None)
            pipe.set_ip_adapter_scale(0.0)
            
            # Run localized img2img
            # We use a lower strength (e.g. 0.3 - 0.5) to keep the original pose and lighting,
            # but completely redraw the messy pixels.
            with torch.no_grad():
                restored_face = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=model_input_face,
                    strength=self.strength,
                    num_inference_steps=30, # Sufficient for low strength
                    guidance_scale=5.0,
                    output_type="pil"
                ).images[0]
                
            # Restore IP scale if it existed
            if old_ip_scale is not None:
                if isinstance(old_ip_scale, list):
                    pipe.set_ip_adapter_scale(old_ip_scale[0])
                else:
                    pipe.set_ip_adapter_scale(old_ip_scale)

            # Resize the restored high-res face back to the original crop size map
            restored_face_resized = restored_face.resize((crop_w, crop_h), Image.LANCZOS)
            
            # Smooth blending (feathering)
            # Create a simple gaussian mask centered on the bounding box
            mask = Image.new("L", (crop_w, crop_h), 0)
            import ImageDraw
            draw = ImageDraw.Draw(mask)
            # Draw a white ellipse in the middle (the actual face) leaving the padded area soft
            ellipse_pad_x = pad_x * 0.5
            ellipse_pad_y = pad_y * 0.5
            draw.ellipse([ellipse_pad_x, ellipse_pad_y, crop_w - ellipse_pad_x, crop_h - ellipse_pad_y], fill=255)
            
            # Blur the mask heavily for a seamless blend
            from PIL import ImageFilter
            mask = mask.filter(ImageFilter.GaussianBlur(radius=min(pad_x, pad_y) * 0.5))
            
            # Paste it back
            result_img.paste(restored_face_resized, (x1, y1), mask)
            
        logger.info("Stage 3: Face Restoration complete.")
        return result_img
