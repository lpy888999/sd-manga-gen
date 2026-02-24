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
    SD/Img2Img pass to fix IP-Adapter distortions.
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
        """Returns a list of [xmin, ymin, xmax, ymax] for detected faces."""
        if self.detector is None:
            return []
            
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24)
        )
        
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([int(x), int(y), int(x + w), int(y + h)])
                
        return boxes

    def restore(
        self,
        image: Image.Image,
        pipe,
        prompt: str,
        negative_prompt: str,
        ip_adapter_image: Image.Image = None
    ) -> Image.Image:
        """
        Detects faces in the PIL image, crops them, runs them through a localized
        img2img pass, and seamlessly blends them back.
        """
        if not self.enabled or self.detector is None:
            return image
            
        img_np = np.array(image)
        boxes = self._get_face_boxes(img_np)
        
        if not boxes:
            logger.info("Stage 3: No faces detected to restore.")
            return image
            
        logger.info(f"Stage 3: Found {len(boxes)} face(s) for restoration.")
        
        from diffusers import StableDiffusionXLImg2ImgPipeline
        i2i_pipe = StableDiffusionXLImg2ImgPipeline(**pipe.components)
        i2i_pipe.set_progress_bar_config(disable=True)
        
        # Disable IP-Adapter influence for face restoration.
        # If IP-Adapter is not loaded this is a no-op.
        try:
            i2i_pipe.set_ip_adapter_scale(0.0)
        except Exception:
            pass

        # Dummy image to satisfy diffusers' requirement that ip_adapter_image
        # must always be passed when IP-Adapter weights are loaded in the UNet,
        # even when scale=0.0.
        dummy_ip_image = ip_adapter_image or Image.new("RGB", (224, 224), (255, 255, 255))
        
        result_img = image.copy()
        
        for i, (xmin, ymin, xmax, ymax) in enumerate(boxes):
            w, h = xmax - xmin, ymax - ymin
            pad_x = int(w * 0.3)
            pad_y = int(h * 0.3)
            
            x1 = max(0, xmin - pad_x)
            y1 = max(0, ymin - pad_y)
            x2 = min(image.width, xmax + pad_x)
            y2 = min(image.height, ymax + pad_y)
            
            face_crop = result_img.crop((x1, y1, x2, y2))
            crop_w, crop_h = face_crop.size
            
            # Scale up to at least 512px on the longest side for SDXL,
            # snapping to multiples of 64.
            target_size = 512
            resize_ratio = target_size / max(crop_w, crop_h)
            if resize_ratio > 1:
                new_w = max(64, (int(crop_w * resize_ratio) // 64) * 64)
                new_h = max(64, (int(crop_h * resize_ratio) // 64) * 64)
            else:
                new_w = max(64, (crop_w // 64) * 64)
                new_h = max(64, (crop_h // 64) * 64)
                
            model_input_face = face_crop.resize((new_w, new_h), Image.LANCZOS)
            logger.info(f"  Restoring face {i+1}/{len(boxes)} (Crop: {crop_w}x{crop_h} -> Input: {new_w}x{new_h})")

            with torch.no_grad():
                restored_face = i2i_pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=model_input_face,
                    strength=self.strength,
                    num_inference_steps=30,
                    guidance_scale=5.0,
                    output_type="pil",
                    ip_adapter_image=dummy_ip_image,  # always passed
                ).images[0]

            # Resize restored face back to original crop dimensions
            restored_face_resized = restored_face.resize((crop_w, crop_h), Image.LANCZOS)
            
            # Build a feathered ellipse mask for seamless blending
            mask = Image.new("L", (crop_w, crop_h), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse(
                [pad_x * 0.5, pad_y * 0.5, crop_w - pad_x * 0.5, crop_h - pad_y * 0.5],
                fill=255
            )
            mask = mask.filter(ImageFilter.GaussianBlur(radius=min(pad_x, pad_y) * 0.5))
            
            result_img.paste(restored_face_resized, (x1, y1), mask)
            
        del i2i_pipe
        torch.cuda.empty_cache()
            
        logger.info("Stage 3: Face Restoration complete.")
        return result_img