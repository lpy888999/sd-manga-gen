"""
Layout Composer — Assemble panel images into a comic page
==========================================================
Supports two fixed layouts:
  - **4-panel**: 2×2 grid
  - **6-panel**: 3×2 grid (3 rows, 2 columns)

Comic-style thick black borders and gutters are applied automatically.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class LayoutComposer:
    """
    Assemble individual panel images into a single comic-page layout.

    Parameters
    ----------
    border_width : int
        Thickness of panel borders in pixels.
    gutter : int
        Spacing between panels in pixels.
    background_color : str or tuple
        Canvas background colour.
    """

    def __init__(
        self,
        border_width: int = 6,
        gutter: int = 12,
        background_color: str = "white",
    ):
        self.border_width = border_width
        self.gutter = gutter
        self.background_color = background_color

    # ── Layout definitions ───────────────────────────────────────────
    LAYOUTS = {
        4: {"rows": 2, "cols": 2},
        6: {"rows": 3, "cols": 2},
    }

    # ── Public API ───────────────────────────────────────────────────
    def compose(
        self,
        images: List[Image.Image],
        output_path: Optional[str] = None,
    ) -> Image.Image:
        """
        Compose panel images into a comic page.

        Parameters
        ----------
        images : list[PIL.Image.Image]
            Panel images (4 or 6).
        output_path : str | None
            If provided, save the final comic to this path.

        Returns
        -------
        PIL.Image.Image
            The assembled comic page.
        """
        panel_count = len(images)
        if panel_count not in self.LAYOUTS:
            raise ValueError(
                f"Unsupported panel count {panel_count}. "
                f"Supported: {list(self.LAYOUTS.keys())}"
            )

        layout = self.LAYOUTS[panel_count]
        rows, cols = layout["rows"], layout["cols"]

        # Resize all panels to match the first panel's dimensions
        target_w, target_h = images[0].size
        resized = [img.resize((target_w, target_h), Image.LANCZOS) for img in images]

        # Calculate canvas dimensions
        canvas_w, canvas_h = self._canvas_size(target_w, target_h, rows, cols)
        canvas = Image.new("RGB", (canvas_w, canvas_h), self.background_color)

        draw = ImageDraw.Draw(canvas)

        # Place panels on canvas
        for idx, img in enumerate(resized):
            row = idx // cols
            col = idx % cols
            x, y = self._panel_position(col, row, target_w, target_h)

            # Draw border rectangle (slightly larger than the panel)
            border = self.border_width
            draw.rectangle(
                [x - border, y - border,
                 x + target_w + border - 1, y + target_h + border - 1],
                outline="black",
                width=border,
            )

            # Paste panel image
            canvas.paste(img, (x, y))

        logger.info(f"Composed {panel_count}-panel comic ({canvas_w}×{canvas_h})")

        if output_path:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(str(out), quality=95)
            logger.info(f"Saved comic to {out}")

        return canvas

    # ── Internal geometry helpers ─────────────────────────────────────
    def _canvas_size(
        self, pw: int, ph: int, rows: int, cols: int
    ) -> Tuple[int, int]:
        """Compute total canvas (width, height)."""
        margin = self.border_width + self.gutter
        w = cols * pw + (cols + 1) * margin
        h = rows * ph + (rows + 1) * margin
        return w, h

    def _panel_position(
        self, col: int, row: int, pw: int, ph: int
    ) -> Tuple[int, int]:
        """Compute top-left (x, y) for panel at (col, row)."""
        margin = self.border_width + self.gutter
        x = margin + col * (pw + margin)
        y = margin + row * (ph + margin)
        return x, y
