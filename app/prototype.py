"""Saved Semester 1 examples from the prototype notebook's validation figure."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image

from viewer_core import Box, to_display

PATIENT = "Lung_Dx-A0164"
FIGURE_SIZE = (1570, 430)
PANEL_TOP = 83
PANEL_SIZE = 337


@dataclass(frozen=True)
class PrototypeCase:
    label: str
    slice_number: int
    left: int
    expert: tuple[int, int, int, int]
    prediction: tuple[int, int, int, int] | None = None
    confidence: float | None = None


# Panel and box positions are traced in the saved figure, in pixels. The two
# cases and displayed confidence are the same as those on the Prototype slide.
CASES = (
    PrototypeCase("Detection", 290, 10, (202, 119, 254, 225), (201, 121, 257, 224), 0.86),
    PrototypeCase("Missed tumor", 306, 819, (193, 198, 236, 229)),
)


def load_validation_figure(path: Path) -> Image.Image:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if 'plt.suptitle("green = radiologist, red = model")' not in source:
            continue
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if encoded:
                with Image.open(BytesIO(base64.b64decode("".join(encoded)))) as image:
                    if image.size != FIGURE_SIZE:
                        raise ValueError("The saved validation figure has a different layout.")
                    return image.convert("RGB")
    raise ValueError("The saved validation figure is missing from the prototype notebook.")


def normalized_box(bounds: tuple[int, int, int, int], confidence: float | None = None) -> Box:
    x1, y1, x2, y2 = bounds
    return Box((x1 + x2) / (2 * PANEL_SIZE), (y1 + y2) / (2 * PANEL_SIZE),
               (x2 - x1) / PANEL_SIZE, (y2 - y1) / PANEL_SIZE, confidence)


def scan_views(figure: Image.Image, case: PrototypeCase) -> dict[str, Image.Image]:
    packed = figure.crop((case.left, PANEL_TOP, case.left + PANEL_SIZE, PANEL_TOP + PANEL_SIZE))
    # Annotation strokes replaced channel samples in the saved plot. Mask their
    # solid pixels near the boxes when displaying the individual channels.
    channels = packed.copy()
    x1, y1, x2, y2 = case.expert
    if case.prediction:
        px1, py1, px2, py2 = case.prediction
        x1, y1 = min(x1, px1), min(y1, py1 - 18)
        x2, y2 = max(x2, px2), max(y2, py2)
    pixels = channels.load()
    for y in range(y1 - 2, y2 + 3):
        for x in range(x1 - 2, x2 + 3):
            if pixels[x, y] in ((0, 255, 0), (255, 0, 0)):
                pixels[x, y] = (0, 0, 0)
    return {
        "ct": to_display(channels, {"mode": "gray", "channel": 0}),
        "pet": to_display(channels, {"mode": "hot", "channel": 2}),
        "fusion": packed,
    }
