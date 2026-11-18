"""Paths and display settings for the patient viewer.

Change the folder names here if your pipeline writes elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The three detectors, keyed by the folder name used under results/predictions.
# Each model was trained on, and predicts from, the images of one modality.
MODELS = {
    "ct": {"label": "CT only", "modality": "ct"},
    "pet": {"label": "PET only", "modality": "pet"},
    "fusion": {"label": "PET/CT fusion", "modality": "fusion"},
}

# The three images shown side by side, left to right.
VIEWS = {
    "ct": "CT",
    "pet": "PET",
    "fusion": "Fused PET/CT",
}

# How each view is shown. The PNG channel packing follows the README:
#   CT only: [lung window, mediastinal window, lung window]
#   PET only: [SUV, SUV, SUV]
#   Fusion:  [CT lung window, CT mediastinal window, PET SUV]
# "gray" shows one channel in grayscale, "hot" shows one channel with a heat
# color map, and "overlay" lays the PET channel in heat colors over the CT one.
# All three images are shown side by side, for whichever model is picked.
DISPLAY = {
    "ct": {"mode": "gray", "channel": 0},
    # PET stored as SUV / 15; show SUV 0-7 across the color range so tumors stand out.
    "pet": {"mode": "hot", "channel": 0, "vmax": 7 / 15},
    # PET is stored as SUV / 15 (README: clipped at 15). Color starts at SUV 2.5
    # (0.17) and is full from SUV 5 (0.33), so normal uptake stays gray.
    "fusion": {"mode": "overlay", "ct_channel": 0, "pet_channel": 2, "alpha": 0.85,
               "pet_floor": 2.5 / 15, "pet_full": 5 / 15},
}

# Chart color for each model (kept away from the green and red used for boxes).
MODEL_COLORS = {"ct": "#4C78A8", "pet": "#F58518", "fusion": "#B279A2"}

DEFAULT_SPLIT = "test"
DEFAULT_THRESHOLD = 0.25   # used only if results/thresholds.json is missing
MATCH_IOU = 0.5            # IoU for counting a predicted box as a hit (README: hit at >= 0.5)


@dataclass(frozen=True)
class Paths:
    """Where things live, relative to the project folder.

    Images and labels use the Ultralytics layout from the prototype notebook,
    one dataset folder per modality:

        yolo_data/<modality>/images/<split>/<patient>_<slice>.png
        yolo_data/<modality>/labels/<split>/<patient>_<slice>.txt

    Per-patient subfolders (images/<split>/<patient>/<slice>.png) also work.
    Saved predictions mirror the image folders:

        results/predictions/<model>/<split>/<patient>_<slice>.txt
    """

    root: Path
    data_dir: str = "yolo_data"
    results_dir: str = "results"

    def images(self, modality: str, split: str) -> Path:
        return self.root / self.data_dir / modality / "images" / split

    def labels(self, modality: str, split: str) -> Path:
        return self.root / self.data_dir / modality / "labels" / split

    def predictions(self, model: str, split: str) -> Path:
        return self.results / "predictions" / model / split

    @property
    def results(self) -> Path:
        return self.root / self.results_dir
