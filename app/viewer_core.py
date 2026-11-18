"""Data loading, box matching and drawing for the patient viewer.

Nothing in here depends on Streamlit, so it can be tested and reused by the
scripts in ``scripts/``.

Expected layout under the project root (all paths configurable in config.py):

    yolo_data/<modality>/images/<split>/<patient>_<slice>.png
    yolo_data/<modality>/labels/<split>/<patient>_<slice>.txt
    results/predictions/<model>/<split>/<patient>_<slice>.txt
    results/thresholds.json
    results/metrics.csv
    results/metrics_by_size.csv
    results/robustness.csv

Label files use YOLO format, one box per line, normalized to [0, 1]:
    class cx cy w h            (ground truth)
    class cx cy w h conf       (predictions)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import Paths

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Box:
    """A box in normalized center format, with optional confidence."""

    cx: float
    cy: float
    w: float
    h: float
    conf: float | None = None
    cls: int = 0

    def xyxy(self, width: int = 1, height: int = 1) -> tuple[float, float, float, float]:
        return (
            (self.cx - self.w / 2) * width,
            (self.cy - self.h / 2) * height,
            (self.cx + self.w / 2) * width,
            (self.cy + self.h / 2) * height,
        )


def read_yolo_file(path: Path) -> list[Box]:
    """Read a YOLO .txt file. A missing or empty file means no boxes."""
    if not path.exists():
        return []
    boxes = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) not in (5, 6):
            raise ValueError(f"{path}:{line_no}: expected 5 or 6 values, got {len(parts)}")
        cls = int(float(parts[0]))
        cx, cy, w, h = (float(v) for v in parts[1:5])
        conf = float(parts[5]) if len(parts) == 6 else None
        boxes.append(Box(cx, cy, w, h, conf, cls))
    return boxes


def write_yolo_file(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for b in boxes:
        vals = [f"{b.cls}", f"{b.cx:.6f}", f"{b.cy:.6f}", f"{b.w:.6f}", f"{b.h:.6f}"]
        if b.conf is not None:
            vals.append(f"{b.conf:.4f}")
        lines.append(" ".join(vals))
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy()
    bx1, by1, bx2, by2 = b.xyxy()
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


@dataclass
class SliceMatch:
    """Result of matching predictions to ground truth on one slice."""

    tp: list[tuple[Box, Box, float]]  # (prediction, ground truth, IoU)
    fp: list[Box]
    fn: list[Box]


def match_boxes(preds: list[Box], gts: list[Box], iou_thr: float = 0.5) -> SliceMatch:
    """Greedy matching by descending confidence, as in VOC/COCO evaluation."""
    order = sorted(preds, key=lambda p: -(p.conf if p.conf is not None else 1.0))
    used: set[int] = set()
    tp, fp = [], []
    for p in order:
        best_j, best_iou = -1, iou_thr
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = iou(p, g)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            used.add(best_j)
            tp.append((p, gts[best_j], best_iou))
        else:
            fp.append(p)
    fn = [g for j, g in enumerate(gts) if j not in used]
    return SliceMatch(tp, fp, fn)


def filter_conf(boxes: list[Box], threshold: float) -> list[Box]:
    return [b for b in boxes if (b.conf if b.conf is not None else 1.0) >= threshold]


# --------------------------------------------------------------------------- #
# Dataset access
# --------------------------------------------------------------------------- #
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def split_stem(rel: str) -> tuple[str, str]:
    """Patient ID and slice key from a path relative to the split folder.

    'Lung_Dx-A0164_042'   -> ('Lung_Dx-A0164', '042')   flat, as in the notebook
    'Lung_Dx-A0164/042'   -> ('Lung_Dx-A0164', '042')   per-patient folders
    """
    if "/" in rel:
        patient, _, key = rel.rpartition("/")
        return patient, key
    patient, sep, key = rel.rpartition("_")
    return (patient, key) if sep else (rel, "")


class Dataset:
    """Read-only access to processed slices, labels and saved predictions.

    Slices are referred to by their path relative to the split folder, without
    extension (e.g. 'Lung_Dx-A0164_042'). Labels and predictions use the same
    relative path with .txt.
    """

    def __init__(self, paths: Paths):
        self.p = paths
        self._index: dict[tuple[str, str], dict[str, list[str]]] = {}

    def _scan(self, modality: str, split: str) -> dict[str, list[str]]:
        key = (modality, split)
        if key not in self._index:
            d = self.p.images(modality, split)
            by_patient: dict[str, list[str]] = {}
            if d.is_dir():
                for f in d.rglob("*"):
                    if f.suffix.lower() in IMAGE_EXTS:
                        rel = f.relative_to(d).with_suffix("").as_posix()
                        by_patient.setdefault(split_stem(rel)[0], []).append(rel)
            for v in by_patient.values():
                v.sort(key=natural_key)
            self._index[key] = by_patient
        return self._index[key]

    # Patients and slices ---------------------------------------------------
    def patients(self, split: str, modality: str) -> list[str]:
        return sorted(self._scan(modality, split), key=natural_key)

    def slices(self, split: str, modality: str, patient: str) -> list[str]:
        """Slice references for a patient, in natural (scan) order."""
        return list(self._scan(modality, split).get(patient, []))

    def image_path(self, split: str, modality: str, stem: str) -> Path | None:
        d = self.p.images(modality, split)
        for ext in IMAGE_EXTS:
            f = d / f"{stem}{ext}"
            if f.exists():
                return f
        return None

    def load_image(self, split, modality, stem) -> Image.Image | None:
        f = self.image_path(split, modality, stem)
        return Image.open(f).convert("RGB") if f else None

    # Boxes -----------------------------------------------------------------
    def ground_truth(self, split: str, modality: str, stem: str) -> list[Box]:
        """Expert boxes. Labels are the same for every modality, so if this
        modality has no label folder, another modality's labels are used."""
        order = [modality] + [m for m in ("fusion", "ct", "pet") if m != modality]
        for m in order:
            d = self.p.labels(m, split)
            if d.is_dir():
                return read_yolo_file(d / f"{stem}.txt")
        return []

    def predictions(self, model: str, split: str, stem: str) -> list[Box]:
        return read_yolo_file(self.p.predictions(model, split) / f"{stem}.txt")

    def has_predictions(self, model: str, split: str) -> bool:
        return self.p.predictions(model, split).is_dir()

    # Saved results ---------------------------------------------------------
    def thresholds(self) -> dict[str, float]:
        f = self.p.results / "thresholds.json"
        if not f.exists():
            return {}
        return {k: float(v) for k, v in json.loads(f.read_text()).items()}

    def is_demo(self) -> bool:
        return (self.p.root / "DEMO_DATA").exists()


def patient_track(ds: Dataset, model: str, split: str, modality: str, stems: list[str]):
    """Per-slice summary for the slice strip: expert box present, top confidence."""
    has_gt = np.zeros(len(stems), dtype=bool)
    top_conf = np.zeros(len(stems), dtype=float)
    for i, s in enumerate(stems):
        has_gt[i] = bool(ds.ground_truth(split, modality, s))
        confs = [b.conf or 0.0 for b in ds.predictions(model, split, s)]
        top_conf[i] = max(confs, default=0.0)
    return has_gt, top_conf


# --------------------------------------------------------------------------- #
# Display
# --------------------------------------------------------------------------- #
GT_COLOR = (46, 204, 64)      # green: radiologist, same as the presentation
PRED_COLOR = (255, 65, 54)    # red: model
FP_COLOR = (255, 165, 0)      # orange: prediction with no matching expert box


def hot_colormap(x: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to RGB like matplotlib's 'hot' (black-red-yellow-white)."""
    x = np.clip(x, 0, 1)
    r = np.clip(x / 0.365, 0, 1)
    g = np.clip((x - 0.365) / 0.381, 0, 1)
    b = np.clip((x - 0.746) / 0.254, 0, 1)
    return np.stack([r, g, b], axis=-1)


def to_display(img: Image.Image, spec: dict | None) -> Image.Image:
    """Turn a packed 3-channel slice into something a person can read.

    spec examples (see config.DISPLAY):
        {"mode": "gray", "channel": 0}
        {"mode": "hot", "channel": 0, "vmax": 0.47}   # vmax: value shown at full color
        {"mode": "overlay", "ct_channel": 0, "pet_channel": 2, "alpha": 0.6,
         "pet_floor": 0.17, "pet_full": 0.33}
        None or {"mode": "rgb"} keeps the image as it is.
    """
    if not spec or spec.get("mode") == "rgb":
        return img
    arr = np.asarray(img).astype(np.float32) / 255.0
    mode = spec["mode"]
    if mode == "gray":
        out = np.repeat(arr[:, :, spec["channel"]][..., None], 3, axis=-1)
    elif mode == "hot":
        out = hot_colormap(arr[:, :, spec["channel"]] / spec.get("vmax", 1.0))
    elif mode == "overlay":
        ct = np.repeat(arr[:, :, spec["ct_channel"]][..., None], 3, axis=-1)
        pet = arr[:, :, spec["pet_channel"]]
        # Color only real uptake: nothing below pet_floor, full color from
        # pet_full up. Normal background uptake stays gray so the anatomy reads.
        lo, hi = spec.get("pet_floor", 0.17), spec.get("pet_full", 0.33)
        ramp = np.clip((pet - lo) / max(hi - lo, 1e-6), 0, 1)
        a = spec.get("alpha", 0.85) * ramp[..., None]
        heat = hot_colormap(0.3 + 0.32 * ramp)  # red to orange; never white, which vanishes on CT
        out = (1 - a) * ct + a * heat
    else:
        raise ValueError(f"Unknown display mode {mode!r}")
    return Image.fromarray((out * 255).round().astype(np.uint8))


def _font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _dashed_rect(draw: ImageDraw.ImageDraw, xyxy, color, width: int, dash: int = 8, gap: int = 6):
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        if length == 0:
            continue
        dx, dy = (bx - ax) / length, (by - ay) / length
        pos = 0
        while pos < length:
            end = min(pos + dash, length)
            draw.line((ax + dx * pos, ay + dy * pos, ax + dx * end, ay + dy * end),
                      fill=color, width=width)
            pos = end + gap


def draw_overlay(
    img: Image.Image,
    gts: list[Box],
    preds: list[Box],
    iou_thr: float = 0.5,
    show_gt: bool = True,
    show_pred: bool = True,
    line_width: int = 3,
) -> Image.Image:
    """Draw expert boxes (green, dashed) and model boxes (solid: red if they
    match an expert box, orange if not) with their confidence.

    The expert box is drawn last so it stays visible when the two overlap.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size
    font = _font(max(12, W // 40))
    m = match_boxes(preds, gts, iou_thr)
    matched = {id(p) for p, _, _ in m.tp}

    if show_pred:
        for p in preds:
            color = PRED_COLOR if id(p) in matched else FP_COLOR
            x1, y1, x2, y2 = p.xyxy(W, H)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            if p.conf is not None:
                label = f"{p.conf:.2f}"
                tb = draw.textbbox((0, 0), label, font=font)
                tw, th = tb[2] - tb[0], tb[3] - tb[1]
                ty = y1 - th - 6 if y1 - th - 6 >= 0 else y2 + 2
                draw.rectangle((x1, ty, x1 + tw + 6, ty + th + 6), fill=color)
                draw.text((x1 + 3, ty + 1), label, fill=(0, 0, 0), font=font)
    if show_gt:
        for g in gts:
            _dashed_rect(draw, g.xyxy(W, H), GT_COLOR, line_width)
    return out
