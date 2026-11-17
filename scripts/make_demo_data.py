"""Make a small synthetic dataset so the viewer can run before real results exist.

    python scripts/make_demo_data.py --out demo
    streamlit run app/streamlit_app.py -- --root demo

Everything written here is fake: cartoon chest phantoms, made-up boxes and
made-up scores. The folder is tagged with a DEMO_DATA file and the app shows
a warning whenever it finds one. Do not put these numbers in a report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from viewer_core import Box, write_yolo_file  # noqa: E402

SIZE = 640
MODELS = {"ct": 0.70, "pet": 0.75, "fusion": 0.85}  # demo detection skill


def phantom(rng, n_slices, tumors):
    """Return CT (HU-like) and PET (SUV-like) volumes, shape (n, SIZE, SIZE)."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    cx, cy = SIZE / 2, SIZE / 2
    body = ((xx - cx) / 260) ** 2 + ((yy - cy) / 200) ** 2 <= 1
    lung_l = ((xx - cx + 110) / 85) ** 2 + ((yy - cy + 10) / 130) ** 2 <= 1
    lung_r = ((xx - cx - 110) / 85) ** 2 + ((yy - cy + 10) / 130) ** 2 <= 1
    spine = ((xx - cx) / 28) ** 2 + ((yy - cy - 140) / 28) ** 2 <= 1
    heart = ((xx - cx + 20) / 70) ** 2 + ((yy - cy - 20) / 60) ** 2 <= 1

    ct = np.full((n_slices, SIZE, SIZE), -1000, np.float32)
    pet = np.zeros((n_slices, SIZE, SIZE), np.float32)
    for z in range(n_slices):
        s = ct[z]
        s[body] = 40
        s[lung_l | lung_r] = -850
        s[heart] = 50
        s[spine] = 700
        s += rng.normal(0, 25, s.shape)
        p = pet[z]
        p[body] = 0.8
        p[heart] = 2.5
        p[lung_l | lung_r] = 0.4
        p += np.abs(rng.normal(0, 0.15, p.shape))

    for t in tumors:
        for z in range(n_slices):
            dz = z - t["z"]
            if abs(dz) >= t["r"]:
                continue
            r = np.sqrt(t["r"] ** 2 - dz ** 2) * t["px_per_slice"]
            m = (xx - t["x"]) ** 2 + (yy - t["y"]) ** 2 <= r ** 2
            ct[z][m] = 30 + rng.normal(0, 15, m.sum())
            pet[z][m] = t["suv"] * (0.6 + 0.4 * (r / (t["r"] * t["px_per_slice"])))
    return ct, pet


def window(hu, center, width):
    lo, hi = center - width / 2, center + width / 2
    return (np.clip((hu - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def suv_png(suv, max_suv=15.0):  # README: SUV clipped at 15
    return (np.clip(suv / max_suv, 0, 1) * 255).astype(np.uint8)


def gt_boxes(tumors, z):
    boxes = []
    for t in tumors:
        dz = z - t["z"]
        if abs(dz) >= t["r"]:
            continue
        r = np.sqrt(t["r"] ** 2 - dz ** 2) * t["px_per_slice"]
        if r < 3:
            continue
        pad = 3
        boxes.append(Box(t["x"] / SIZE, t["y"] / SIZE, (2 * r + pad) / SIZE, (2 * r + pad) / SIZE))
    return boxes


def fake_predictions(rng, gts, skill):
    preds = []
    for g in gts:
        size = g.w * SIZE
        p_detect = skill * min(1.0, size / 40)  # small tumors get missed more
        if rng.random() < p_detect:
            j = rng.normal(0, 0.08, 4)
            preds.append(Box(
                g.cx + j[0] * g.w, g.cy + j[1] * g.h,
                max(0.005, g.w * (1 + j[2])), max(0.005, g.h * (1 + j[3])),
                conf=float(np.clip(rng.normal(0.35 + 0.55 * skill * min(1, size / 50), 0.1), 0.05, 0.99)),
            ))
    if rng.random() < (1 - skill) * 0.5:
        preds.append(Box(rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), 0.04, 0.04,
                         conf=float(rng.uniform(0.05, 0.45))))
    return preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="demo")
    ap.add_argument("--patients", type=int, default=4)
    ap.add_argument("--slices", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out)
    data = root / "yolo_data"
    pred_root = root / "results/predictions"

    for i in range(args.patients):
        pid = f"DEMO-{i + 1:04d}"
        n_tumors = 1 + (i % 2)
        tumors = []
        for _ in range(n_tumors):
            side = rng.choice([-1, 1])
            tumors.append(dict(
                x=SIZE / 2 + side * rng.uniform(80, 140), y=SIZE / 2 + rng.uniform(-80, 60),
                z=rng.uniform(15, args.slices - 15), r=rng.uniform(4, 12),
                px_per_slice=rng.uniform(2.0, 3.5), suv=rng.uniform(4, 12),
            ))
        ct, pet = phantom(rng, args.slices, tumors)
        for z in range(args.slices):
            stem = f"{pid}_{z:03d}"  # same naming as the prototype notebook
            lung = window(ct[z], -600, 1500)
            med = window(ct[z], 40, 350)
            suv = suv_png(pet[z])
            images = {  # channel packing from the README
                "ct": np.dstack([lung, med, lung]),
                "pet": np.dstack([suv, suv, suv]),
                "fusion": np.dstack([lung, med, suv]),
            }
            gts = gt_boxes(tumors, z)
            for mod, arr in images.items():
                d = data / mod / "images" / "test"
                d.mkdir(parents=True, exist_ok=True)
                Image.fromarray(arr).save(d / f"{stem}.png")
                write_yolo_file(data / mod / "labels" / "test" / f"{stem}.txt", gts)
            for m, skill in MODELS.items():
                write_yolo_file(pred_root / m / "test" / f"{stem}.txt",
                                fake_predictions(rng, gts, skill))

    res = root / "results"
    (res / "thresholds.json").write_text(json.dumps({"ct": 0.40, "pet": 0.35, "fusion": 0.30}, indent=2))

    rows = []
    for m, skill in MODELS.items():
        for metric, base in [("precision", 0.80), ("recall", 0.75), ("F1", 0.77), ("mean IoU", 0.70),
                             ("mAP50", 0.75), ("mAP50-95", 0.40)]:
            v = base * skill / 0.85
            rows.append(dict(model=m, metric=metric, value=round(v, 3),
                             ci_low=round(v - 0.06, 3), ci_high=round(min(1, v + 0.06), 3)))
    pd.DataFrame(rows).to_csv(res / "metrics.csv", index=False)

    rows = []
    for m, skill in MODELS.items():
        for size_bin, f in [("small (<1 cm)", 0.45), ("medium (1-3 cm)", 0.85), ("large (>3 cm)", 1.0)]:
            for metric in ["recall", "mAP50"]:
                v = 0.9 * skill * f
                rows.append(dict(model=m, size_bin=size_bin, metric=metric, value=round(v, 3),
                                 ci_low=round(max(0, v - 0.1), 3), ci_high=round(min(1, v + 0.1), 3)))
    pd.DataFrame(rows).to_csv(res / "metrics_by_size.csv", index=False)

    # The three tests in the README: added noise, a missing modality, PET misaligned
    tests = {
        "added noise": [("none", 0), ("low", 1), ("medium", 2), ("high", 3)],
        "missing modality": [("none", 0), ("PET blanked", 3), ("CT blanked", 2)],
        "PET misaligned": [("0 px", 0), ("2 px", 1), ("5 px", 2), ("10 px", 3)],
    }
    rows = []
    for m, skill in MODELS.items():
        for pert, levels in tests.items():
            for label, k in levels:
                hit = 0.0 if m == "ct" and pert != "added noise" and "CT" not in label else 0.08 * k
                if m == "ct" and label == "CT blanked":
                    hit = 0.9
                if m == "pet" and label == "PET blanked":
                    hit = 0.9
                v = max(0.0, 0.75 * skill / 0.85 * (1 - hit))
                rows.append(dict(model=m, perturbation=pert, level=label, metric="mAP50", value=round(v, 3)))
    pd.DataFrame(rows).to_csv(res / "robustness.csv", index=False)

    (root / "DEMO_DATA").write_text("Synthetic data from scripts/make_demo_data.py. Not real results.\n")
    print(f"Wrote demo data for {args.patients} patients to {root.resolve()}")


if __name__ == "__main__":
    main()
