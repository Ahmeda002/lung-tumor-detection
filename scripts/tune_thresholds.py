"""Pick each model's confidence threshold on the validation set.

    python scripts/tune_thresholds.py            # writes results/thresholds.json

For each model, tries thresholds from 0.05 to 0.95 and keeps the one with
the best F1 score, counting a predicted box as a hit when it overlaps an
expert box at IoU >= 0.5. Uses only validation patients, never test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import config  # noqa: E402
from viewer_core import Dataset, filter_conf, match_boxes  # noqa: E402


def counts_at(ds: Dataset, model: str, split: str, thresholds: np.ndarray):
    """Total TP, FP, FN over every slice of every patient, per threshold."""
    tp = np.zeros(len(thresholds), int)
    fp = np.zeros(len(thresholds), int)
    fn = np.zeros(len(thresholds), int)
    modality = config.MODELS[model]["modality"]
    for patient in ds.patients(split, modality):
        for stem in ds.slices(split, modality, patient):
            gts = ds.ground_truth(split, modality, stem)
            preds = ds.predictions(model, split, stem)
            for i, t in enumerate(thresholds):
                m = match_boxes(filter_conf(preds, t), gts, config.MATCH_IOU)
                tp[i] += len(m.tp)
                fp[i] += len(m.fp)
                fn[i] += len(m.fn)
    return tp, fp, fn


def best_threshold(tp, fp, fn, thresholds):
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp), float), where=denom > 0)
    i = int(np.argmax(f1))
    return float(thresholds[i]), float(f1[i])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    ds = Dataset(config.Paths(Path(args.root).resolve()))
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    out = {}
    for model in config.MODELS:
        if not ds.has_predictions(model, args.split):
            print(f"[{model}] no {args.split} predictions, skipped")
            continue
        t, f1 = best_threshold(*counts_at(ds, model, args.split, thresholds), thresholds)
        out[model] = t
        print(f"[{model}] threshold {t:.2f}  (F1 {f1:.3f} on {args.split})")

    if not out:
        sys.exit("No predictions found. Run scripts/export_predictions.py first.")
    f = ds.p.results / "thresholds.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {f}")


if __name__ == "__main__":
    main()
