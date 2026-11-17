"""Run each trained detector once and save its boxes for the viewer.

    python scripts/export_predictions.py \
        --weights ct=runs/ct/weights/best.pt \
                  pet=runs/pet/weights/best.pt \
                  fusion=runs/fusion/weights/best.pt \
        --splits val test

Run it on every slice of the val and test patients, not only the slices with
a tumor, so the viewer can show false alarms on healthy slices.

Writes results/predictions/<model>/<split>/<patient>_<slice>.txt, mirroring
the image names, with lines
"class cx cy w h conf". Boxes are saved down to a very low confidence so the
threshold can be tuned later (scripts/tune_thresholds.py) and changed in the
app without running the model again.

Needs the ultralytics package. The viewer itself does not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
import config  # noqa: E402
from viewer_core import IMAGE_EXTS, Box, write_yolo_file  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="Project folder with data/ and results/")
    ap.add_argument("--weights", nargs="+", required=True, metavar="MODEL=PATH",
                    help=f"One per model. Models: {', '.join(config.MODELS)}")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--min-conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.5, help="Same as in the presentation")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    ap.add_argument("--chunk", type=int, default=256, help="Images per predict call")
    args = ap.parse_args()

    from ultralytics import YOLO

    paths = config.Paths(Path(args.root).resolve())
    for spec in args.weights:
        name, _, weights = spec.partition("=")
        if name not in config.MODELS or not weights:
            sys.exit(f"Bad --weights entry {spec!r}. Use MODEL=PATH with MODEL in {list(config.MODELS)}.")
        modality = config.MODELS[name]["modality"]
        model = YOLO(weights)

        for split in args.splits:
            split_dir = paths.images(modality, split)
            files = sorted(f for f in split_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS) \
                if split_dir.is_dir() else []
            if not files:
                print(f"[{name}] skip {split}: no images in {split_dir}")
                continue
            out_dir = paths.predictions(name, split)
            for i in range(0, len(files), args.chunk):
                chunk = [str(f) for f in files[i:i + args.chunk]]
                for r in model.predict(source=chunk, conf=args.min_conf, iou=args.nms_iou,
                                       imgsz=args.imgsz, device=args.device, stream=True, verbose=False):
                    boxes = []
                    if r.boxes is not None and len(r.boxes):
                        xywhn = r.boxes.xywhn.cpu().numpy()
                        conf = r.boxes.conf.cpu().numpy()
                        cls = r.boxes.cls.cpu().numpy().astype(int)
                        boxes = [Box(*map(float, b), conf=float(c), cls=int(k))
                                 for b, c, k in zip(xywhn, conf, cls)]
                    rel = Path(r.path).resolve().relative_to(split_dir.resolve()).with_suffix(".txt")
                    write_yolo_file(out_dir / rel, boxes)
            print(f"[{name}] {split}: {len(files)} slices -> {out_dir}")

if __name__ == "__main__":
    main()
