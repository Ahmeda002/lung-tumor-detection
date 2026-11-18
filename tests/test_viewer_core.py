import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from viewer_core import (  # noqa: E402
    Box, filter_conf, iou, match_boxes, read_yolo_file, write_yolo_file, draw_overlay,
    to_display, natural_key, split_stem, hot_colormap,
)


def test_iou_identical_and_disjoint():
    a = Box(0.5, 0.5, 0.2, 0.2)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, Box(0.1, 0.1, 0.05, 0.05)) == 0.0


def test_iou_half_overlap():
    a = Box(0.5, 0.5, 0.2, 0.2)
    b = Box(0.6, 0.5, 0.2, 0.2)  # shifted by half a width
    assert iou(a, b) == pytest.approx(0.02 / 0.06)


def test_match_prefers_high_confidence():
    gt = [Box(0.5, 0.5, 0.2, 0.2)]
    low = Box(0.5, 0.5, 0.2, 0.2, conf=0.3)
    high = Box(0.51, 0.5, 0.2, 0.2, conf=0.9)
    m = match_boxes([low, high], gt)
    assert len(m.tp) == 1 and m.tp[0][0] is high
    assert m.fp == [low] and m.fn == []


def test_match_counts_misses():
    gt = [Box(0.2, 0.2, 0.1, 0.1), Box(0.8, 0.8, 0.1, 0.1)]
    m = match_boxes([Box(0.2, 0.2, 0.1, 0.1, conf=0.5)], gt)
    assert len(m.tp) == 1 and len(m.fn) == 1


def test_filter_conf():
    boxes = [Box(0.5, 0.5, 0.1, 0.1, conf=c) for c in (0.1, 0.5, 0.9)]
    assert [b.conf for b in filter_conf(boxes, 0.5)] == [0.5, 0.9]


def test_yolo_roundtrip(tmp_path):
    f = tmp_path / "a" / "1.txt"
    boxes = [Box(0.5, 0.4, 0.1, 0.2), Box(0.3, 0.3, 0.05, 0.05, conf=0.77)]
    write_yolo_file(f, boxes)
    back = read_yolo_file(f)
    assert len(back) == 2 and back[0].conf is None and back[1].conf == pytest.approx(0.77)
    assert read_yolo_file(tmp_path / "missing.txt") == []
    write_yolo_file(f, [])
    assert read_yolo_file(f) == []


def test_bad_label_line(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("0 0.5 0.5\n")
    with pytest.raises(ValueError):
        read_yolo_file(f)


def test_natural_order():
    assert sorted(["10", "2", "1"], key=natural_key) == ["1", "2", "10"]


def test_overlay_draws_colors():
    img = Image.new("RGB", (640, 640), (0, 0, 0))
    out = draw_overlay(img, [Box(0.5, 0.5, 0.2, 0.2)], [Box(0.25, 0.25, 0.1, 0.1, conf=0.8)])
    arr = np.asarray(out)
    assert (arr == (46, 204, 64)).all(axis=-1).any()   # green expert box
    assert (arr == (255, 165, 0)).all(axis=-1).any()   # unmatched prediction is orange


def test_to_display_modes():
    arr = np.zeros((4, 4, 3), np.uint8)
    arr[..., 0] = 100   # CT
    arr[0, 0, 2] = 64   # one PET pixel with moderate uptake (about SUV 3.8)
    img = Image.fromarray(arr)
    gray = np.asarray(to_display(img, {"mode": "gray", "channel": 0}))
    assert (gray == 100).all()
    fused = np.asarray(to_display(img, {"mode": "overlay", "ct_channel": 0, "pet_channel": 2, "alpha": 0.5}))
    assert (fused[1:, 1:] == 100).all()                 # no uptake: plain CT
    assert fused[0, 0, 0] > fused[0, 0, 2]              # uptake: tinted warm
    assert np.asarray(to_display(img, None)).tolist() == arr.tolist()


def test_hot_colormap_ends():
    assert hot_colormap(np.array([0.0])).tolist() == [[0, 0, 0]]
    assert hot_colormap(np.array([1.0])).tolist() == [[1, 1, 1]]


def test_split_stem():
    assert split_stem("Lung_Dx-A0164_042") == ("Lung_Dx-A0164", "042")
    assert split_stem("Lung_Dx-A0164/042") == ("Lung_Dx-A0164", "042")


def test_flat_and_folder_layouts(tmp_path):
    import config
    from viewer_core import Dataset
    paths = config.Paths(tmp_path)
    flat = paths.images("fusion", "test")
    flat.mkdir(parents=True)
    for z in (10, 2, 1):
        Image.new("RGB", (8, 8)).save(flat / f"Lung_Dx-A0164_{z:03d}.png")
    (flat / "Lung_Dx-B0001").mkdir()
    Image.new("RGB", (8, 8)).save(flat / "Lung_Dx-B0001" / "5.png")
    write_yolo_file(paths.labels("fusion", "test") / "Lung_Dx-A0164_002.txt", [Box(0.5, 0.5, 0.1, 0.1)])
    ds = Dataset(paths)
    assert ds.patients("test", "fusion") == ["Lung_Dx-A0164", "Lung_Dx-B0001"]
    assert ds.slices("test", "fusion", "Lung_Dx-A0164") == [
        "Lung_Dx-A0164_001", "Lung_Dx-A0164_002", "Lung_Dx-A0164_010"]
    assert ds.slices("test", "fusion", "Lung_Dx-B0001") == ["Lung_Dx-B0001/5"]
    assert len(ds.ground_truth("test", "fusion", "Lung_Dx-A0164_002")) == 1
    # CT has no label folder here, so it falls back to the fusion labels
    assert len(ds.ground_truth("test", "ct", "Lung_Dx-A0164_002")) == 1


def test_demo_data_and_threshold_tuning(tmp_path, monkeypatch):
    import make_demo_data
    import tune_thresholds
    import config
    from viewer_core import Dataset

    monkeypatch.setattr(sys, "argv", ["x", "--out", str(tmp_path), "--patients", "2", "--slices", "30"])
    make_demo_data.main()
    ds = Dataset(config.Paths(tmp_path))
    assert ds.is_demo()
    pats = ds.patients("test", "fusion")
    assert len(pats) == 2
    stems = ds.slices("test", "fusion", pats[0])
    assert len(stems) == 30
    assert stems[0] == f"{pats[0]}_000"
    assert any(ds.ground_truth("test", "fusion", s) for s in stems)
    assert ds.has_predictions("fusion", "test")
    assert set(ds.thresholds()) == {"ct", "pet", "fusion"}

    # Tuning on the demo test split exercises the full path
    ts = np.round(np.arange(0.05, 0.951, 0.05), 2)
    t, f1 = tune_thresholds.best_threshold(*tune_thresholds.counts_at(ds, "fusion", "test", ts), ts)
    assert 0.05 <= t <= 0.95 and 0 < f1 <= 1


def test_low_uptake_stays_gray():
    arr = np.zeros((2, 2, 3), np.uint8)
    arr[..., 0] = 120
    arr[..., 2] = 15  # about SUV 0.9, normal background
    out = np.asarray(to_display(Image.fromarray(arr), {"mode": "overlay", "ct_channel": 0, "pet_channel": 2}))
    assert (out == 120).all()


def test_expert_box_is_dashed():
    img = Image.new("RGB", (200, 200), (0, 0, 0))
    out = np.asarray(draw_overlay(img, [Box(0.5, 0.5, 0.5, 0.5)], []))
    top_edge = (out[50, 50:150] == (46, 204, 64)).all(axis=-1)
    assert top_edge.any() and not top_edge.all()   # green with gaps
