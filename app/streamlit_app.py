"""Lung tumor detection: Patient Viewer Application.

Run from the repository root:

    streamlit run app/streamlit_app.py                 # shows the Semester 1 prototype if no test dataset is present
    streamlit run app/streamlit_app.py -- --root demo  # also enables the synthetic dataset

The layout includes a compact control sidebar, three side-by-side scan panels,
and a model comparison below the scans.

The app only reads saved slices, labels, predictions and result tables.
It never trains or runs a model.
"""

from __future__ import annotations

import argparse
import math
import sys
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import prototype  # noqa: E402
from viewer_core import (  # noqa: E402
    Dataset,
    draw_overlay,
    filter_conf,
    match_boxes,
    patient_track,
    to_display,
)

st.set_page_config(page_title="Lung Tumor Detection Viewer", layout="wide", initial_sidebar_state="auto")
stylesheet = Path(__file__).with_name("viewer.css").read_text(encoding="utf-8")
st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)

PANEL_CHART_HEIGHT = 150
STRIP_HEIGHT = 60


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
def parse_root() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Project folder with yolo_data/ and results/")
    args, _ = parser.parse_known_args()
    return Path(args.root).resolve()


ROOT = parse_root()
DS = Dataset(config.Paths(ROOT))
SPLIT = config.DEFAULT_SPLIT
MODEL_KEYS = list(config.MODELS)
PROTOTYPE_NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "lung_tumor_prototype.ipynb"


@st.cache_data(show_spinner=False)
def cached_patients(root: str, split: str, modality: str) -> list[str]:
    return DS.patients(split, modality)


@st.cache_data(show_spinner=False)
def cached_slices(root: str, split: str, modality: str, patient: str) -> list[str]:
    return DS.slices(split, modality, patient)


@st.cache_data(show_spinner=False)
def cached_track(root: str, model: str, split: str, modality: str, stems: tuple[str, ...]):
    return patient_track(DS, model, split, modality, list(stems))


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["model_label"] = df["model"].map(lambda m: config.MODELS.get(m, {}).get("label", m))
    return df


def model_label(key: str) -> str:
    return config.MODELS[key]["label"]


MODEL_COLOR_SCALE = alt.Scale(
    domain=[model_label(m) for m in MODEL_KEYS],
    range=[config.MODEL_COLORS[m] for m in MODEL_KEYS],
)

def comparison_rows(df: pd.DataFrame, selected_model: str) -> str:
    order = {key: index for index, key in enumerate(MODEL_KEYS)}
    df = df.sort_values("model", key=lambda models: models.map(order), kind="stable")
    for column in ("value", "ci_low", "ci_high"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    rows = []
    for row in df.itertuples(index=False):
        label = escape(str(row.model_label))
        score = float(row.value)
        active = ' class="selected-model"' if row.model == selected_model else ""
        score_label = f"{score:.3f}" if math.isfinite(score) else "—"
        width = min(100, max(0, score * 100)) if math.isfinite(score) else 0
        low = getattr(row, "ci_low", math.nan)
        high = getattr(row, "ci_high", math.nan)
        interval = ""
        description = ""
        if math.isfinite(low) and math.isfinite(high):
            start = min(100, max(0, low * 100))
            end = min(100, max(start, high * 100))
            interval = (f'<span class="score-interval" '
                        f'style="left:{start:.2f}%;width:{end - start:.2f}%"></span>')
            description = f"95% CI {low:.3f} to {high:.3f}"
        rows.append(
            f'<li{active}><span>{label}</span>'
            f'<span class="score-track" aria-hidden="true" title="{description}">'
            f'<span class="score-fill" style="width:{width:.2f}%"></span>{interval}</span>'
            f'<span class="score-value">{score_label} <span class="score-detail">{description}</span></span></li>'
        )
    return '<ul class="comparison-rows" role="list">' + "".join(rows) + "</ul>"

def render_scans(
    images: dict[str, Image.Image | None], legend: str, labels: dict[str, str] | None = None,
) -> None:
    for col, (view, label) in zip(st.columns([1, 1, 1.4]), (labels or config.VIEWS).items()):
        with col.container(key=f"scan-{view}"):
            img = images[view]
            if img is None:
                st.markdown("<div class='scan-frame'>No image for this slice</div>", unsafe_allow_html=True)
            else:
                st.image(img, width="stretch")
            st.caption(label)
            if view == "fusion":
                st.markdown(f"<div class='box-legend'>{legend}</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def prototype_views(path: Path, case: prototype.PrototypeCase) -> dict[str, Image.Image]:
    return prototype.scan_views(prototype.load_validation_figure(path), case)


def select_prototype_case(index: int) -> None:
    st.session_state.prototype_case = index


def render_prototype() -> None:
    case_index = st.session_state.get("prototype_case", 0)
    with st.sidebar.container(key="prototype-controls"):
        st.markdown("**Patient**")
        st.caption(prototype.PATIENT)
        st.markdown("**Validation examples**")
        for index, option in enumerate(prototype.CASES):
            st.button(option.label, key=f"prototype-{index}", width="stretch",
                      type="primary" if index == case_index else "secondary",
                      on_click=select_prototype_case, args=(index,))

    case = prototype.CASES[case_index]
    st.markdown("### Single-patient prototype")
    st.caption(f"{prototype.PATIENT} · Validation slice {case.slice_number} · PET/CT fusion")
    try:
        images = prototype_views(PROTOTYPE_NOTEBOOK, case)
    except (OSError, ValueError) as error:
        st.error(f"Could not load the saved prototype validation figure. {error}")
        return

    expert_boxes = [prototype.normalized_box(case.expert)]
    predictions = [prototype.normalized_box(case.prediction, case.confidence)] if case.prediction else []
    images["fusion"] = draw_overlay(images["fusion"], expert_boxes, predictions,
                                    line_width=4, font_size=22)
    render_scans(
        images,
        "<strong>Read the boxes on the fused scan</strong>"
        "<span><i class='expert'></i>Radiologist</span>"
        "<span><i class='matched'></i>Prototype prediction</span>",
        labels={**config.VIEWS, "ct": "CT channel", "pet": "PET channel"},
    )
    if case.confidence is not None:
        st.markdown(f"**Detection** · The prototype marked the tumor with confidence **{case.confidence:.2f}**.")
    else:
        st.markdown("**Missed tumor** · The radiologist marked a tumor. The prototype did not predict a box.")
    st.caption("CT and PET are channel views from the saved notebook figure. "
               "Training and validation slices came from the same patient.")

    with st.container(key="model-comparison"):
        st.markdown("### Model comparison")
        st.markdown("**Example scores** · Illustrative mAP50 values for the planned three-model comparison.")
        examples = pd.DataFrame({
            "model": MODEL_KEYS,
            "model_label": [model_label(key) for key in MODEL_KEYS],
            "value": [0.618, 0.662, 0.750],
        })
        st.markdown(comparison_rows(examples, "fusion"), unsafe_allow_html=True)
    with st.expander("By tumor size"):
        st.caption("A tumor-size breakdown is planned for the held-out patient evaluation.")
    with st.expander("Robustness"):
        st.caption("Noise, missing inputs, and misalignment tests are planned for the held-out patient evaluation.")


# Images are looked up in the fused folder first; any modality works for the
# patient and slice lists since all three share file names.
LIST_MODALITY = next(
    (m for m in ("fusion", "ct", "pet") if cached_patients(str(ROOT), SPLIT, m)), "fusion"
)
patients = cached_patients(str(ROOT), SPLIT, LIST_MODALITY)
is_demo = DS.is_demo()

st.sidebar.markdown("### Patient viewer")
full_viewer = "Synthetic dataset" if is_demo else "Test dataset"
viewer = "Semester 1 prototype"
if patients:
    viewer = st.sidebar.selectbox("Viewer", [viewer, full_viewer], index=0 if is_demo else 1)
if viewer == "Semester 1 prototype":
    render_prototype()
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar: test patient, slice, model (as in the mockup)
# --------------------------------------------------------------------------- #
def reset_slice():
    st.session_state.slice_no = 1


if is_demo:
    st.sidebar.markdown("<div class='demo-notice'>Demo data · synthetic scans and scores</div>",
                        unsafe_allow_html=True)

patient = st.sidebar.selectbox("Test patient", patients, on_change=reset_slice)
stems = cached_slices(str(ROOT), SPLIT, LIST_MODALITY, patient)
n = len(stems)

slice_controls = st.sidebar.container()
with st.sidebar.container(key="model-picker"):
    model = st.radio("Model", MODEL_KEYS, format_func=model_label, index=2, key="model", width="stretch")

has_gt, top_conf = cached_track(str(ROOT), model, SPLIT, LIST_MODALITY, tuple(stems))
tumor_slices = [i + 1 for i, v in enumerate(has_gt) if v]
if "slice_no" not in st.session_state or not 1 <= st.session_state.slice_no <= n:
    st.session_state.slice_no = (
        max(tumor_slices, key=lambda s: top_conf[s - 1]) if is_demo and tumor_slices else 1
    )


def step(delta: int):
    st.session_state.slice_no = min(max(st.session_state.slice_no + delta, 1), n)


def jump_tumor(direction: int):
    cur = st.session_state.slice_no
    cands = [s for s in tumor_slices if (s > cur if direction > 0 else s < cur)]
    if cands:
        st.session_state.slice_no = cands[0] if direction > 0 else cands[-1]


if n == 1:
    slice_controls.markdown("**Slice** · 1 of 1")
else:
    slice_controls.slider(f"Slice (of {n})", 1, n, key="slice_no")
b = slice_controls.columns(2)
current_slice = st.session_state.slice_no
b[0].button("◀ Prev", on_click=step, args=(-1,), width="stretch", disabled=current_slice == 1)
b[1].button("Next ▶", on_click=step, args=(1,), width="stretch", disabled=current_slice == n)
b[0].button("◀ Tumor", on_click=jump_tumor, args=(-1,), width="stretch",
            disabled=not tumor_slices or current_slice <= tumor_slices[0],
            help="Previous slice with an expert box")
b[1].button("Tumor ▶", on_click=jump_tumor, args=(1,), width="stretch",
            disabled=not tumor_slices or current_slice >= tumor_slices[-1],
            help="Next slice with an expert box")

tuned = DS.thresholds()
with st.sidebar.expander("Threshold and display"):
    use_tuned = st.checkbox(
        "Use threshold tuned on validation", value=bool(tuned), disabled=not tuned,
        help="Read from results/thresholds.json." if tuned else "results/thresholds.json not found.",
    )
    if use_tuned:
        threshold = tuned.get(model, config.DEFAULT_THRESHOLD)
    else:
        threshold = st.slider("Confidence threshold", 0.0, 1.0, config.DEFAULT_THRESHOLD, 0.01)
    show_gt = st.checkbox("Expert box", value=True)
    show_pred = st.checkbox("Model box", value=True)
    show_all_views = st.checkbox("Show boxes on CT and PET too")

    st.caption(f"Data folder: `{ROOT}`")


# --------------------------------------------------------------------------- #
# Top: the slice in CT, PET and fused, with boxes on the fused view by default
# --------------------------------------------------------------------------- #
idx = st.session_state.slice_no - 1
stem = stems[idx]
gts = DS.ground_truth(SPLIT, LIST_MODALITY, stem)
have_preds = DS.has_predictions(model, SPLIT)
preds = filter_conf(DS.predictions(model, SPLIT, stem), threshold) if have_preds else []
res = match_boxes(preds, gts, config.MATCH_IOU)

st.markdown(f"**{patient}** · Slice {idx + 1} of {n} · {model_label(model)}")
if not have_preds:
    st.warning(f"No saved predictions for {model_label(model)} in "
               f"`{DS.p.predictions(model, SPLIT).relative_to(ROOT)}`. Showing expert boxes only.")

images = {}
for view in config.VIEWS:
    img = DS.load_image(SPLIT, view, stem)
    if img is not None:
        img = to_display(img, config.DISPLAY.get(view))
        if view == "fusion" or show_all_views:
            img = draw_overlay(img, gts, preds, config.MATCH_IOU, show_gt, show_pred,
                               line_width=max(4, img.width // 90), font_size=max(16, img.width // 17))
    images[view] = img
render_scans(
    images,
    "<span><i class='expert'></i>Expert box</span>"
    f"<span><i class='matched'></i>Matched prediction (IoU ≥ {config.MATCH_IOU})</span>"
    "<span><i class='unmatched'></i>Unmatched prediction</span>",
)

summary = (f"Expert boxes {len(gts)} · Found {len(res.tp)} · Missed {len(res.fn)} · "
           f"False alarms {len(res.fp)} · Threshold {threshold:.2f}")
if res.tp:
    summary += " · " + ", ".join(f"Confidence {p.conf:.2f}, IoU {v:.2f}" for p, _, v in res.tp)
st.caption(summary)

# Whole-scan strip: where the expert marked tumor vs. the model's confidence
strip = pd.DataFrame({"slice": range(1, n + 1), "expert": has_gt.astype(float), "conf": top_conf})
base = alt.Chart(strip).encode(x=alt.X("slice:Q", title=None, scale=alt.Scale(domain=[1, n], nice=False)))
scan_track = (
    base.mark_bar(color="#2ECC40", opacity=0.3).encode(
        y=alt.Y("expert:Q", axis=None, scale=alt.Scale(domain=[0, 1])), tooltip=["slice"])
    + base.mark_line(color="#FF4136").encode(
        y=alt.Y("conf:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(values=[0, 0.5, 1])),
        tooltip=["slice", alt.Tooltip("conf:Q", title="top confidence", format=".2f")])
    + alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="t:Q")
    + alt.Chart(pd.DataFrame({"s": [idx + 1]})).mark_rule(color="#888", strokeWidth=2).encode(x="s:Q")
).properties(height=STRIP_HEIGHT)


# --------------------------------------------------------------------------- #
# Bottom: model comparison, tumor size, robustness (read from saved files)
# --------------------------------------------------------------------------- #
def missing(name: str, columns: str):
    st.caption(f"Write `results/{name}` with columns `{columns}` to fill this panel.")


def bars_with_ci(df: pd.DataFrame, x: str) -> alt.Chart | alt.LayerChart:
    enc = dict(
        x=alt.X(f"{x}:N", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
        xOffset=alt.XOffset("model_label:N", sort=[model_label(m) for m in MODEL_KEYS]),
    )
    bars = alt.Chart(df).mark_bar().encode(
        **enc,
        y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model_label:N", scale=MODEL_COLOR_SCALE, legend=None),
        opacity=alt.condition(alt.datum.model == model, alt.value(1.0), alt.value(0.45)),
        tooltip=[c for c in ("model_label", x, "value", "ci_low", "ci_high") if c in df.columns],
    )
    if {"ci_low", "ci_high"} <= set(df.columns):
        bars = bars + alt.Chart(df).mark_errorbar(ticks=True, color="#bbb").encode(
            **enc, y="ci_low:Q", y2="ci_high:Q")
    return bars.properties(height=PANEL_CHART_HEIGHT)


with st.container(key="model-comparison"):
    metrics = read_csv(str(DS.p.results / "metrics.csv"))
    if metrics is None or metrics.empty:
        st.markdown("### Model comparison")
        missing("metrics.csv", "model, metric, value, ci_low, ci_high")
    else:
        heading, picker = st.columns([3, 1], vertical_alignment="center")
        heading.markdown("### Model comparison")
        options = list(dict.fromkeys(metrics["metric"]))
        metric = picker.selectbox("Comparison metric", options,
                                  index=options.index("mAP50") if "mAP50" in options else 0,
                                  label_visibility="collapsed")
        if is_demo:
            st.markdown("**Example scores** · Synthetic values for demonstrating the viewer.")
        st.markdown(comparison_rows(metrics[metrics["metric"] == metric], model),
                    unsafe_allow_html=True)
        if {"ci_low", "ci_high"} <= set(metrics.columns):
            st.caption("Example confidence intervals are synthetic." if is_demo
                       else "Whiskers show 95% bootstrap confidence intervals.")

with st.expander("Whole-scan confidence"):
    st.altair_chart(scan_track, width="stretch")
    st.caption("Green bands mark slices with an expert box. The red line shows the model's highest "
               "confidence per slice. The dashed line marks the threshold and the gray line marks the current slice.")

legend = " · ".join(
    f"<span style='color:{config.MODEL_COLORS[m]}'>■</span> {model_label(m)}" for m in MODEL_KEYS)

with st.expander("By tumor size"):
    by_size = read_csv(str(DS.p.results / "metrics_by_size.csv"))
    if by_size is None or by_size.empty:
        missing("metrics_by_size.csv", "model, size_bin, metric, value, ci_low, ci_high")
    else:
        opts = list(dict.fromkeys(by_size["metric"]))
        metric = st.selectbox("Size metric", opts, index=opts.index("recall") if "recall" in opts else 0,
                              width=220)
        st.altair_chart(bars_with_ci(by_size[by_size["metric"] == metric], "size_bin"), width="stretch")
        st.markdown(f"<small>{legend}</small>", unsafe_allow_html=True)

with st.expander("Robustness"):
    rob = read_csv(str(DS.p.results / "robustness.csv"))
    if rob is None or rob.empty:
        missing("robustness.csv", "model, perturbation, level, metric, value")
    else:
        tests = list(dict.fromkeys(rob["perturbation"]))
        test = st.selectbox("Robustness test", tests, width=220)
        sub = rob[rob["perturbation"] == test]
        metric_name = sub["metric"].iloc[0] if len(sub) else ""
        sub = sub[sub["metric"] == metric_name]
        chart = alt.Chart(sub).mark_line(point=True).encode(
            x=alt.X("level:O", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title=metric_name, scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("model_label:N", scale=MODEL_COLOR_SCALE, legend=None),
            strokeWidth=alt.condition(alt.datum.model == model, alt.value(3.5), alt.value(1.5)),
            tooltip=["model_label", "level", alt.Tooltip("value:Q", format=".3f")],
        ).properties(height=PANEL_CHART_HEIGHT)
        st.altair_chart(chart, width="stretch")
        st.markdown(f"<small>{legend}</small>", unsafe_allow_html=True)
