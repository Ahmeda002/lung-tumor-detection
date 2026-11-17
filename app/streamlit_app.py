"""Lung tumor detection: Patient Viewer Application.

Run from the repository root:

    streamlit run app/streamlit_app.py                 # uses ./yolo_data and ./results
    streamlit run app/streamlit_app.py -- --root demo  # uses the demo data

Layout follows the mockup in the presentation: patient, slice and model in the
sidebar; the CT, PET and fused images of one slice side by side with the
expert's and the model's boxes; the model comparison underneath. Everything
fits on one screen.

The app only reads saved slices, labels, predictions and result tables.
It never trains or runs a model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
from viewer_core import (  # noqa: E402
    Dataset,
    draw_overlay,
    filter_conf,
    match_boxes,
    patient_track,
    to_display,
)

st.set_page_config(page_title="Lung Tumor Detection Viewer", page_icon="🫁", layout="wide")

# Fit everything on one screen: less padding, images sized to the window height.
st.markdown(
    """
    <style>
    .block-container {padding-top: 4rem; padding-bottom: 0.5rem; max-width: 100%;}
    [data-testid="stSidebarContent"] {padding-top: 0.5rem;}
    [data-testid="stSidebarUserContent"] {padding-top: 1rem;}
    [data-testid="stVerticalBlock"] {gap: 0.55rem;}
    [data-testid="stImage"] img {max-height: 38vh; width: auto !important; margin: 0 auto; display: block;}
    [data-testid="stImageCaption"] {margin-top: 0;}
    h5 {padding: 0.2rem 0 0.5rem 0 !important; margin: 0 !important; line-height: 1.3 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Images are looked up in the fused folder first; any modality works for the
# patient and slice lists since all three share file names.
LIST_MODALITY = next(
    (m for m in ("fusion", "ct", "pet") if cached_patients(str(ROOT), SPLIT, m)), "fusion"
)
patients = cached_patients(str(ROOT), SPLIT, LIST_MODALITY)

if not patients:
    st.title("Lung Tumor Detection Viewer")
    st.error(
        f"No test slices found under `{ROOT / DS.p.data_dir}`.\n\n"
        "Point the app at your project folder with `-- --root <folder>`, or try it with "
        "demo data:\n\n```\npython scripts/make_demo_data.py --out demo\n"
        "streamlit run app/streamlit_app.py -- --root demo\n```"
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Sidebar: test patient, slice, model (as in the mockup)
# --------------------------------------------------------------------------- #
def reset_slice():
    st.session_state.slice_no = 1


st.sidebar.markdown("### 🫁 Patient viewer")
if DS.is_demo():
    st.sidebar.warning("**Demo data:** synthetic scans and scores, not results.")

patient = st.sidebar.selectbox(f"Test patient ({len(patients)})", patients, on_change=reset_slice)
stems = cached_slices(str(ROOT), SPLIT, LIST_MODALITY, patient)
n = len(stems)

seg = getattr(st.sidebar, "segmented_control", None)
if seg is not None:
    model = seg("Model", MODEL_KEYS, format_func=model_label, default="fusion",
                key="model", selection_mode="single") or "fusion"
else:  # older Streamlit
    model = st.sidebar.radio("Model", MODEL_KEYS, format_func=model_label, index=2, key="model")

if "slice_no" not in st.session_state or not 1 <= st.session_state.slice_no <= n:
    st.session_state.slice_no = 1

has_gt, top_conf = cached_track(str(ROOT), model, SPLIT, LIST_MODALITY, tuple(stems))
tumor_slices = [i + 1 for i, v in enumerate(has_gt) if v]


def step(delta: int):
    st.session_state.slice_no = min(max(st.session_state.slice_no + delta, 1), n)


def jump_tumor(direction: int):
    cur = st.session_state.slice_no
    cands = [s for s in tumor_slices if (s > cur if direction > 0 else s < cur)]
    if cands:
        st.session_state.slice_no = cands[0] if direction > 0 else cands[-1]


st.sidebar.slider(f"Slice (of {n})", 1, max(n, 1), key="slice_no")
b = st.sidebar.columns(2)
b[0].button("◀ Prev", on_click=step, args=(-1,), width="stretch")
b[1].button("Next ▶", on_click=step, args=(1,), width="stretch")
b[0].button("◀ Tumor", on_click=jump_tumor, args=(-1,), width="stretch",
            disabled=not tumor_slices, help="Previous slice with an expert box")
b[1].button("Tumor ▶", on_click=jump_tumor, args=(1,), width="stretch",
            disabled=not tumor_slices, help="Next slice with an expert box")

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

st.sidebar.caption(
    f"Expert box: green, dashed. Model box: solid red if it matches an expert box "
    f"(IoU ≥ {config.MATCH_IOU}), orange if not."
)
st.sidebar.caption(f"Data: `{ROOT}`")


# --------------------------------------------------------------------------- #
# Top: the slice in CT, PET and fused, with this model's boxes on all three
# --------------------------------------------------------------------------- #
idx = st.session_state.slice_no - 1
stem = stems[idx]
gts = DS.ground_truth(SPLIT, LIST_MODALITY, stem)
have_preds = DS.has_predictions(model, SPLIT)
preds = filter_conf(DS.predictions(model, SPLIT, stem), threshold) if have_preds else []
res = match_boxes(preds, gts, config.MATCH_IOU)

summary = (f"**{patient}** · slice {idx + 1} of {n} · **{model_label(model)}** model "
           f"at threshold {threshold:.2f} · expert boxes {len(gts)} · found {len(res.tp)} · "
           f"missed {len(res.fn)} · false alarms {len(res.fp)}")
if res.tp:
    summary += " · " + ", ".join(f"conf {p.conf:.2f}, IoU {v:.2f}" for p, _, v in res.tp)
st.markdown(summary)
if not have_preds:
    st.warning(f"No saved predictions for {model_label(model)} in "
               f"`{DS.p.predictions(model, SPLIT).relative_to(ROOT)}`. Showing expert boxes only.")

cols = st.columns(3)
for col, view in zip(cols, config.VIEWS):
    with col:
        img = DS.load_image(SPLIT, view, stem)
        if img is None:
            st.info(f"No {config.VIEWS[view]} image for this slice.")
            continue
        shown = draw_overlay(to_display(img, config.DISPLAY.get(view)), gts, preds,
                             config.MATCH_IOU, show_gt, show_pred)
        st.image(shown, caption=config.VIEWS[view], width="stretch")

# Whole-scan strip: where the expert marked tumor vs. the model's confidence
strip = pd.DataFrame({"slice": range(1, n + 1), "expert": has_gt.astype(float), "conf": top_conf})
base = alt.Chart(strip).encode(x=alt.X("slice:Q", title=None, scale=alt.Scale(domain=[1, n], nice=False)))
chart = (
    base.mark_bar(color="#2ECC40", opacity=0.3).encode(
        y=alt.Y("expert:Q", axis=None, scale=alt.Scale(domain=[0, 1])), tooltip=["slice"])
    + base.mark_line(color="#FF4136").encode(
        y=alt.Y("conf:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(values=[0, 0.5, 1])),
        tooltip=["slice", alt.Tooltip("conf:Q", title="top confidence", format=".2f")])
    + alt.Chart(pd.DataFrame({"t": [threshold]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="t:Q")
    + alt.Chart(pd.DataFrame({"s": [idx + 1]})).mark_rule(color="#888", strokeWidth=2).encode(x="s:Q")
).properties(height=STRIP_HEIGHT)
st.altair_chart(chart, use_container_width=True)
st.caption("Whole scan · green bands: slices with an expert box · red line: the model's highest "
           "confidence on each slice · dashed: threshold · gray line: current slice")


# --------------------------------------------------------------------------- #
# Bottom: model comparison, tumor size, robustness (read from saved files)
# --------------------------------------------------------------------------- #
def missing(name: str, columns: str):
    st.caption(f"Write `results/{name}` with columns `{columns}` to fill this panel.")


def bars_with_ci(df: pd.DataFrame, x: str) -> alt.Chart:
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


legend = " · ".join(
    f"<span style='color:{config.MODEL_COLORS[m]}'>■</span> {model_label(m)}" for m in MODEL_KEYS)

p1, p2, p3 = st.columns([1.4, 1, 1])

with p1:
    st.markdown("##### Model comparison · test patients")
    metrics = read_csv(str(DS.p.results / "metrics.csv"))
    if metrics is None:
        missing("metrics.csv", "model, metric, value, ci_low, ci_high")
    else:
        st.altair_chart(bars_with_ci(metrics, "metric"), use_container_width=True)
        st.markdown(f"<small>{legend} · whiskers: 95% bootstrap CI</small>", unsafe_allow_html=True)

with p2:
    by_size = read_csv(str(DS.p.results / "metrics_by_size.csv"))
    if by_size is None:
        st.markdown("##### By tumor size")
        missing("metrics_by_size.csv", "model, size_bin, metric, value, ci_low, ci_high")
    else:
        opts = list(dict.fromkeys(by_size["metric"]))
        h, s = st.columns([1.3, 1])
        h.markdown("##### By tumor size")
        metric = s.selectbox("Size metric", opts, index=opts.index("recall") if "recall" in opts else 0,
                             label_visibility="collapsed")
        st.altair_chart(bars_with_ci(by_size[by_size["metric"] == metric], "size_bin"),
                        use_container_width=True)

with p3:
    rob = read_csv(str(DS.p.results / "robustness.csv"))
    if rob is None:
        st.markdown("##### Robustness")
        missing("robustness.csv", "model, perturbation, level, metric, value")
    else:
        tests = list(dict.fromkeys(rob["perturbation"]))
        h, s = st.columns([1, 1.3])
        h.markdown("##### Robustness")
        test = s.selectbox("Robustness test", tests, label_visibility="collapsed")
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
        st.altair_chart(chart, use_container_width=True)
