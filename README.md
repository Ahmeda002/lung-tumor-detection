# Deep Learning-Based Lung Tumor Detection Using CT, PET, and PET/CT Fusion

COMP 4500 Senior Project · Tennessee State University, Department of Computer Science

**Team:** Ahmed Abdikadir and Julian Padron
**Advisor:** Dr. Ali Sekmen

## Overview

CT shows anatomy (shape and edges) and PET shows metabolic activity, but neither scan alone gives the full picture. This project trains one YOLO detector three times — on CT only, PET only, and fused PET/CT — under identical conditions, and compares how well each finds and localizes lung tumors.

**Hypothesis:** fusion beats both single-modality models, with the biggest gains on small tumors and tumors against the chest wall. PET alone is expected to find tumors but draw looser boxes and raise false alarms on normal uptake (e.g. the heart).

## Approach

**Data:** [Lung-PET-CT-Dx](https://doi.org/10.7937/TCIA.2020.NNC2-0461) (The Cancer Imaging Archive). 355 patients in the collection; the 133 with a paired PET/CT study are used. Split by patient (about 75% train / 10% val / 15% test) so no slices leak between sets. Single class: `tumor`.

**Fusion as a 3-channel image.** Each slice is packed into the three slots a pretrained detector expects from an RGB photo, so no architecture changes are needed:

| Channel | CT only | PET only | Fusion |
|---|---|---|---|
| 1 | CT lung window (W 1500, L −600) | PET SUV | CT lung window |
| 2 | CT mediastinal window (W 350, L 40) | PET SUV | CT mediastinal window |
| 3 | CT lung window | PET SUV | PET SUV, resampled to the CT grid, clipped at 15 |

**Training recipe (shared by all three runs):** YOLO11s from COCO weights, up to 100 epochs with early stopping after 20, AdamW at 0.001, batch 16, seed 0.

**Evaluation:** IoU (hit at ≥ 0.5), precision, recall, F1, mAP50 and mAP50-95, broken down by tumor size. Robustness tests without retraining: added noise, a missing modality, and PET misaligned by 2, 5 or 10 pixels.

**App (planned):** a Streamlit viewer to step through a test patient's slices, switch models, and compare expert vs. predicted boxes and saved metrics.

## Status

- [x] Need, requirements and design (report chapters 1–3)
- [x] End-to-end prototype on one patient (raw DICOM → fused slices → YOLO boxes)
- [ ] Scale to all 133 patients with the patient-level split
- [ ] Train the CT, PET and fusion models (Spring 2027)
- [ ] Evaluation, size analysis and stress tests (Spring 2027)
- [ ] Streamlit app and final report (Spring 2027)

The prototype trains and validates on slices from the same patient, so it reports no scores yet. It is a pipeline check, not a result.

## Repository contents

| Path | What it is |
|---|---|
| `report/senior_project_report_ch1-3.pdf` | Written report, Chapters 1–3: introduction and need analysis, the 12 functional requirements, and the system design |
| `report/chapter3_latex/` | LaTeX source and figures for Chapter 3 (Design) |
| `docs/ethical_issues_report.md` | Ethical issues report: patient privacy, missed tumors vs. false alarms, over-reliance, security and tampering |
| `figures/` | Pipeline diagram (SVG and PNG) |
| `notebooks/lung_tumor_prototype.ipynb` | Colab notebook for the one-patient prototype: downloads annotations and DICOM series from TCIA/IDC, aligns PET to CT, builds the fused 3-channel slices, exports YOLO labels, and trains YOLO11n for 50 epochs |
| `notebooks/lung_tumor_yolo_demo.ipynb` | Earlier end-to-end demo notebook: windowing, PET normalization, fusion and box conversion on a small multi-patient pull from IDC |
| `presentation/index.html` | Project presentation. Open it in a browser; use ← / → or click to move, F for full screen |
| `app/` | Patient Viewer Application (Streamlit): step through a test patient's slices, switch models, compare expert and predicted boxes, and view saved metrics and robustness results |
| `scripts/` | Helpers for the app: save each model's predictions, tune confidence thresholds on validation, and make synthetic demo data |
| `tests/` | Unit tests for the app's box matching, file reading and drawing |

## Running the notebooks

Open them in Google Colab with a GPU runtime and run the cells top to bottom. They install `idc-index`, `pydicom`, `SimpleITK` and `ultralytics`, and download the raw scan volumes. The repository includes rendered examples in the notebook and presentation.

## Patient Viewer Application

The presentation’s **Patient viewer demo** slide plays a narrated walkthrough of the Streamlit app. The **Semester 1 prototype** view opens on patient Lung_Dx-A0164 and lets you jump between the detection on slice 290 and the missed tumor on slice 306. It uses the saved notebook figure shown on the **Prototype** slide. CT and PET are channel views derived from that figure. The model comparison shows labeled example scores for the planned evaluation. Training and validation slices for the prototype came from the same patient.

**Open the presentation viewer** from the repository root:

```bash
pip install -r requirements-app.txt
streamlit run app/streamlit_app.py
```

**Try it with demo data** (synthetic scans and made-up scores, clearly flagged in the app):

```bash
pip install -r requirements-app.txt
python scripts/make_demo_data.py --out demo
streamlit run app/streamlit_app.py -- --root demo
```

Choose **Synthetic dataset** from the sidebar to try the full viewer controls.

**Use it with real results.** The test dataset view lets you choose a patient, slice and model, compare saved predictions with expert boxes, and read saved metrics. It opens by default when test data is present. The app expects one Ultralytics dataset folder per modality, with the same file names as the prototype notebook (`<patient>_<slice>.png`):

```
yolo_data/{ct,pet,fusion}/images/{val,test}/Lung_Dx-A0164_042.png
yolo_data/{ct,pet,fusion}/labels/{val,test}/Lung_Dx-A0164_042.txt
```

Export **every** slice of the val and test patients, not only the slices with a tumor, so the viewer can show false alarms on healthy slices. Then:

```bash
# 1. Save each model's boxes once (needs ultralytics)
python scripts/export_predictions.py --weights ct=runs/ct/weights/best.pt \
    pet=runs/pet/weights/best.pt fusion=runs/fusion/weights/best.pt --splits val test
# 2. Pick each model's confidence threshold on the validation patients
python scripts/tune_thresholds.py
# 3. Open the viewer
streamlit run app/streamlit_app.py
```

For the comparison and robustness tabs, the evaluation code writes these tables to `results/`:

| File | Columns |
|---|---|
| `metrics.csv` | `model, metric, value, ci_low, ci_high` |
| `metrics_by_size.csv` | `model, size_bin, metric, value, ci_low, ci_high` |
| `robustness.csv` | `model, perturbation, level, metric, value` |

`model` is `ct`, `pet` or `fusion`. Folder names and display settings are in `app/config.py`. Run the tests with `pytest tests`.

## Data and citation

Li, P., Wang, S., Li, T., Lu, J., HuangFu, Y., & Wang, D. (2020). *A Large-Scale CT and PET/CT Dataset for Lung Cancer Diagnosis (Lung-PET-CT-Dx)* [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.2020.NNC2-0461
