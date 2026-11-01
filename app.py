"""NeuroScan AI - Brain tumor MRI classification (Streamlit)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.config import METRICS_PATH, MODEL_PATH, SAMPLE_DIR
from src.predict import predict_bytes

st.set_page_config(page_title="NeuroScan AI | Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

ACCENT, GOOD, WARN, BAD = "#22d3ee", "#34d399", "#fbbf24", "#f87171"
CSS = """
<style>
.block-container{padding-top:1.6rem;max-width:1250px}
.hero{padding:1.6rem 2rem;border-radius:18px;margin-bottom:1.2rem;
 background:linear-gradient(120deg,#0e7490 0%,#312e81 55%,#581c87 100%);box-shadow:0 10px 30px rgba(0,0,0,.35)}
.hero h1{margin:0;font-size:2.1rem;color:#fff}.hero p{margin:.3rem 0 0;color:#dbeafe;font-size:1.02rem}
.card{background:#131a2e;border:1px solid #243049;border-radius:16px;padding:1.1rem 1.3rem;height:100%}
.kpi{background:linear-gradient(160deg,#131a2e,#0f172a);border:1px solid #243049;border-radius:16px;padding:1rem 1.2rem}
.kpi .v{font-size:1.9rem;font-weight:700;color:#22d3ee;line-height:1.1}.kpi .l{color:#94a3b8;font-size:.82rem;letter-spacing:.06em;text-transform:uppercase}
.result{border-radius:16px;padding:1.2rem 1.4rem;border:1px solid #243049;background:#131a2e}
.result .name{font-size:1.9rem;font-weight:700}.result .sub{color:#94a3b8}
.pill{display:inline-block;padding:.15rem .7rem;border-radius:99px;font-size:.78rem;font-weight:600;margin-bottom:.4rem}
.disc{font-size:.8rem;color:#94a3b8;border-left:3px solid #fbbf24;padding:.4rem .8rem;background:#1a1f33;border-radius:6px}
footer,#MainMenu{visibility:hidden}
</style>"""
st.markdown(CSS, unsafe_allow_html=True)


def style(fig, h=320):
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=h, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div></div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    from src.model import load_trained   # imported lazily so TensorFlow loads once
    return load_trained()


@st.cache_data
def get_metrics():
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None


def is_negative(label: str) -> bool:
    return label.lower().startswith("no ")


# ---------------------------------------------------------------- header
st.markdown('<div class="hero"><h1>🧠 NeuroScan AI</h1><p>Deep-learning assistant for brain MRI tumor classification '
            '— upload a scan, get class probabilities, and inspect genuine model performance.</p></div>', unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error("Trained model not found. Run `python train.py` first (see README).")
    st.stop()
model, class_names = get_model()
metrics = get_metrics()

with st.sidebar:
    st.markdown("### 🧠 NeuroScan AI")
    st.caption("B.Tech Minor Project · CNN · TensorFlow/Keras")
    st.markdown("**Classes**")
    for c in class_names:
        st.markdown(f"- {c}")
    if metrics:
        st.metric("Held-out test accuracy", f"{metrics['test']['accuracy']*100:.1f}%",
                  help=f"Measured on {metrics['test']['n']} images never used in training.")
    st.markdown('<div class="disc">⚠️ Educational demo only. Not a medical device — never use it for diagnosis or treatment decisions.</div>',
                unsafe_allow_html=True)

tab_pred, tab_perf, tab_model, tab_about = st.tabs(["🔍 Diagnosis", "📊 Performance Dashboard", "🧩 Model Info", "ℹ️ About"])

# ---------------------------------------------------------------- prediction
with tab_pred:
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.subheader("1 · Provide an MRI scan")
        up = st.file_uploader("Upload MRI image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
                              help="Single 2-D brain MRI slice. Max 10 MB.")
        samples = sorted(SAMPLE_DIR.glob("*")) if SAMPLE_DIR.exists() else []
        sample = st.selectbox("…or try a held-out test sample", ["—"] + [p.name for p in samples],
                              help="Images from the test split that the model never saw during training.")
        data, source = None, None
        if up is not None:
            data, source = up.getvalue(), up.name
        elif sample != "—":
            data, source = (SAMPLE_DIR / sample).read_bytes(), sample
            st.caption(f"Sample file name suggests ground truth: **{sample.split('_')[0]}**")
        if data:
            try:
                st.image(Image.open(__import__("io").BytesIO(data)), caption=source, width="stretch")
            except Exception:
                st.error("This file could not be opened as an image.")
                data = None

    with right:
        st.subheader("2 · Prediction")
        if not data:
            st.info("Upload an MRI image or pick a sample to see the prediction.")
        else:
            try:
                with st.spinner("Analysing scan…"):
                    res = predict_bytes(model, class_names, data)
            except ValueError as e:
                st.error(str(e)); st.stop()
            conf, label = res["confidence"], res["label"]
            color = GOOD if is_negative(label) else BAD
            level = ("High", GOOD) if conf >= 0.85 else ("Moderate", WARN) if conf >= 0.65 else ("Low", BAD)
            st.markdown(f'<div class="result"><span class="pill" style="background:{level[1]}22;color:{level[1]}">{level[0]} confidence</span>'
                        f'<div class="name" style="color:{color}">{label}</div>'
                        f'<div class="sub">Model confidence: <b>{conf*100:.1f}%</b></div></div>', unsafe_allow_html=True)
            if conf < 0.65:
                st.warning("Low confidence — the model is unsure about this image. Treat the result as unreliable.")
            c1, c2 = st.columns(2)
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=conf * 100, number={"suffix": "%"},
                                           gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
                                                  "bgcolor": "#1e293b"}, title={"text": "Confidence"}))
            c1.plotly_chart(style(gauge, 260), width="stretch")
            c2.image(res["model_input"], caption="Preprocessed model input (128×128)", width="stretch")
            probs = res["probabilities"]
            order = sorted(probs, key=probs.get)
            bar = go.Figure(go.Bar(x=[probs[k] * 100 for k in order], y=order, orientation="h",
                                   marker_color=[color if k == label else "#475569" for k in order],
                                   text=[f"{probs[k]*100:.1f}%" for k in order], textposition="auto"))
            bar.update_layout(title="Class probabilities", xaxis=dict(range=[0, 100], title="%"))
            st.plotly_chart(style(bar, 240), width="stretch")

# ---------------------------------------------------------------- performance
with tab_perf:
    if not metrics:
        st.warning("No evaluation report found. Run `python train.py` to generate `reports/metrics.json`.")
    else:
        t, ds = metrics["test"], metrics["dataset"]
        st.subheader("Held-out test-set results")
        k = st.columns(5)
        kpi(k[0], "Accuracy", f"{t['accuracy']*100:.1f}%")
        kpi(k[1], "Precision (macro)", f"{t['precision_macro']*100:.1f}%")
        kpi(k[2], "Recall (macro)", f"{t['recall_macro']*100:.1f}%")
        kpi(k[3], "F1-score (macro)", f"{t['f1_macro']*100:.1f}%")
        auc = (t.get("roc") or {}).get("auc", (t.get("roc") or {}).get("auc_ovr_macro"))
        kpi(k[4], "ROC-AUC", f"{auc:.3f}" if auc is not None else "—")
        lo, hi = t["accuracy_ci95"]
        st.caption(f"Computed on **{t['n']}** unseen images. 95% confidence interval for accuracy: "
                   f"**{lo*100:.0f}% – {hi*100:.0f}%** (small test set → wide uncertainty). "
                   f"Duplicate images ({ds['duplicates_removed']}) were removed before splitting to prevent leakage.")

        c1, c2 = st.columns(2, gap="large")
        cm = np.array(t["confusion_matrix"])
        fig = go.Figure(go.Heatmap(z=cm, x=metrics["classes"], y=metrics["classes"], colorscale="Tealgrn",
                                   text=cm, texttemplate="%{text}", showscale=False))
        fig.update_layout(title="Confusion matrix", xaxis_title="Predicted", yaxis_title="Actual", yaxis_autorange="reversed")
        c1.plotly_chart(style(fig, 380), width="stretch")

        rep = t["classification_report"]
        rows = [{"Class": c, "Precision": rep[c]["precision"], "Recall": rep[c]["recall"], "F1-score": rep[c]["f1-score"],
                 "Support": int(rep[c]["support"])} for c in metrics["classes"]]
        rows.append({"Class": "Macro avg", "Precision": rep["macro avg"]["precision"], "Recall": rep["macro avg"]["recall"],
                     "F1-score": rep["macro avg"]["f1-score"], "Support": int(rep["macro avg"]["support"])})
        c2.markdown("**Classification report**")
        c2.dataframe(pd.DataFrame(rows).style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1-score": "{:.3f}"}),
                     hide_index=True, width="stretch")
        roc = t.get("roc") or {}
        if "fpr" in roc:
            rf = go.Figure([go.Scatter(x=roc["fpr"], y=roc["tpr"], name=f"AUC = {roc['auc']:.3f}", line=dict(color=ACCENT, width=3)),
                            go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash", color="#64748b"), showlegend=False)])
            rf.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
            c2.plotly_chart(style(rf, 250), width="stretch")

        st.subheader("Training history")
        h = metrics["history"]
        c3, c4 = st.columns(2)
        for col, keys, title in ((c3, ("accuracy", "val_accuracy"), "Accuracy"), (c4, ("loss", "val_loss"), "Loss")):
            f = go.Figure([go.Scatter(y=h[kk], name=kk, mode="lines") for kk in keys]); f.update_layout(title=title, xaxis_title="Epoch")
            col.plotly_chart(style(f, 300), width="stretch")

        st.subheader("Dataset")
        d1, d2 = st.columns(2)
        cc = ds["class_counts"]
        f = go.Figure(go.Bar(x=list(cc.keys()), y=list(cc.values()), marker_color=ACCENT)); f.update_layout(title="Unique images per class")
        d1.plotly_chart(style(f, 280), width="stretch")
        d2.dataframe(pd.DataFrame({"Split": ["Train", "Validation", "Test"],
                                   "Images": [ds["split_sizes"][s] for s in ("train", "val", "test")]}),
                     hide_index=True, width="stretch")

# ---------------------------------------------------------------- model info
with tab_model:
    if not metrics:
        st.warning("Run `python train.py` to generate model information.")
    else:
        tr, env = metrics["training"], metrics["environment"]
        k = st.columns(4)
        kpi(k[0], "Parameters", f"{tr['total_params']:,}")
        kpi(k[1], "Input size", f"{tr['input_size'][0]}×{tr['input_size'][1]}×3")
        kpi(k[2], "Classes", str(len(metrics["classes"])))
        kpi(k[3], "Best epoch", f"{tr['best_epoch']} / {tr['epochs_run']}")
        a, b = st.columns([1.3, 1], gap="large")
        a.markdown("**Architecture**")
        a.dataframe(pd.DataFrame(metrics["architecture"]).rename(columns={"layer": "Layer", "type": "Type",
                    "output_shape": "Output shape", "params": "Params"}), hide_index=True, width="stretch")
        b.markdown("**Training configuration**")
        b.json({k_: v for k_, v in tr.items() if k_ not in ("input_size",)}, expanded=False)
        b.markdown("**Environment**")
        b.json(env, expanded=False)
        b.caption(f"Report generated {metrics['generated_utc']} UTC")
        if len(metrics["classes"]) == 2:
            st.info("This model is **binary** (Tumor / No Tumor) because the supplied dataset only has those two labels. "
                    "Tumor-type classification (Glioma / Meningioma / Pituitary) needs a dataset labelled with those types — "
                    "put it in `dataset/<class_name>/` and re-run `python train.py`; the app adapts automatically.")

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Pipeline** — MRI image → decode (BGR→RGB) → resize 128×128 → scale to [0,1] → CNN (3 conv blocks + dense head) → softmax probabilities.
The *same* preprocessing function is used for training and inference.

**Limitations (please read)**
- Trained on a small public dataset (a few hundred images, 2-D slices). Results will not transfer reliably to other scanners, sequences or hospitals.
- The model does not check that an upload is actually a brain MRI; unrelated images still receive a prediction.
- The test set is small, so reported metrics have wide confidence intervals.
- Not clinically validated. For education and demonstration only.
""")
