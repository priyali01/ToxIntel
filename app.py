import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
import sys
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

# Ensure src module can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.prescription_pipeline import run_prescription_pipeline
from src.featurize import TARGET_COLS, smiles_to_morgan_with_info

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ToxIntel™ | Molecular Prescription System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SCIENTIFIC DESIGN SYSTEM (CSS) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background-color: #0A0C10;
        color: #C9D1D9;
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background-color: #0D1117;
        border-right: 1px solid #30363D;
    }

    .badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
        margin: 2px 4px;
    }
    .badge-known  { background: #238636; color: #fff; }
    .badge-similar { background: #9E6A03; color: #fff; }
    .badge-novel  { background: #8B949E; color: #fff; }
    .badge-toxic  { background: #F85149; color: #fff; }
    .badge-safe   { background: #3FB950; color: #fff; }
    .badge-caution { background: #D29922; color: #fff; }

    h1, h2, h3 {
        color: #F0F6FC;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .smiles-text {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #8B949E;
        background: #0D1117;
        padding: 8px;
        border-radius: 4px;
        word-break: break-all;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 700;
        color: #58A6FF;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #8B949E;
        text-transform: uppercase;
    }
    .risk-high   { color: #F85149; font-weight: 700; }
    .risk-safe   { color: #3FB950; font-weight: 700; }
    .risk-caution { color: #D29922; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# --- UTILS ---
@st.cache_resource
def load_artifact():
    artifact_path = 'models/model_artifact.pkl'
    if not os.path.exists(artifact_path): return None
    with open(artifact_path, 'rb') as f: return pickle.load(f)

def render_mol_svg(smiles, size=(400, 400)):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    d2d = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = d2d.drawOptions()
    opts.clearBackground = True
    opts.backgroundColour = (1, 1, 1, 1)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def get_mol_properties(smiles):
    """Calculate key molecular descriptors."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return {}
    return {
        'MW': round(Descriptors.MolWt(mol), 2),
        'LogP': round(Descriptors.MolLogP(mol), 2),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'TPSA': round(Descriptors.TPSA(mol), 2),
        'RotBonds': Descriptors.NumRotatableBonds(mol),
        'Atoms': mol.GetNumHeavyAtoms(),
        'Rings': Descriptors.RingCount(mol),
    }

# --- APP START ---
artifact = load_artifact()
if not artifact:
    st.error("⚠ SYSTEM ERROR: `models/model_artifact.pkl` not found. Run Phase 5 first.")
    st.stop()

thresholds = artifact.get('thresholds', {ep: 0.5 for ep in TARGET_COLS})

# --- SIDEBAR ---
with st.sidebar:
    st.title("🧪 ToxIntel™")
    st.caption("v1.2 · Research Edition")
    st.divider()

    target_smiles = st.text_input("🔬 Input SMILES", "c1ccccc1N",
                                   help="Paste any valid SMILES string.")

    st.markdown("### ⚙ Therapeutic Override")
    ignored = st.multiselect("Bypass Endpoints (e.g. for targeted therapies)",
                              TARGET_COLS, [])
    st.divider()
    run_btn = st.button("🚀 Analyze & Prescribe", type="primary",
                         use_container_width=True)

    with st.expander("ℹ About This System"):
        st.markdown("""
        **ToxIntel™** predicts toxicity across **12 Tox21 endpoints**, explains
        risk via SHAP attribution, and prescribes safer bioisosteric replacements
        using Pareto optimization.

        Uncertainty is quantified via **Mondrian Conformal Prediction** with
        90% coverage guarantees.
        """)

# --- LANDING STATE ---
if not run_btn:
    st.markdown("## 🔬 ToxIntel™ — Prediction-to-Prescription System")
    st.markdown("""
    Enter a SMILES string in the sidebar and click **Analyze & Prescribe** to begin.

    The system will:
    1. **Predict** toxicity across 12 biological endpoints
    2. **Explain** risk factors via SHAP structural attribution
    3. **Assess** prediction reliability with Mondrian Conformal Prediction
    4. **Prescribe** safer bioisosteric replacements ranked by Pareto dominance
    """)
    st.stop()

# --- PIPELINE EXECUTION ---
with st.spinner("Running 4-Step Prescription Pipeline…"):
    result = run_prescription_pipeline(target_smiles, artifact)

if result.get('error'):
    st.error(f"Pipeline Error: {result['error']}")
    st.stop()

if result.get('status') == 'ABORTED':
    st.warning(f"Pipeline Aborted: {result['reason']}")
    if 'validation' in result:
        with st.expander("View Validation Details"):
            st.json(result['validation'])
    st.stop()

if result.get('status') != 'SUCCESS':
    st.error(f"Unexpected result: {result}")
    st.stop()

# =====================================================================
# SECTION 1: MOLECULAR IDENTITY & AT-A-GLANCE METRICS
# =====================================================================
st.markdown("## 🧬 Molecular Insight & Diagnostics")

orig_probs = result['original_probs']
sugs = result.get('suggestions', [])

# --- Risk summary metrics ---
toxic_count = sum(1 for i, ep in enumerate(TARGET_COLS)
                  if orig_probs[i] > thresholds.get(ep, 0.5) and ep not in ignored)
safe_count = len(TARGET_COLS) - len(ignored) - toxic_count
max_tox_idx = int(np.argmax(orig_probs))
max_tox_ep = TARGET_COLS[max_tox_idx]
max_tox_prob = orig_probs[max_tox_idx]

# Top-level metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Endpoints Analyzed", f"{len(TARGET_COLS) - len(ignored)} / {len(TARGET_COLS)}")
m2.metric("🔴 Flagged Toxic", toxic_count)
m3.metric("🟢 Within Safety", safe_count)
m4.metric("⚠ Highest Risk", f"{max_tox_ep}", delta=f"{max_tox_prob:.3f}",
          delta_color="inverse")

st.divider()

# --- Molecule + Properties + Reliability ---
c1, c2, c3 = st.columns([1.4, 1, 1])

with c1:
    with st.container(border=True):
        st.markdown("### 🧪 Target Molecule")
        svg = render_mol_svg(target_smiles)
        if svg:
            st.write(f'<div style="display:flex;justify-content:center;background:white;'
                     f'border-radius:8px;padding:10px;">{svg}</div>',
                     unsafe_allow_html=True)
        st.markdown(f'<div class="smiles-text" style="margin-top:10px;">'
                    f'{target_smiles}</div>', unsafe_allow_html=True)

with c2:
    with st.container(border=True):
        st.markdown("### 📊 Molecular Properties")
        props = get_mol_properties(target_smiles)
        pc1, pc2 = st.columns(2)
        for i, (k, v) in enumerate(props.items()):
            (pc1 if i % 2 == 0 else pc2).metric(k, v)

with c3:
    with st.container(border=True):
        st.markdown("### 🛡 Reliability & OOD")

        # Get the Mondrian group for the original molecule itself
        mondrian = artifact['mondrian_predictor']
        fp_orig, _ = smiles_to_morgan_with_info(target_smiles)
        orig_mondrian_results = mondrian.predict_with_uncertainty(target_smiles, fp_orig)
        orig_group = orig_mondrian_results[0]['mondrian_group']
        uncertain_count = sum(1 for r in orig_mondrian_results if r['uncertain'])

        badge_class = f"badge-{orig_group}"
        st.markdown(f'<span class="badge {badge_class}">{orig_group.upper()} SPACE</span>',
                    unsafe_allow_html=True)

        if orig_group == 'novel':
            st.error("⚠ **OOD WARNING**: This molecule is structurally novel. "
                     "Predictions are exploratory — manual validation recommended.")
        elif uncertain_count > 6:
            st.warning(f"⚠ High uncertainty: {uncertain_count}/12 endpoints are uncertain.")
        else:
            st.success("✅ Predictions are within the model's training domain.")

        st.metric("Uncertain Endpoints", f"{uncertain_count} / 12")

# =====================================================================
# SECTION 2: TOXICITY RADAR + PER-ENDPOINT BREAKDOWN
# =====================================================================
st.divider()
st.markdown("## 📡 Toxicity Profile (Radar)")

col_radar, col_table = st.columns([1, 1])

with col_radar:
    fig = go.Figure()

    # Original molecule trace
    fig.add_trace(go.Scatterpolar(
        r=list(orig_probs), theta=TARGET_COLS, fill='toself',
        name='Original Molecule', line_color='#F85149',
        fillcolor='rgba(248, 81, 73, 0.25)'
    ))

    # Safety threshold trace
    threshold_vals = [thresholds.get(ep, 0.5) for ep in TARGET_COLS]
    fig.add_trace(go.Scatterpolar(
        r=threshold_vals, theta=TARGET_COLS,
        mode='lines', name='Safety Threshold',
        line=dict(color='#8B949E', dash='dash', width=2)
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363D",
                            tickfont=dict(color="#8B949E")),
            angularaxis=dict(gridcolor="#30363D", tickfont=dict(color="#C9D1D9"))
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color="#C9D1D9")),
        margin=dict(l=60, r=60, t=40, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.markdown("### Per-Endpoint Breakdown")
    rows = []
    for i, ep in enumerate(TARGET_COLS):
        prob = orig_probs[i]
        thresh = thresholds.get(ep, 0.5)
        bypassed = ep in ignored
        if bypassed:
            status = "BYPASSED"
        elif prob > thresh:
            status = "🔴 TOXIC"
        elif prob > thresh * 0.8:
            status = "🟡 CAUTION"
        else:
            status = "🟢 SAFE"
        # Uncertainty for this endpoint
        unc = orig_mondrian_results[i]['uncertain']
        rows.append({
            'Endpoint': ep,
            'Probability': round(prob, 4),
            'Threshold': round(thresh, 4),
            'Status': status,
            'Uncertain': '⚠' if unc else '✓'
        })
    ep_df = pd.DataFrame(rows)
    st.dataframe(ep_df, use_container_width=True, height=450)

# =====================================================================
# SECTION 3: BIOISOSTERE PRESCRIPTION ENGINE
# =====================================================================
st.divider()
st.markdown("## 💊 Bioisostere Prescription Engine")

if not sugs:
    st.info("No viable bioisosteric replacements found that pass ADME and Pareto filters. "
            "Consider relaxing the Therapeutic Override constraints.")
else:
    df = pd.DataFrame(sugs)

    # Pareto summary
    dominates = df[df['pareto_status'] == 'Dominates']
    tradeoffs = df[df['pareto_status'] == 'Trade-off']
    dominated = df[~df['pareto_status'].isin(['Dominates', 'Trade-off'])]

    s1, s2, s3 = st.columns(3)
    s1.metric("✅ Pareto Dominant", len(dominates))
    s2.metric("⚖ Trade-offs", len(tradeoffs))
    s3.metric("❌ Dominated", len(dominated))

    st.dataframe(
        df[['smiles', 'replacement', 'pareto_status', 'improved', 'worsened',
            'sa_score', 'delta_logp', 'ood_warning', 'mondrian_group']].style.map(
            lambda x: 'color: #3FB950; font-weight:bold' if x == 'Dominates'
            else ('color: #D29922' if x == 'Trade-off' else ''),
            subset=['pareto_status']
        ),
        use_container_width=True, height=200
    )

    # =====================================================================
    # SECTION 4: SIDE-BY-SIDE COMPARISON VIEW
    # =====================================================================
    st.divider()
    st.markdown("## 🔁 Comparison View — Original vs. Candidate")

    selected = st.selectbox("Select a candidate to compare:", df['smiles'].tolist())
    cand_data = df[df['smiles'] == selected].iloc[0]

    # Get candidate predictions
    fp_cand, _ = smiles_to_morgan_with_info(selected)
    cand_mondrian = mondrian.predict_with_uncertainty(selected, fp_cand)
    cand_probs = [r['prob'] for r in cand_mondrian]
    cand_group = cand_mondrian[0]['mondrian_group']

    comp1, comp2 = st.columns(2)

    with comp1:
        with st.container(border=True):
            st.markdown("#### Original Molecule")
            svg1 = render_mol_svg(target_smiles, size=(300, 300))
            if svg1:
                st.write(f'<div style="display:flex;justify-content:center;background:white;'
                         f'border-radius:8px;padding:8px;">{svg1}</div>',
                         unsafe_allow_html=True)
            st.markdown(f'<div class="smiles-text">{target_smiles}</div>',
                        unsafe_allow_html=True)

    with comp2:
        with st.container(border=True):
            st.markdown("#### Candidate Molecule")
            svg2 = render_mol_svg(selected, size=(300, 300))
            if svg2:
                st.write(f'<div style="display:flex;justify-content:center;background:white;'
                         f'border-radius:8px;padding:8px;">{svg2}</div>',
                         unsafe_allow_html=True)
            st.markdown(f'<div class="smiles-text">{selected}</div>',
                        unsafe_allow_html=True)

            if cand_data['ood_warning']:
                st.error("⚠ OOD WARNING: This candidate is outside the training domain.")

    # Overlaid Radar
    st.markdown("### Toxicity Comparison (Radar Overlay)")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatterpolar(
        r=list(orig_probs), theta=TARGET_COLS, fill='toself',
        name='Original', line_color='#F85149',
        fillcolor='rgba(248,81,73,0.2)'
    ))
    fig2.add_trace(go.Scatterpolar(
        r=cand_probs, theta=TARGET_COLS, fill='toself',
        name='Candidate', line_color='#3FB950',
        fillcolor='rgba(63,185,80,0.2)'
    ))
    fig2.add_trace(go.Scatterpolar(
        r=threshold_vals, theta=TARGET_COLS,
        mode='lines', name='Safety Threshold',
        line=dict(color='#8B949E', dash='dash', width=2)
    ))
    fig2.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#30363D"),
            angularaxis=dict(gridcolor="#30363D", tickfont=dict(color="#C9D1D9"))
        ),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(font=dict(color="#C9D1D9")),
        margin=dict(l=60, r=60, t=40, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Delta table
    st.markdown("### Per-Endpoint Improvement")
    delta_rows = []
    for i, ep in enumerate(TARGET_COLS):
        o = orig_probs[i]
        c = cand_probs[i]
        delta = c - o
        if delta < -0.01:
            verdict = "✅ Improved"
        elif delta > 0.01:
            verdict = "❌ Worsened"
        else:
            verdict = "➖ Unchanged"
        delta_rows.append({
            'Endpoint': ep,
            'Original': round(o, 4),
            'Candidate': round(c, 4),
            'Δ Probability': round(delta, 4),
            'Verdict': verdict
        })
    delta_df = pd.DataFrame(delta_rows)
    st.dataframe(delta_df, use_container_width=True)

    # Net Safety Gain
    net_gain = float(np.mean(orig_probs) - np.mean(cand_probs))
    net_pct = net_gain / max(np.mean(orig_probs), 1e-6) * 100

    g1, g2, g3 = st.columns(3)
    g1.metric("Net Safety Gain", f"{net_gain:+.4f}",
              help="Avg. reduction in toxicity probability across all endpoints.")
    g2.metric("Relative Improvement", f"{net_pct:+.1f}%")
    g3.metric("Candidate Mondrian Group", cand_group.upper())

# =====================================================================
# SECTION 5: SYSTEM INSIGHTS (Footer)
# =====================================================================
st.divider()
with st.expander("📋 System Insights & Model Metadata"):
    si1, si2 = st.columns(2)
    with si1:
        st.markdown("**Model Architecture:** ToxNet (Multi-task MLP)")
        st.markdown("**Loss Function:** Focal Loss (γ=2, α=0.75)")
        st.markdown("**Splitting:** Murcko Scaffold Split (Train/Val/Calib/Test)")
        st.markdown("**Calibration:** Mondrian Conformal Prediction (90% coverage)")
    with si2:
        st.markdown("**Endpoints:** 12 Tox21 biological assays")
        st.markdown(f"**Thresholds:** Per-endpoint (calibrated for ≥85% recall)")
        st.markdown("**Bioisostere Source:** Mock ChEMBL cache")
        st.markdown("**SHAP:** Simulated attribution (production: DeepExplainer)")
