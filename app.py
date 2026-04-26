import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
from rdkit import Chem
from rdkit.Chem import Draw
import sys

# Ensure src module can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.prescription_pipeline import run_prescription_pipeline
from src.featurize import TARGET_COLS

# Page config
st.set_page_config(
    page_title="Tox21 Prediction & Prescription",
    page_icon="🧪",
    layout="wide"
)

# Premium Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
    }
    .stApp {
        color: #FAFAFA;
    }
    .metric-card {
        background: #1E2329;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .warning-text {
        color: #ff4b4b;
        font-weight: bold;
    }
    .safe-text {
        color: #00cc96;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_artifact():
    artifact_path = 'models/model_artifact.pkl'
    if not os.path.exists(artifact_path):
        return None
    with open(artifact_path, 'rb') as f:
        return pickle.load(f)

# Sidebar
st.sidebar.title("🧪 Tox21 AI Dashboard")
st.sidebar.markdown("### Molecular Optimization & Safety Prediction")

artifact = load_artifact()

if artifact is None:
    st.error("Model artifact not found! Please ensure you have run Phase 4/5 and generated `models/model_artifact.pkl`.")
    st.stop()

# Default to Aniline (toxic)
target_smiles = st.sidebar.text_input("Target SMILES", "c1ccccc1N")

st.sidebar.markdown("### Therapeutic Override")
ignored_endpoints = st.sidebar.multiselect(
    "Select endpoints to ignore (e.g. for targeted therapies)",
    options=TARGET_COLS,
    default=[]
)

run_btn = st.sidebar.button("Run Prescription Pipeline", type="primary")

# Main Area
st.title("Toxicity Analysis & Bioisostere Prescription")

if run_btn and target_smiles:
    mol = Chem.MolFromSmiles(target_smiles)
    if not mol:
        st.error("Invalid SMILES string provided.")
    else:
        with st.spinner("Running deep learning pipeline..."):
            result = run_prescription_pipeline(target_smiles, artifact)
            
        if result.get('error'):
            st.error(f"Pipeline Error: {result['error']}")
        elif result.get('status') == 'ABORTED':
            st.warning(f"Pipeline Aborted: {result['reason']}")
            if 'validation' in result:
                st.write(result['validation'])
        else:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Original Molecule")
                img = Draw.MolToImage(mol, size=(400, 400))
                st.image(img, use_container_width=True)
                
            with col2:
                st.markdown("### Toxicity Profile (Radar)")
                orig_probs = result['original_probs']
                
                # Radar Chart
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=orig_probs,
                    theta=TARGET_COLS,
                    fill='toself',
                    name='Original Molecule',
                    line_color='#ff4b4b'
                ))
                
                # Add thresholds line (0.5 for demonstration, or from artifact if available)
                thresholds = [artifact['thresholds'].get(ep, 0.5) for ep in TARGET_COLS]
                fig.add_trace(go.Scatterpolar(
                    r=thresholds,
                    theta=TARGET_COLS,
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='Safety Threshold'
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=True,
                    template="plotly_dark",
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.divider()
            
            st.markdown("### 💊 Safer Bioisostere Prescriptions")
            suggestions = result.get('suggestions', [])
            
            if not suggestions:
                st.info("No viable safer alternatives found that pass ADME and Pareto filters.")
            else:
                # Filter out ones that are entirely worsened
                valid_sugs = []
                for s in suggestions:
                    if s['pareto_status'] in ['Dominates', 'Trade-off']:
                        valid_sugs.append(s)
                
                if valid_sugs:
                    df = pd.DataFrame(valid_sugs)
                    
                    # Highlight Pareto dominant
                    def highlight_status(val):
                        if val == 'Dominates':
                            return 'color: #00cc96; font-weight: bold'
                        elif val == 'Trade-off':
                            return 'color: #ffa15a'
                        return ''
                    
                    st.dataframe(
                        df[['smiles', 'replacement', 'pareto_status', 'improved', 'worsened', 'sa_score', 'delta_logp', 'ood_warning', 'mondrian_group']].style.map(highlight_status, subset=['pareto_status']),
                        use_container_width=True,
                        height=300
                    )
                    
                    st.markdown("#### Candidate Comparison")
                    selected_candidate = st.selectbox("Select a candidate to view details:", df['smiles'].tolist())
                    
                    cand_data = df[df['smiles'] == selected_candidate].iloc[0]
                    cand_mol = Chem.MolFromSmiles(selected_candidate)
                    
                    # We need to run predict on the candidate to get its radar chart
                    mondrian = artifact['mondrian_predictor']
                    from src.featurize import smiles_to_morgan_with_info
                    fp, _ = smiles_to_morgan_with_info(selected_candidate)
                    cand_results = mondrian.predict_with_uncertainty(selected_candidate, fp)
                    cand_probs = [r['prob'] for r in cand_results]
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        img2 = Draw.MolToImage(cand_mol, size=(400, 400))
                        st.image(img2, use_container_width=True)
                        st.write(f"**Replacement Rule:** {cand_data['replacement']}")
                        st.write(f"**Mondrian Group:** {cand_data['mondrian_group']}")
                        if cand_data['ood_warning']:
                            st.markdown("<span class='warning-text'>⚠ WARNING: Candidate is Out-of-Distribution or highly uncertain. Proceed with caution.</span>", unsafe_allow_html=True)
                            
                    with c2:
                        fig2 = go.Figure()
                        fig2.add_trace(go.Scatterpolar(
                            r=orig_probs,
                            theta=TARGET_COLS,
                            mode='lines',
                            name='Original',
                            line_color='rgba(255, 75, 75, 0.5)'
                        ))
                        fig2.add_trace(go.Scatterpolar(
                            r=cand_probs,
                            theta=TARGET_COLS,
                            fill='toself',
                            name='Candidate',
                            line_color='#00cc96'
                        ))
                        fig2.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            template="plotly_dark",
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.info("No Pareto dominant or acceptable trade-off candidates found.")
