import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from models import BreastCancerPipeline
from data_generator import get_full_dataset, KENYAN_COUNTIES, KENYAN_NAMES, ETHNIC_GROUPS, EDUCATION_LEVELS, OCCUPATION

# ─────────────────────── PAGE CONFIG ───────────────────────
st.set_page_config(
    page_title="MAMA AFYA – Kenya Breast Cancer AI System",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────── CUSTOM CSS ───────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #8B1A4A 0%, #C2185B 50%, #E91E63 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(139, 26, 74, 0.3);
    }

    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        opacity: 0.9;
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
    }

    .pipeline-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 5px solid #C2185B;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #FFF0F5, #FCE4EC);
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #F8BBD9;
    }

    .metric-number {
        font-size: 2rem;
        font-weight: 700;
        color: #8B1A4A;
        font-family: 'Playfair Display', serif;
    }

    .metric-label {
        font-size: 0.8rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .risk-high { background: #FFEBEE; border-left: 4px solid #F44336; padding: 1rem; border-radius: 8px; }
    .risk-moderate { background: #FFF8E1; border-left: 4px solid #FF9800; padding: 1rem; border-radius: 8px; }
    .risk-low { background: #E8F5E9; border-left: 4px solid #4CAF50; padding: 1rem; border-radius: 8px; }

    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        color: #8B1A4A;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #FCE4EC;
    }

    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #C2185B, #E91E63);
    }

    .pill {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.2rem;
    }

    .pill-pink { background: #FCE4EC; color: #880E4F; }
    .pill-green { background: #E8F5E9; color: #1B5E20; }
    .pill-orange { background: #FFF3E0; color: #E65100; }
    .pill-red { background: #FFEBEE; color: #B71C1C; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: #FCE4EC;
        border-radius: 8px 8px 0 0;
        color: #8B1A4A;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: #C2185B !important;
        color: white !important;
    }

    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────── INIT PIPELINE ───────────────────────
@st.cache_resource(show_spinner=False)
def load_pipeline():
    pipeline = BreastCancerPipeline()
    pipeline.train_all()
    return pipeline

@st.cache_data(show_spinner=False)
def load_data():
    return get_full_dataset()

# ─────────────────────── HEADER ───────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎗️ MAMA AFYA</h1>
    <p>AI-Powered Breast Cancer Early Detection & Management System — Kenya</p>
    <p style="font-size:0.85rem; opacity:0.75;">Population Risk · Imaging AI · Diagnosis · Treatment · Recurrence Monitoring</p>
</div>
""", unsafe_allow_html=True)

# Load pipeline
with st.spinner("🔬 Initializing AI models with Kenyan clinical data..."):
    pipeline = load_pipeline()
    df = load_data()

st.success("✅ All 5 AI pipeline models loaded and ready.")

# ─────────────────────── SIDEBAR ───────────────────────
st.sidebar.markdown("## 🧭 Navigation")
page = st.sidebar.radio("", [
    "🏠 Dashboard",
    "👩 Patient Assessment",
    "📊 Population Analytics",
    "🤖 Model Performance",
    "📋 About the System"
])

# ══════════════════════════════════════════
#  PAGE 1: DASHBOARD
# ══════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown('<div class="section-title">📈 Population Overview – Kenya Synthetic Dataset</div>', unsafe_allow_html=True)

    total = len(df)
    cancer_count = df["has_cancer"].sum()
    high_risk = (df["risk_score"] >= 0.7).sum()
    stage34 = df[df["cancer_stage"] >= 3].shape[0]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{total}</div><div class="metric-label">Total Patients</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{cancer_count}</div><div class="metric-label">Diagnosed Cancer</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{high_risk}</div><div class="metric-label">High Risk Women</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-number">{stage34}</div><div class="metric-label">Stage 3–4 Cases</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        stage_df = df[df["has_cancer"] == 1]["cancer_stage"].value_counts().reset_index()
        stage_df.columns = ["Stage", "Count"]
        stage_df["Stage"] = stage_df["Stage"].apply(lambda x: f"Stage {x}")
        stage_df = stage_df.sort_values("Stage")
        fig = px.bar(stage_df, x="Stage", y="Count",
                     color="Stage",
                     color_discrete_sequence=["#F48FB1", "#F06292", "#E91E63", "#880E4F"],
                     title="Cancer Cases by Stage",
                     template="plotly_white")
        fig.update_layout(showlegend=False, title_font_family="Playfair Display",
                          title_font_size=16, title_font_color="#8B1A4A")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        county_df = df[df["has_cancer"] == 1]["county"].value_counts().reset_index()
        county_df.columns = ["County", "Cases"]
        fig2 = px.bar(county_df.head(10), x="Cases", y="County", orientation="h",
                      color="Cases",
                      color_continuous_scale=["#FCE4EC", "#C2185B"],
                      title="Cancer Cases by County (Top 10)",
                      template="plotly_white")
        fig2.update_layout(title_font_family="Playfair Display",
                           title_font_size=16, title_font_color="#8B1A4A",
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(df, x="age", color="has_cancer",
                            color_discrete_map={0: "#F8BBD9", 1: "#C2185B"},
                            labels={"has_cancer": "Cancer", "age": "Age"},
                            title="Age Distribution by Cancer Status",
                            template="plotly_white", barmode="overlay", opacity=0.75)
        fig3.update_layout(title_font_family="Playfair Display",
                           title_font_size=16, title_font_color="#8B1A4A")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        tumor_df = df[df["tumor_type"] != "None"]["tumor_type"].value_counts().reset_index()
        tumor_df.columns = ["Subtype", "Count"]
        fig4 = px.pie(tumor_df, names="Subtype", values="Count",
                      color_discrete_sequence=px.colors.sequential.RdPu,
                      title="Tumor Molecular Subtypes",
                      template="plotly_white")
        fig4.update_layout(title_font_family="Playfair Display",
                           title_font_size=16, title_font_color="#8B1A4A")
        st.plotly_chart(fig4, use_container_width=True)

    # Pipeline visual
    st.markdown('<div class="section-title">🔄 AI Pipeline Stages</div>', unsafe_allow_html=True)
    stages = [
        ("🏘️", "POPULATION", "Risk Stratification", "GBM model flags high-risk women from community data"),
        ("🩻", "SCREENING", "Imaging AI", "CNN-based mammogram analysis with BI-RADS scoring"),
        ("🔬", "DIAGNOSIS", "NLP + Biomarkers", "Confirms cancer, grades tumor, identifies molecular subtype"),
        ("💊", "TREATMENT", "Personalized Protocol", "ML recommends treatment based on Kenya drug availability"),
        ("📡", "FOLLOW-UP", "Recurrence Monitor", "Predictive model monitors for post-treatment recurrence"),
    ]
    cols = st.columns(5)
    for i, (icon, level, title, desc) in enumerate(stages):
        with cols[i]:
            st.markdown(f"""
            <div style="background:white; border-radius:12px; padding:1rem; text-align:center;
                        border-top: 4px solid #C2185B; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
                <div style="font-size:1.8rem">{icon}</div>
                <div style="font-size:0.65rem; color:#C2185B; font-weight:700; letter-spacing:1px;">{level}</div>
                <div style="font-weight:600; font-size:0.9rem; margin:0.3rem 0;">{title}</div>
                <div style="font-size:0.75rem; color:#666;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════
#  PAGE 2: PATIENT ASSESSMENT
# ══════════════════════════════════════════
elif page == "👩 Patient Assessment":
    st.markdown('<div class="section-title">👩 Individual Patient Assessment</div>', unsafe_allow_html=True)
    st.info("Complete the patient profile below to run through all 5 pipeline levels.")

    with st.expander("⚡ Load Sample Patient", expanded=False):
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sample_type = st.selectbox("Patient Profile", ["Low Risk", "Moderate Risk", "High Risk – TNBC", "Stage 3 Case"])
        with col_s2:
            if st.button("Load Sample"):
                samples = {
                    "Low Risk": dict(age=32, bmi=22.5, parity=2, age_first_birth=24, breastfeeding_months=18,
                                     menarche_age=14, menopause=0, family_history=0, brca_mutation=0,
                                     contraceptive_years=2, hrt_use=0, alcohol_use=0, smoking=0,
                                     physical_activity="High", breast_density="B", prev_biopsy=0,
                                     birads_category=1, mass_present=0, mass_shape="None", mass_margin="None",
                                     calcifications=0, skin_thickening=0, nipple_retraction=0, ai_malignancy_probability=0.05,
                                     er_positive=1, pr_positive=1, her2_positive=0, ki67_percent=8,
                                     ca153_u_ml=12, cea_ng_ml=1.5, hemoglobin_g_dl=13.2, wbc_10e3=6.5,
                                     platelets_10e3=290, lymph_node_involvement=0),
                    "High Risk – TNBC": dict(age=42, bmi=29.5, parity=4, age_first_birth=18, breastfeeding_months=6,
                                              menarche_age=12, menopause=0, family_history=1, brca_mutation=1,
                                              contraceptive_years=8, hrt_use=0, alcohol_use=1, smoking=0,
                                              physical_activity="Low", breast_density="D", prev_biopsy=1,
                                              birads_category=5, mass_present=1, mass_shape="Irregular",
                                              mass_margin="Spiculated", calcifications=1, skin_thickening=1,
                                              nipple_retraction=1, ai_malignancy_probability=0.92,
                                              er_positive=0, pr_positive=0, her2_positive=0, ki67_percent=65,
                                              ca153_u_ml=185, cea_ng_ml=12.5, hemoglobin_g_dl=10.1, wbc_10e3=11.2,
                                              platelets_10e3=420, lymph_node_involvement=1),
                    "Stage 3 Case": dict(age=55, bmi=31.0, parity=5, age_first_birth=17, breastfeeding_months=24,
                                          menarche_age=13, menopause=1, family_history=1, brca_mutation=0,
                                          contraceptive_years=5, hrt_use=1, alcohol_use=0, smoking=0,
                                          physical_activity="Low", breast_density="C", prev_biopsy=0,
                                          birads_category=4, mass_present=1, mass_shape="Irregular",
                                          mass_margin="Indistinct", calcifications=1, skin_thickening=1,
                                          nipple_retraction=0, ai_malignancy_probability=0.78,
                                          er_positive=1, pr_positive=0, her2_positive=1, ki67_percent=38,
                                          ca153_u_ml=95, cea_ng_ml=6.5, hemoglobin_g_dl=11.5, wbc_10e3=9.5,
                                          platelets_10e3=350, lymph_node_involvement=1),
                    "Moderate Risk": dict(age=46, bmi=26.0, parity=3, age_first_birth=22, breastfeeding_months=12,
                                           menarche_age=13, menopause=0, family_history=0, brca_mutation=0,
                                           contraceptive_years=6, hrt_use=0, alcohol_use=0, smoking=0,
                                           physical_activity="Moderate", breast_density="C", prev_biopsy=0,
                                           birads_category=3, mass_present=0, mass_shape="None", mass_margin="None",
                                           calcifications=1, skin_thickening=0, nipple_retraction=0,
                                           ai_malignancy_probability=0.35,
                                           er_positive=1, pr_positive=1, her2_positive=0, ki67_percent=18,
                                           ca153_u_ml=32, cea_ng_ml=2.8, hemoglobin_g_dl=12.5, wbc_10e3=7.2,
                                           platelets_10e3=310, lymph_node_involvement=0),
                }
                st.session_state["sample"] = samples[sample_type]
                st.success(f"Sample loaded: {sample_type}")

    sample = st.session_state.get("sample", {})

    tab1, tab2, tab3 = st.tabs(["👤 Demographics & History", "🩻 Imaging Findings", "🧪 Biomarkers & Lab"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 18, 90, int(sample.get("age", 45)))
            bmi = st.number_input("BMI", 15.0, 55.0, float(sample.get("bmi", 26.0)), step=0.1)
            parity = st.number_input("Parity (no. of children)", 0, 15, int(sample.get("parity", 3)))
            age_first_birth = st.number_input("Age at first birth", 0, 45, int(sample.get("age_first_birth", 22)))
        with c2:
            breastfeeding_months = st.number_input("Total breastfeeding (months)", 0, 72, int(sample.get("breastfeeding_months", 12)))
            menarche_age = st.number_input("Age at menarche", 9, 18, int(sample.get("menarche_age", 13)))
            menopause = st.selectbox("Menopausal status", ["No", "Yes"], index=int(sample.get("menopause", 0)))
            contraceptive_years = st.number_input("Contraceptive use (years)", 0.0, 30.0, float(sample.get("contraceptive_years", 3.0)), step=0.5)
        with c3:
            family_history = st.selectbox("Family history of breast cancer", ["No", "Yes"], index=int(sample.get("family_history", 0)))
            brca_mutation = st.selectbox("BRCA1/2 mutation", ["No", "Yes"], index=int(sample.get("brca_mutation", 0)))
            hrt_use = st.selectbox("Hormone replacement therapy", ["No", "Yes"], index=int(sample.get("hrt_use", 0)))
            alcohol_use = st.selectbox("Alcohol use", ["No", "Yes"], index=int(sample.get("alcohol_use", 0)))
            physical_activity = st.selectbox("Physical activity level", ["Low", "Moderate", "High"],
                                             index=["Low", "Moderate", "High"].index(sample.get("physical_activity", "Moderate")))
            breast_density = st.selectbox("Breast density (BI-RADS)", ["A", "B", "C", "D"],
                                          index=["A", "B", "C", "D"].index(sample.get("breast_density", "B")))
            prev_biopsy = st.selectbox("Previous breast biopsy", ["No", "Yes"], index=int(sample.get("prev_biopsy", 0)))

    with tab2:
        c1, c2, c3 = st.columns(3)
        with c1:
            birads = st.selectbox("BI-RADS Category", [1, 2, 3, 4, 5], index=int(sample.get("birads_category", 1)) - 1)
            mass_present = st.selectbox("Mass present", ["No", "Yes"], index=int(sample.get("mass_present", 0)))
            mass_shape = st.selectbox("Mass shape", ["None", "Round", "Oval", "Lobular", "Irregular"],
                                      index=["None", "Round", "Oval", "Lobular", "Irregular"].index(sample.get("mass_shape", "None")))
        with c2:
            mass_margin = st.selectbox("Mass margin", ["None", "Circumscribed", "Microlobulated", "Indistinct", "Spiculated"],
                                       index=["None", "Circumscribed", "Microlobulated", "Indistinct", "Spiculated"].index(sample.get("mass_margin", "None")))
            calcifications = st.selectbox("Calcifications", ["No", "Yes"], index=int(sample.get("calcifications", 0)))
            skin_thickening = st.selectbox("Skin thickening", ["No", "Yes"], index=int(sample.get("skin_thickening", 0)))
        with c3:
            nipple_retraction = st.selectbox("Nipple retraction", ["No", "Yes"], index=int(sample.get("nipple_retraction", 0)))
            ai_prob = st.slider("AI Malignancy Probability (imaging model output)", 0.0, 1.0,
                                float(sample.get("ai_malignancy_probability", 0.2)), step=0.01)
            lymph_node = st.selectbox("Lymph node involvement", ["No", "Yes"], index=int(sample.get("lymph_node_involvement", 0)))

    with tab3:
        c1, c2, c3 = st.columns(3)
        with c1:
            er_pos = st.selectbox("ER (Estrogen Receptor)", ["Negative", "Positive"], index=int(sample.get("er_positive", 1)))
            pr_pos = st.selectbox("PR (Progesterone Receptor)", ["Negative", "Positive"], index=int(sample.get("pr_positive", 1)))
            her2_pos = st.selectbox("HER2 Status", ["Negative", "Positive"], index=int(sample.get("her2_positive", 0)))
        with c2:
            ki67 = st.number_input("Ki-67 Index (%)", 0.0, 100.0, float(sample.get("ki67_percent", 15.0)), step=0.5)
            ca153 = st.number_input("CA 15-3 (U/mL)", 5.0, 500.0, float(sample.get("ca153_u_ml", 20.0)), step=1.0)
            cea = st.number_input("CEA (ng/mL)", 0.5, 100.0, float(sample.get("cea_ng_ml", 2.0)), step=0.1)
        with c3:
            hemoglobin = st.number_input("Hemoglobin (g/dL)", 5.0, 18.0, float(sample.get("hemoglobin_g_dl", 12.5)), step=0.1)
            wbc = st.number_input("WBC (×10³/μL)", 1.0, 20.0, float(sample.get("wbc_10e3", 7.0)), step=0.1)
            platelets = st.number_input("Platelets (×10³/μL)", 50, 800, int(sample.get("platelets_10e3", 280)))

    # Build patient dict
    patient_data = {
        "age": age, "bmi": bmi, "parity": parity, "age_first_birth": age_first_birth,
        "breastfeeding_months": breastfeeding_months, "menarche_age": menarche_age,
        "menopause": 1 if menopause == "Yes" else 0,
        "family_history": 1 if family_history == "Yes" else 0,
        "brca_mutation": 1 if brca_mutation == "Yes" else 0,
        "contraceptive_years": contraceptive_years,
        "hrt_use": 1 if hrt_use == "Yes" else 0,
        "alcohol_use": 1 if alcohol_use == "Yes" else 0,
        "smoking": 0, "prev_biopsy": 1 if prev_biopsy == "Yes" else 0,
        "physical_activity": physical_activity,
        "breast_density": breast_density,
        "birads_category": birads,
        "mass_present": 1 if mass_present == "Yes" else 0,
        "mass_shape": mass_shape, "mass_margin": mass_margin,
        "calcifications": 1 if calcifications == "Yes" else 0,
        "skin_thickening": 1 if skin_thickening == "Yes" else 0,
        "nipple_retraction": 1 if nipple_retraction == "Yes" else 0,
        "ai_malignancy_probability": ai_prob,
        "lymph_node_involvement": 1 if lymph_node == "Yes" else 0,
        "er_positive": 1 if er_pos == "Positive" else 0,
        "pr_positive": 1 if pr_pos == "Positive" else 0,
        "her2_positive": 1 if her2_pos == "Positive" else 0,
        "ki67_percent": ki67, "ca153_u_ml": ca153, "cea_ng_ml": cea,
        "hemoglobin_g_dl": hemoglobin, "wbc_10e3": wbc, "platelets_10e3": platelets,
    }

    st.markdown("<br>", unsafe_allow_html=True)
    run_col, _ = st.columns([1, 3])
    with run_col:
        run_pipeline = st.button("🚀 Run Full AI Pipeline", type="primary", use_container_width=True)

    if run_pipeline:
        with st.spinner("Running patient through all 5 pipeline levels..."):
            results = pipeline.run_full_pipeline(patient_data)

        st.markdown("---")
        st.markdown('<div class="section-title">📋 Pipeline Results</div>', unsafe_allow_html=True)

        # ── LEVEL 1: Risk
        st.markdown("### 🏘️ Level 1 — Population Risk Stratification")
        risk = results["risk"]
        rl = risk["risk_level"]
        css_class = {"High": "risk-high", "Moderate": "risk-moderate", "Low": "risk-low"}[rl]
        color_emoji = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}[rl]
        st.markdown(f"""
        <div class="{css_class}">
            <strong>{color_emoji} Risk Level: {rl}</strong> &nbsp;|&nbsp; 
            Probability: <strong>{risk['risk_probability']*100:.1f}%</strong><br>
            <em>{risk['recommendation']}</em><br>
            <small>Top risk factors: {', '.join(risk['top_risk_factors'])}</small>
        </div>
        """, unsafe_allow_html=True)
        st.progress(risk["risk_probability"])

        st.markdown("---")

        # ── LEVEL 2: Imaging
        st.markdown("### 🩻 Level 2 — Imaging AI Analysis")
        img = results["imaging"]
        img_color = {"red": "risk-high", "orange": "risk-moderate", "green": "risk-low"}[img["color"]]
        st.markdown(f"""
        <div class="{img_color}">
            <strong>{img['flag']}</strong><br>
            Malignancy probability: <strong>{img['malignancy_probability']*100:.1f}%</strong><br>
            BI-RADS {birads}: {img['birads_interpretation']}<br>
            Findings: {' | '.join(img['findings_summary'])}
        </div>
        """, unsafe_allow_html=True)
        st.progress(img["malignancy_probability"])

        st.markdown("---")

        # ── LEVEL 3: Diagnosis
        st.markdown("### 🔬 Level 3 — Diagnosis & Staging")
        diag = results["diagnosis"]
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.metric("Cancer Probability", f"{diag['cancer_probability']*100:.1f}%")
            st.metric("Predicted Stage", f"Stage {diag['predicted_stage']}")
            st.metric("Lymph Node Status", diag["lymph_node_status"])
        with d_col2:
            st.info(f"**Molecular Subtype:** {diag['molecular_subtype']}")
            st.info(f"**Tumor Grade:** {diag['tumor_grade']}")
            st.success(f"**Prognosis:** {diag['prognosis']}")

        stage_df = pd.DataFrame(list(diag["stage_probabilities"].items()), columns=["Stage", "Probability"])
        fig_stage = px.bar(stage_df, x="Stage", y="Probability",
                           color="Probability", color_continuous_scale=["#FCE4EC", "#C2185B"],
                           title="Stage Probability Distribution",
                           template="plotly_white")
        fig_stage.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_stage, use_container_width=True)

        st.markdown("---")

        # ── LEVEL 4: Treatment
        st.markdown("### 💊 Level 4 — Personalized Treatment Protocol")
        tx = results["treatment"]
        st.markdown("**Recommended Protocol:**")
        for i, step in enumerate(tx["primary_protocol"], 1):
            availability = tx["kenya_availability"].get(step, "")
            avail_text = f" — *Available at: {availability}*" if availability else ""
            st.markdown(f"{i}. **{step}**{avail_text}")

        if tx["additional_considerations"]:
            st.markdown("**Additional Considerations:**")
            for c in tx["additional_considerations"]:
                st.markdown(f"• {c}")

        st.markdown(f"**Follow-up Plan:** {tx['monitoring']}")
        if tx["clinical_trial_eligible"]:
            st.warning("⚗️ Patient may be eligible for clinical trials. Consider referral to Aga Khan University Hospital or KEMRI.")

        st.markdown("---")

        # ── LEVEL 5: Recurrence
        st.markdown("### 📡 Level 5 — Recurrence Risk Monitoring")
        rec = results["recurrence"]
        r_class = "risk-high" if rec["recurrence_probability"] >= 0.65 else \
                  "risk-moderate" if rec["recurrence_probability"] >= 0.35 else "risk-low"
        st.markdown(f"""
        <div class="{r_class}">
            <strong>{rec['recurrence_risk']}</strong> — 
            Probability: <strong>{rec['recurrence_probability']*100:.1f}%</strong><br>
            Action: {rec['recommended_action']}<br>
            Follow-up interval: <strong>{rec['follow_up_interval']}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.progress(rec["recurrence_probability"])

        # Summary gauge
        st.markdown("---")
        st.markdown("### 📊 Overall Risk Summary")
        fig_gauge = go.Figure()
        overall_risk = (risk["risk_probability"] + img["malignancy_probability"] + diag["cancer_probability"]) / 3

        fig_gauge.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=overall_risk * 100,
            title={"text": "Overall Cancer Risk Score", "font": {"size": 18}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#C2185B"},
                "steps": [
                    {"range": [0, 40], "color": "#E8F5E9"},
                    {"range": [40, 70], "color": "#FFF8E1"},
                    {"range": [70, 100], "color": "#FFEBEE"},
                ],
                "threshold": {"line": {"color": "#880E4F", "width": 3}, "thickness": 0.75, "value": 70}
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ══════════════════════════════════════════
#  PAGE 3: POPULATION ANALYTICS
# ══════════════════════════════════════════
elif page == "📊 Population Analytics":
    st.markdown('<div class="section-title">📊 Population Analytics</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(df, x="age", y="bmi", color="has_cancer",
                         color_discrete_map={0: "#F8BBD9", 1: "#C2185B"},
                         symbol="physical_activity", opacity=0.6,
                         title="Age vs BMI by Cancer Status",
                         labels={"has_cancer": "Cancer"},
                         template="plotly_white")
        fig.update_layout(title_font_family="Playfair Display", title_font_color="#8B1A4A")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        density_df = df.groupby(["breast_density", "has_cancer"]).size().reset_index(name="count")
        density_df["Status"] = density_df["has_cancer"].map({0: "No Cancer", 1: "Cancer"})
        fig2 = px.bar(density_df, x="breast_density", y="count", color="Status",
                      color_discrete_map={"No Cancer": "#F8BBD9", "Cancer": "#C2185B"},
                      title="Breast Density vs Cancer Prevalence",
                      template="plotly_white", barmode="group")
        fig2.update_layout(title_font_family="Playfair Display", title_font_color="#8B1A4A")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.box(df, x="cancer_stage", y="ca153_u_ml",
                      color="cancer_stage",
                      color_discrete_sequence=px.colors.sequential.RdPu,
                      title="CA 15-3 Levels by Cancer Stage",
                      template="plotly_white",
                      labels={"cancer_stage": "Stage", "ca153_u_ml": "CA 15-3 (U/mL)"})
        fig3.update_layout(showlegend=False, title_font_family="Playfair Display", title_font_color="#8B1A4A")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        risk_county = df.groupby("county")["risk_score"].mean().reset_index().sort_values("risk_score", ascending=False)
        fig4 = px.bar(risk_county, x="county", y="risk_score",
                      color="risk_score",
                      color_continuous_scale=["#FCE4EC", "#C2185B"],
                      title="Average Risk Score by County",
                      template="plotly_white")
        fig4.update_layout(xaxis_tickangle=45, coloraxis_showscale=False,
                           title_font_family="Playfair Display", title_font_color="#8B1A4A")
        st.plotly_chart(fig4, use_container_width=True)

    # Correlation heatmap
    st.markdown("#### Feature Correlation with Cancer Risk")
    num_cols = ["age", "bmi", "parity", "breastfeeding_months", "ki67_percent",
                "ca153_u_ml", "risk_score", "ai_malignancy_probability", "has_cancer"]
    corr = df[num_cols].corr()
    fig5 = px.imshow(corr, color_continuous_scale="RdBu_r", title="Correlation Heatmap",
                     template="plotly_white", text_auto=".2f")
    fig5.update_layout(title_font_family="Playfair Display", title_font_color="#8B1A4A", height=450)
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════════════
#  PAGE 4: MODEL PERFORMANCE
# ══════════════════════════════════════════
elif page == "🤖 Model Performance":
    st.markdown('<div class="section-title">🤖 Model Performance Metrics</div>', unsafe_allow_html=True)

    metrics = pipeline.metrics

    col1, col2, col3, col4 = st.columns(4)
    auc_vals = {
        "Risk Model": metrics.get("risk", {}).get("auc", 0),
        "Imaging AI": metrics.get("imaging", {}).get("auc", 0),
        "Diagnosis": metrics.get("diagnosis", {}).get("auc", 0),
        "Recurrence": metrics.get("recurrence", {}).get("auc", 0),
    }
    for col, (name, auc) in zip([col1, col2, col3, col4], auc_vals.items()):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-number">{auc:.3f}</div><div class="metric-label">{name} AUC-ROC</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    fig_auc = px.bar(
        x=list(auc_vals.keys()), y=list(auc_vals.values()),
        color=list(auc_vals.values()),
        color_continuous_scale=["#FCE4EC", "#C2185B"],
        labels={"x": "Model", "y": "AUC-ROC"},
        title="AUC-ROC by Pipeline Level",
        template="plotly_white"
    )
    fig_auc.update_layout(coloraxis_showscale=False, yaxis_range=[0, 1],
                          title_font_family="Playfair Display", title_font_color="#8B1A4A")
    fig_auc.add_hline(y=0.8, line_dash="dash", line_color="#C2185B",
                      annotation_text="Good threshold (0.8)", annotation_position="top right")
    st.plotly_chart(fig_auc, use_container_width=True)

    # Feature importance
    st.markdown("#### Feature Importance – Risk Stratification Model")
    feature_names = pipeline.risk_model.feature_names
    importances = pipeline.risk_model.model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=True)
    fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance", color_continuous_scale=["#FCE4EC", "#C2185B"],
                    template="plotly_white", title="Feature Importance (Risk Model)")
    fig_fi.update_layout(coloraxis_showscale=False,
                         title_font_family="Playfair Display", title_font_color="#8B1A4A", height=500)
    st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════
#  PAGE 5: ABOUT
# ══════════════════════════════════════════
elif page == "📋 About the System":
    st.markdown('<div class="section-title">📋 About MAMA AFYA</div>', unsafe_allow_html=True)

    st.markdown("""
    **MAMA AFYA** (*Swahili for "Healthy Mother"*) is an AI-powered breast cancer early detection and 
    management pipeline designed specifically for the Kenyan healthcare context.

    ### 🎯 Mission
    To reduce late-stage breast cancer diagnosis in Kenya by bringing AI-powered screening, 
    risk stratification, and treatment guidance to clinicians and community health workers.

    ### 🔬 Pipeline Architecture
    | Level | Module | Algorithm | Purpose |
    |---|---|---|---|
    | 1 | Risk Stratification | Gradient Boosting | Identify high-risk women from population data |
    | 2 | Imaging AI | Random Forest + BI-RADS | Flag suspicious mammogram/ultrasound findings |
    | 3 | Diagnosis & Staging | GBM + Biomarkers | Confirm cancer, grade tumor, molecular subtype |
    | 4 | Treatment Recommendation | Rule-based + ML | Kenya-contextualized treatment protocols |
    | 5 | Recurrence Monitoring | Gradient Boosting | Post-treatment surveillance and risk scoring |

    ### 🇰🇪 Kenya-Specific Features
    - **TNBC awareness**: Triple Negative Breast Cancer (more prevalent in African women) is explicitly modeled
    - **Drug availability**: Treatment recommendations reference KNH, Aga Khan, MP Shah, KEMSA availability
    - **County-level analytics**: Risk stratified by Kenyan counties
    - **CHW integration**: Designed to work with Kenya's community health volunteer network
    - **Low-resource settings**: Algorithms optimized to work with data that is realistically collectible in Kenya

    ### 📊 Data
    This system uses **500 synthetic patients** generated to reflect Kenyan epidemiological patterns including:
    - Higher parity, earlier first birth, longer breastfeeding
    - Higher prevalence of dense breast tissue
    - TNBC subtype representation
    - County-level geographic variation
    - Realistic income, education, and distance-to-facility distributions

    ### ⚠️ Disclaimer
    This is a **research and educational prototype**. It is not intended for clinical use without 
    validation on real patient data and regulatory approval. Always consult qualified medical professionals.

    ### 👩‍💻 Technology Stack
    `Python` · `Scikit-learn` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#999; font-size:0.8rem; padding:1rem;">
    🎗️ MAMA AFYA — AI Breast Cancer Detection System | Kenya | Built with Streamlit & Scikit-learn
</div>
""", unsafe_allow_html=True)
