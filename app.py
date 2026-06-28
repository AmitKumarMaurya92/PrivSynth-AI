import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os

# ══════════════════════════════════════════════════════════════
# Page Config
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="PrivSynth-AI | Privacy-Preserving Synthetic Data",
    layout="wide",
    page_icon="🛡️",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# Custom CSS – Dark Premium Theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ---------- Import Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---------- Root Variables ---------- */
:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d2e;
    --bg-card: #1e2235;
    --accent-blue: #6366f1;
    --accent-purple: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-color: rgba(99, 102, 241, 0.15);
    --glow-blue: rgba(99, 102, 241, 0.25);
}

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background: linear-gradient(135deg, #0f1117 0%, #1a1d2e 50%, #0f1117 100%);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(30, 34, 53, 0.95) 0%, rgba(15, 17, 23, 0.98) 100%);
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--border-color);
}

section[data-testid="stSidebar"] .stRadio > label {
    color: var(--text-secondary) !important;
    font-weight: 600;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

section[data-testid="stSidebar"] .stRadio > div > label {
    padding: 0.65rem 1rem !important;
    border-radius: 10px !important;
    margin-bottom: 4px !important;
    transition: all 0.3s ease !important;
    border: 1px solid transparent !important;
}

section[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(99, 102, 241, 0.08) !important;
    border-color: var(--border-color) !important;
}

section[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"],
section[data-testid="stSidebar"] .stRadio > div [data-testid="stMarkdownContainer"] {
    color: var(--text-primary) !important;
}

/* ---------- Metric Cards ---------- */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(30, 34, 53, 0.6) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px var(--glow-blue);
}

div[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-weight: 500;
    font-size: 0.8rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700;
    font-size: 1.8rem;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}

/* ---------- Download Button ---------- */
.stDownloadButton > button {
    background: linear-gradient(135deg, var(--accent-emerald) 0%, #059669 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}

.stDownloadButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.5) !important;
}

/* ---------- DataFrames ---------- */
.stDataFrame {
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    overflow: hidden;
}

/* ---------- File Uploader ---------- */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border-color) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-blue) !important;
}

/* ---------- Select Box ---------- */
.stSelectbox > div > div {
    border-radius: 10px !important;
    border-color: var(--border-color) !important;
}

/* ---------- Info / Warning / Success Alerts ---------- */
.stAlert {
    border-radius: 12px !important;
    border: none !important;
}

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 0.5rem 1.2rem !important;
    font-weight: 500 !important;
}

/* ---------- Divider ---------- */
hr {
    border-color: var(--border-color) !important;
    margin: 1.5rem 0 !important;
}

/* ---------- Custom Classes ---------- */
.hero-title {
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 30%, #06b6d4 70%, #10b981 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-bottom: 2rem;
}

.section-header {
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent-blue);
    display: inline-block;
}

.tech-badge {
    display: inline-block;
    padding: 0.35rem 0.9rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
    border: 1px solid;
}

.badge-blue { background: rgba(99,102,241,0.1); color: #818cf8; border-color: rgba(99,102,241,0.3); }
.badge-purple { background: rgba(139,92,246,0.1); color: #a78bfa; border-color: rgba(139,92,246,0.3); }
.badge-cyan { background: rgba(6,182,212,0.1); color: #22d3ee; border-color: rgba(6,182,212,0.3); }
.badge-emerald { background: rgba(16,185,129,0.1); color: #34d399; border-color: rgba(16,185,129,0.3); }
.badge-amber { background: rgba(245,158,11,0.1); color: #fbbf24; border-color: rgba(245,158,11,0.3); }
.badge-rose { background: rgba(244,63,94,0.1); color: #fb7185; border-color: rgba(244,63,94,0.3); }

.glass-card {
    background: linear-gradient(135deg, rgba(30,34,53,0.7) 0%, rgba(15,17,23,0.5) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.workflow-step {
    display: flex;
    align-items: center;
    padding: 1rem 1.2rem;
    border-radius: 14px;
    margin-bottom: 0.6rem;
    background: rgba(30, 34, 53, 0.5);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.workflow-step:hover {
    background: rgba(99, 102, 241, 0.08);
    transform: translateX(6px);
}

.step-number {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
    color: white;
    font-weight: 700;
    font-size: 0.85rem;
    margin-right: 1rem;
    flex-shrink: 0;
}

.step-text {
    color: var(--text-primary);
    font-weight: 500;
    font-size: 0.95rem;
}

.privacy-tag {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 8px;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.privacy-very-high { background: rgba(16,185,129,0.15); color: #34d399; }
.privacy-high { background: rgba(6,182,212,0.15); color: #22d3ee; }
.privacy-medium { background: rgba(245,158,11,0.15); color: #fbbf24; }
.privacy-low { background: rgba(244,63,94,0.15); color: #fb7185; }

.stat-label {
    color: var(--text-secondary);
    font-size: 0.78rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
}
.stat-value {
    color: var(--text-primary);
    font-size: 2rem;
    font-weight: 800;
}
.stat-card {
    text-align: center;
    padding: 1.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, var(--bg-card) 0%, rgba(30,34,53,0.4) 100%);
    border: 1px solid var(--border-color);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.stat-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 36px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Session State
# ══════════════════════════════════════════════════════════════
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "synthetic_df" not in st.session_state:
    st.session_state.synthetic_df = None

# ══════════════════════════════════════════════════════════════
# Sidebar Navigation
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.5rem 0 1rem 0;">
        <div style="font-size: 2.5rem; margin-bottom: 0.3rem;">🛡️</div>
        <div style="font-size: 1.3rem; font-weight: 800;
             background: linear-gradient(135deg, #6366f1, #06b6d4);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             background-clip: text;">PrivSynth-AI</div>
        <div style="font-size: 0.7rem; color: #64748b; font-weight: 500;
             letter-spacing: 0.15em; text-transform: uppercase; margin-top: 0.2rem;">
             Dashboard v2.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        ["🏠  Home", "📂  Dataset Upload", "⚙️  Generate Synthetic Data",
         "📊  Comparison", "📈  Evaluation", "💾  Download"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="padding: 0.8rem; border-radius: 12px;
         background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.12);
         text-align: center;">
        <div style="font-size: 0.68rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 0.1em; font-weight: 600; margin-bottom: 0.3rem;">Status</div>
        <div style="font-size: 0.82rem; color: #34d399; font-weight: 600;">● Online</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# Helper: Load Model
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model(model_name):
    model_paths = {
        "Baseline CTGAN": "models/baseline_ctgan.pkl",
        "DP ε=5.0": "models/dp_ctgan_eps50.pkl",
        "DP ε=1.0": "models/dp_ctgan_eps10.pkl",
        "DP ε=0.5": "models/dp_ctgan_eps05.pkl",
    }
    path = model_paths.get(model_name)
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ══════════════════════════════════════════════════════════════
# Helper: Plotly Theme
# ══════════════════════════════════════════════════════════════
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#e2e8f0"),
    margin=dict(l=40, r=40, t=50, b=40),
)

COLORS = {
    "original": "#6366f1",
    "synthetic": "#06b6d4",
    "accent": "#8b5cf6",
    "emerald": "#10b981",
    "amber": "#f59e0b",
    "rose": "#f43f5e",
}


# ══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠  Home":
    # Hero
    st.markdown("""
    <div style="padding: 1rem 0 0.5rem 0;">
        <div class="hero-title">PrivSynth-AI</div>
        <div class="hero-subtitle">
            Privacy-Preserving Synthetic Data Generation with Differential Privacy
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Description Card
    st.markdown("""
    <div class="glass-card">
        <p style="color: #cbd5e1; font-size: 1.02rem; line-height: 1.8; margin: 0;">
            <strong style="color: #f1f5f9;">PrivSynth-AI</strong> enables organisations to generate
            high-fidelity synthetic datasets that preserve the statistical properties of the original
            data while providing <strong style="color: #8b5cf6;">formal differential-privacy
            guarantees</strong>. Upload your sensitive dataset, choose a privacy budget (ε), generate
            a synthetic counterpart, and inspect the <em>privacy–utility trade-off</em> — all from
            this dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # Tech Stack
        st.markdown('<div class="section-header">🧰 Technology Stack</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="glass-card" style="padding: 1.5rem 1.8rem;">
            <div style="margin-bottom: 0.9rem;">
                <span style="color: #94a3b8; font-weight: 600; font-size: 0.78rem;
                      text-transform: uppercase; letter-spacing: 0.1em;">Data Processing</span><br>
                <span class="tech-badge badge-blue">Pandas</span>
                <span class="tech-badge badge-blue">NumPy</span>
                <span class="tech-badge badge-blue">Scikit-Learn</span>
            </div>
            <div style="margin-bottom: 0.9rem;">
                <span style="color: #94a3b8; font-weight: 600; font-size: 0.78rem;
                      text-transform: uppercase; letter-spacing: 0.1em;">Generative Models</span><br>
                <span class="tech-badge badge-purple">CTGAN</span>
                <span class="tech-badge badge-purple">DP-SGD</span>
                <span class="tech-badge badge-purple">Opacus</span>
                <span class="tech-badge badge-purple">SmartNoise</span>
            </div>
            <div style="margin-bottom: 0.9rem;">
                <span style="color: #94a3b8; font-weight: 600; font-size: 0.78rem;
                      text-transform: uppercase; letter-spacing: 0.1em;">Privacy & Evaluation</span><br>
                <span class="tech-badge badge-rose">Membership Inference</span>
                <span class="tech-badge badge-rose">Distance Attacks</span>
                <span class="tech-badge badge-amber">XGBoost</span>
            </div>
            <div>
                <span style="color: #94a3b8; font-weight: 600; font-size: 0.78rem;
                      text-transform: uppercase; letter-spacing: 0.1em;">Dashboard</span><br>
                <span class="tech-badge badge-emerald">Streamlit</span>
                <span class="tech-badge badge-emerald">Plotly</span>
                <span class="tech-badge badge-cyan">Matplotlib</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        # Workflow
        st.markdown('<div class="section-header">🔄 Workflow</div>', unsafe_allow_html=True)
        steps = [
            ("1", "Upload your dataset (CSV)"),
            ("2", "Select & load a CTGAN model"),
            ("3", "Generate synthetic data"),
            ("4", "Compare distributions"),
            ("5", "Evaluate privacy metrics"),
            ("6", "Download results"),
        ]
        html_steps = ""
        for num, text in steps:
            html_steps += f"""
            <div class="workflow-step">
                <span class="step-number">{num}</span>
                <span class="step-text">{text}</span>
            </div>"""
        st.markdown(f'<div class="glass-card" style="padding: 1.2rem 1.5rem;">{html_steps}</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 2 — DATASET UPLOAD
# ══════════════════════════════════════════════════════════════
elif page == "📂  Dataset Upload":
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">📂 Dataset Upload</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Upload a CSV file to begin exploring and generating synthetic data.</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"],
                                     label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.original_df = df
        st.success("✅  File uploaded successfully!")

        # Stats row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Numeric Features", f"{df.select_dtypes(include=np.number).shape[1]}")
        c4.metric("Categorical Features", f"{df.select_dtypes(include='object').shape[1]}")

        st.markdown("---")

        tab_preview, tab_stats, tab_missing = st.tabs(
            ["📋 Data Preview", "📊 Column Statistics", "⚠️ Missing Values"]
        )

        with tab_preview:
            st.dataframe(df.head(20), width="stretch", height=400)

        with tab_stats:
            st.dataframe(df.describe(include="all").T, width="stretch", height=400)

        with tab_missing:
            missing = df.isnull().sum()
            missing_pct = (missing / len(df) * 100).round(2)
            missing_df = pd.DataFrame({"Missing Count": missing, "Percentage (%)": missing_pct})
            missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values(
                "Missing Count", ascending=False
            )
            if missing_df.empty:
                st.info("🎉 No missing values found in the dataset!")
            else:
                st.dataframe(missing_df, width="stretch")
                # Bar chart of missing values
                fig = px.bar(
                    missing_df.reset_index(),
                    x="index", y="Percentage (%)",
                    color="Percentage (%)",
                    color_continuous_scale=["#6366f1", "#f43f5e"],
                    labels={"index": "Column"},
                    title="Missing Values by Column",
                )
                fig.update_layout(**PLOTLY_LAYOUT, height=350)
                st.plotly_chart(fig, width="stretch")

    elif st.session_state.original_df is not None:
        df = st.session_state.original_df
        st.info("📌 Dataset already loaded from a previous upload.")
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]}")
        st.dataframe(df.head(10), width="stretch")
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 3rem 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">📁</div>
            <div style="color: #94a3b8; font-size: 1rem;">
                Drag and drop your CSV file above, or click <strong>Browse files</strong> to get started.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PAGE 3 — GENERATE SYNTHETIC DATA
# ══════════════════════════════════════════════════════════════
elif page == "⚙️  Generate Synthetic Data":
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">⚙️ Generate Synthetic Data</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Select a pre-trained model and generate privacy-preserving synthetic records.</div>',
                unsafe_allow_html=True)

    if st.session_state.original_df is None:
        st.warning("⚠️  Please upload a dataset in the **Dataset Upload** page first.")
    else:
        col_config, col_info = st.columns([2, 1], gap="large")

        with col_config:
            st.markdown('<div class="section-header">Model Configuration</div>',
                        unsafe_allow_html=True)
            model_choice = st.selectbox(
                "Select Model",
                ["Baseline CTGAN", "DP ε=5.0", "DP ε=1.0", "DP ε=0.5"],
                help="Lower ε → more privacy, less utility. Baseline has no DP noise.",
            )
            num_rows = st.slider(
                "Number of records to generate",
                min_value=100, max_value=50000, value=1000, step=100,
            )
            generate_btn = st.button("🚀  Generate Synthetic Data", width="stretch")

        with col_info:
            st.markdown('<div class="section-header">Model Info</div>', unsafe_allow_html=True)
            info_map = {
                "Baseline CTGAN": ("No DP", "Highest utility, no privacy guarantee.", "badge-rose"),
                "DP ε=5.0": ("ε = 5.0", "Moderate privacy, high utility.", "badge-amber"),
                "DP ε=1.0": ("ε = 1.0", "Strong privacy, good utility.", "badge-cyan"),
                "DP ε=0.5": ("ε = 0.5", "Very strong privacy, lower utility.", "badge-emerald"),
            }
            eps_label, desc, badge_cls = info_map[model_choice]
            st.markdown(f"""
            <div class="glass-card">
                <div style="margin-bottom: 0.5rem;">
                    <span class="tech-badge {badge_cls}" style="font-size: 0.9rem;">{eps_label}</span>
                </div>
                <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.7; margin: 0.5rem 0 0 0;">
                    {desc}
                </p>
            </div>
            """, unsafe_allow_html=True)

        if generate_btn:
            with st.spinner(f"Loading **{model_choice}** and generating **{num_rows:,}** rows…"):
                try:
                    model = load_model(model_choice)
                    if model is not None:
                        synthetic_data = model.sample(num_rows)
                        st.session_state.synthetic_df = synthetic_data
                        st.success(f"✅  Successfully generated **{num_rows:,}** synthetic records!")
                    else:
                        st.error(f"❌  Model file for **{model_choice}** not found in `models/`.")
                except Exception as e:
                    st.error(f"❌  Error during generation: `{e}`")

        if st.session_state.synthetic_df is not None:
            st.markdown("---")
            st.markdown('<div class="section-header">Generated Data Preview</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Generated Records", f"{len(st.session_state.synthetic_df):,}")
            c2.metric("Features", f"{st.session_state.synthetic_df.shape[1]}")
            st.dataframe(st.session_state.synthetic_df.head(15), width="stretch", height=380)


# ══════════════════════════════════════════════════════════════
# PAGE 4 — COMPARISON
# ══════════════════════════════════════════════════════════════
elif page == "📊  Comparison":
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">📊 Comparison</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Side-by-side analysis of the original and synthetic datasets.</div>',
                unsafe_allow_html=True)

    if st.session_state.original_df is None or st.session_state.synthetic_df is None:
        st.warning("⚠️  Please upload a dataset **and** generate synthetic data first.")
    else:
        orig = st.session_state.original_df
        synth = st.session_state.synthetic_df

        # Overview cards
        st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Rows", f"{orig.shape[0]:,}")
        c2.metric("Original Columns", f"{orig.shape[1]}")
        c3.metric("Synthetic Rows", f"{synth.shape[0]:,}")
        c4.metric("Synthetic Columns", f"{synth.shape[1]}")

        st.markdown("---")

        # Side-by-side preview
        tab_preview, tab_dist, tab_corr = st.tabs(
            ["📋 Side-by-Side Preview", "📊 Feature Distributions", "🔗 Correlation Heatmaps"]
        )

        with tab_preview:
            col_orig, col_synth = st.columns(2)
            with col_orig:
                st.markdown("##### Original Dataset")
                st.dataframe(orig.head(10), width="stretch", height=350)
            with col_synth:
                st.markdown("##### Synthetic Dataset")
                st.dataframe(synth.head(10), width="stretch", height=350)

        with tab_dist:
            common_cols = sorted(set(orig.columns) & set(synth.columns))
            if not common_cols:
                st.error("No common columns found between original and synthetic datasets.")
            else:
                selected_col = st.selectbox("Select a Feature", common_cols)

                if pd.api.types.is_numeric_dtype(orig[selected_col]):
                    fig = make_subplots(rows=1, cols=2, subplot_titles=(
                        f"Original: {selected_col}", f"Synthetic: {selected_col}"
                    ))
                    fig.add_trace(
                        go.Histogram(x=orig[selected_col], name="Original",
                                     marker_color=COLORS["original"], opacity=0.85,
                                     nbinsx=30),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Histogram(x=synth[selected_col], name="Synthetic",
                                     marker_color=COLORS["synthetic"], opacity=0.85,
                                     nbinsx=30),
                        row=1, col=2
                    )
                    fig.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False)
                    fig.update_xaxes(gridcolor="rgba(99,102,241,0.08)")
                    fig.update_yaxes(gridcolor="rgba(99,102,241,0.08)")
                    st.plotly_chart(fig, width="stretch")

                    # Overlay
                    fig2 = go.Figure()
                    fig2.add_trace(go.Histogram(
                        x=orig[selected_col], name="Original",
                        marker_color=COLORS["original"], opacity=0.6, nbinsx=30))
                    fig2.add_trace(go.Histogram(
                        x=synth[selected_col], name="Synthetic",
                        marker_color=COLORS["synthetic"], opacity=0.6, nbinsx=30))
                    fig2.update_layout(
                        **PLOTLY_LAYOUT, barmode="overlay", height=380,
                        title=f"Overlaid Distribution: {selected_col}",
                    )
                    fig2.update_xaxes(gridcolor="rgba(99,102,241,0.08)")
                    fig2.update_yaxes(gridcolor="rgba(99,102,241,0.08)")
                    st.plotly_chart(fig2, width="stretch")

                else:
                    orig_counts = orig[selected_col].value_counts(normalize=True).head(10).reset_index()
                    orig_counts.columns = ["Category", "Proportion"]
                    orig_counts["Source"] = "Original"

                    synth_counts = synth[selected_col].value_counts(normalize=True).head(10).reset_index()
                    synth_counts.columns = ["Category", "Proportion"]
                    synth_counts["Source"] = "Synthetic"

                    combined = pd.concat([orig_counts, synth_counts])

                    fig = px.bar(
                        combined, x="Proportion", y="Category", color="Source",
                        barmode="group", orientation="h",
                        color_discrete_map={"Original": COLORS["original"],
                                            "Synthetic": COLORS["synthetic"]},
                        title=f"Category Proportions: {selected_col} (Top 10)",
                    )
                    fig.update_layout(**PLOTLY_LAYOUT, height=450)
                    fig.update_xaxes(gridcolor="rgba(99,102,241,0.08)")
                    fig.update_yaxes(gridcolor="rgba(99,102,241,0.08)")
                    st.plotly_chart(fig, width="stretch")

        with tab_corr:
            numeric_orig = orig.select_dtypes(include=np.number)
            numeric_synth = synth.select_dtypes(include=np.number)
            common_num = sorted(set(numeric_orig.columns) & set(numeric_synth.columns))

            if len(common_num) < 2:
                st.info("Not enough common numeric columns to compute correlation matrices.")
            else:
                col_h1, col_h2 = st.columns(2)
                with col_h1:
                    corr_orig = numeric_orig[common_num].corr()
                    fig_c1 = px.imshow(
                        corr_orig, text_auto=".2f",
                        color_continuous_scale=["#0f1117", "#6366f1", "#f1f5f9"],
                        title="Original Correlation",
                    )
                    fig_c1.update_layout(**PLOTLY_LAYOUT, height=450)
                    st.plotly_chart(fig_c1, width="stretch")
                with col_h2:
                    corr_synth = numeric_synth[common_num].corr()
                    fig_c2 = px.imshow(
                        corr_synth, text_auto=".2f",
                        color_continuous_scale=["#0f1117", "#06b6d4", "#f1f5f9"],
                        title="Synthetic Correlation",
                    )
                    fig_c2.update_layout(**PLOTLY_LAYOUT, height=450)
                    st.plotly_chart(fig_c2, width="stretch")


# ══════════════════════════════════════════════════════════════
# PAGE 5 — EVALUATION
# ══════════════════════════════════════════════════════════════
elif page == "📈  Evaluation":
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">📈 Evaluation Metrics</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Pre-calculated privacy and utility metrics for all model configurations.</div>',
                unsafe_allow_html=True)

    # ── Utility Metrics ──
    utility_path = "results/utility_metrics.csv"
    privacy_path = "results/privacy_metrics.csv"
    tradeoff_path = "results/tradeoff_summary.csv"

    tab_tradeoff, tab_utility, tab_privacy, tab_graphs = st.tabs(
        ["🔀 Trade-off Summary", "🎯 Utility Metrics", "🔒 Privacy Metrics", "📉 Graphs"]
    )

    with tab_tradeoff:
        if os.path.exists(tradeoff_path):
            tdf = pd.read_csv(tradeoff_path)
            st.markdown('<div class="section-header">Privacy–Utility Trade-off</div>',
                        unsafe_allow_html=True)

            # Metric cards
            cols = st.columns(len(tdf))
            privacy_cls_map = {
                "Very High": "privacy-very-high",
                "High": "privacy-high",
                "Medium": "privacy-medium",
                "Low": "privacy-low",
            }
            for idx, row in tdf.iterrows():
                with cols[idx]:
                    pcls = privacy_cls_map.get(row.get("Privacy_Level", ""), "privacy-medium")
                    st.markdown(f"""
                    <div class="stat-card">
                        <div style="font-size: 1rem; font-weight: 700; color: #f1f5f9;
                             margin-bottom: 0.8rem;">{row['Model']}</div>
                        <div class="stat-label">Accuracy</div>
                        <div class="stat-value">{row['Utility_Accuracy']:.0%}</div>
                        <div style="margin-top: 0.6rem;">
                            <span class="stat-label">Similarity</span>
                            <span style="color: #e2e8f0; font-weight: 600; font-size: 0.95rem;">
                                {row['Similarity']:.0%}</span>
                        </div>
                        <div style="margin-top: 0.4rem;">
                            <span class="stat-label">Attack Acc</span>
                            <span style="color: #e2e8f0; font-weight: 600; font-size: 0.95rem;">
                                {row['Attack_Accuracy']:.0%}</span>
                        </div>
                        <div style="margin-top: 0.6rem;">
                            <span class="privacy-tag {pcls}">{row['Privacy_Level']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.dataframe(tdf, width="stretch")

            # Interactive scatter
            fig = px.scatter(
                tdf, x="Attack_Accuracy", y="Utility_Accuracy",
                size="Similarity", color="Model",
                color_discrete_sequence=["#10b981", "#06b6d4", "#f59e0b", "#f43f5e"],
                hover_data=["F1_Score", "Privacy_Level"],
                title="Privacy-Utility Trade-off (bubble size = similarity)",
                size_max=40,
            )
            fig.update_layout(**PLOTLY_LAYOUT, height=420)
            fig.update_xaxes(title="Privacy Leakage (Attack Acc) → Lower is Better",
                             gridcolor="rgba(99,102,241,0.08)")
            fig.update_yaxes(title="Utility (Accuracy) → Higher is Better",
                             gridcolor="rgba(99,102,241,0.08)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("⚠️  `results/tradeoff_summary.csv` not found. Run the trade-off analysis script first.")

    with tab_utility:
        if os.path.exists(utility_path):
            udf = pd.read_csv(utility_path)
            st.markdown('<div class="section-header">Baseline CTGAN — Utility</div>',
                        unsafe_allow_html=True)

            metric_cols = st.columns(len(udf))
            color_cycle = [COLORS["original"], COLORS["synthetic"], COLORS["accent"],
                           COLORS["emerald"], COLORS["amber"]]
            for idx, (_, row) in enumerate(udf.iterrows()):
                with metric_cols[idx]:
                    val = row["Value"]
                    st.metric(row["Metric"], f"{val:.4f}")

            # Radar chart
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=udf["Value"].tolist() + [udf["Value"].iloc[0]],
                theta=udf["Metric"].tolist() + [udf["Metric"].iloc[0]],
                fill="toself",
                fillcolor="rgba(99,102,241,0.15)",
                line=dict(color=COLORS["original"], width=2),
                name="Utility",
            ))
            fig.update_layout(
                **PLOTLY_LAYOUT, height=420,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(range=[0, 1], gridcolor="rgba(99,102,241,0.15)",
                                    tickfont=dict(size=10)),
                    angularaxis=dict(gridcolor="rgba(99,102,241,0.15)"),
                ),
                title="Utility Radar",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("⚠️  `results/utility_metrics.csv` not found.")

    with tab_privacy:
        if os.path.exists(privacy_path):
            pdf = pd.read_csv(privacy_path)
            st.markdown('<div class="section-header">Membership Inference Attack Results</div>',
                        unsafe_allow_html=True)
            st.dataframe(pdf, width="stretch")

            # Grouped bar chart
            metrics_to_plot = [c for c in pdf.columns if c != "Model"]
            fig = go.Figure()
            colors_seq = [COLORS["original"], COLORS["synthetic"],
                          COLORS["amber"], COLORS["rose"], COLORS["accent"]]
            for i, metric in enumerate(metrics_to_plot):
                fig.add_trace(go.Bar(
                    x=pdf["Model"], y=pdf[metric], name=metric,
                    marker_color=colors_seq[i % len(colors_seq)],
                    opacity=0.88,
                ))
            fig.update_layout(
                **PLOTLY_LAYOUT, barmode="group", height=420,
                title="Privacy Metrics by Model",
                xaxis_title="Model", yaxis_title="Score",
            )
            fig.update_xaxes(gridcolor="rgba(99,102,241,0.08)")
            fig.update_yaxes(gridcolor="rgba(99,102,241,0.08)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.warning("⚠️  `results/privacy_metrics.csv` not found.")

    with tab_graphs:
        st.markdown('<div class="section-header">Pre-generated Analysis Graphs</div>',
                    unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("results/graphs/accuracy_vs_epsilon.png"):
                st.image("results/graphs/accuracy_vs_epsilon.png",
                         caption="Utility vs Privacy Budget (ε)", width="stretch")
            else:
                st.info("Graph not found: `accuracy_vs_epsilon.png`")
        with col2:
            if os.path.exists("results/graphs/privacy_vs_epsilon.png"):
                st.image("results/graphs/privacy_vs_epsilon.png",
                         caption="Privacy Leakage vs Privacy Budget (ε)", width="stretch")
            else:
                st.info("Graph not found: `privacy_vs_epsilon.png`")

        col3, col4 = st.columns(2)
        with col3:
            if os.path.exists("results/graphs/similarity_vs_epsilon.png"):
                st.image("results/graphs/similarity_vs_epsilon.png",
                         caption="Similarity vs Privacy Budget (ε)", width="stretch")
            else:
                st.info("Graph not found: `similarity_vs_epsilon.png`")
        with col4:
            if os.path.exists("results/graphs/tradeoff.png"):
                st.image("results/graphs/tradeoff.png",
                         caption="Privacy-Utility Trade-off Scatter", width="stretch")
            else:
                st.info("Graph not found: `tradeoff.png`")


# ══════════════════════════════════════════════════════════════
# PAGE 6 — DOWNLOAD
# ══════════════════════════════════════════════════════════════
elif page == "💾  Download":
    st.markdown('<div class="hero-title" style="font-size: 2.2rem;">💾 Download</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Export your generated or pre-generated synthetic datasets.</div>',
                unsafe_allow_html=True)

    # Session-generated download
    if st.session_state.synthetic_df is not None:
        st.markdown('<div class="section-header">Session-Generated Data</div>',
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card" style="display: flex; align-items: center; gap: 1.5rem;">
            <div style="font-size: 2.5rem;">📄</div>
            <div>
                <div style="color: #f1f5f9; font-weight: 700; font-size: 1.05rem;">
                    generated_synthetic_data.csv</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">
                    {len(st.session_state.synthetic_df):,} rows ×
                    {st.session_state.synthetic_df.shape[1]} columns</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        csv_data = st.session_state.synthetic_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️  Download Generated Dataset (CSV)",
            data=csv_data,
            file_name="generated_synthetic_data.csv",
            mime="text/csv",
            width="stretch",
        )
        st.markdown("---")

    # Pre-generated files
    st.markdown('<div class="section-header">Pre-generated Datasets</div>',
                unsafe_allow_html=True)

    prebuilt_files = {
        "baseline.csv": {
            "path": "data/synthetic/baseline.csv",
            "desc": "No differential privacy — highest fidelity.",
            "icon": "🟥",
            "privacy": "No DP",
        },
        "epsilon_50.csv": {
            "path": "data/synthetic/epsilon_50.csv",
            "desc": "ε = 5.0 — moderate privacy, high utility.",
            "icon": "🟧",
            "privacy": "ε = 5.0",
        },
        "epsilon_10.csv": {
            "path": "data/synthetic/epsilon_10.csv",
            "desc": "ε = 1.0 — strong privacy, good utility.",
            "icon": "🟦",
            "privacy": "ε = 1.0",
        },
        "epsilon_05.csv": {
            "path": "data/synthetic/epsilon_05.csv",
            "desc": "ε = 0.5 — very strong privacy, lower utility.",
            "icon": "🟩",
            "privacy": "ε = 0.5",
        },
    }

    cols = st.columns(2)
    for idx, (fname, info) in enumerate(prebuilt_files.items()):
        with cols[idx % 2]:
            if os.path.exists(info["path"]):
                file_size = os.path.getsize(info["path"])
                size_mb = file_size / (1024 * 1024)
                st.markdown(f"""
                <div class="glass-card" style="padding: 1.2rem 1.5rem;">
                    <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.8rem;">
                        <span style="font-size: 1.8rem;">{info['icon']}</span>
                        <div>
                            <div style="color: #f1f5f9; font-weight: 700;">{fname}</div>
                            <div style="color: #94a3b8; font-size: 0.78rem;">{info['desc']}</div>
                        </div>
                    </div>
                    <div style="display: flex; gap: 1rem; margin-bottom: 0.8rem;">
                        <span class="tech-badge badge-purple">{info['privacy']}</span>
                        <span class="tech-badge badge-blue">{size_mb:.1f} MB</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                with open(info["path"], "rb") as f:
                    st.download_button(
                        f"⬇️  Download {fname}",
                        data=f,
                        file_name=fname,
                        mime="text/csv",
                        width="stretch",
                        key=f"dl_{fname}",
                    )
            else:
                st.markdown(f"""
                <div class="glass-card" style="padding: 1.2rem 1.5rem; opacity: 0.5;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <span style="font-size: 1.8rem;">⛔</span>
                        <div>
                            <div style="color: #f1f5f9; font-weight: 700;">{fname}</div>
                            <div style="color: #f43f5e; font-size: 0.78rem;">File not found</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
