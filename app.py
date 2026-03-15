"""
MMM Platform — Streamlit Application

Multi-client Marketing Mix Modeling dashboard.
Run with: streamlit run app.py
"""

import streamlit as st
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from ui.layout import render_sidebar

st.set_page_config(
    page_title="MMM Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS: Inter font + dark theme polish ──────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stMarkdown, .stText, .stDataFrame,
h1, h2, h3, h4, h5, h6, p, span, div, label, input, textarea, select, button {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Plotly charts: transparent background to match theme */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}

/* Hide default auto-generated sidebar nav (we build our own below) */
[data-testid="stSidebarNav"] {
    display: none;
}

/* Narrower left sidebar */
[data-testid="stSidebar"] {
    min-width: 180px !important;
    max-width: 180px !important;
    width: 180px !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: 180px !important;
}

/* ── All headings: smaller, lighter ── */
h1, h1 span { font-size: 1.3rem !important; font-weight: 300 !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
h2, h2 span { font-size: 1.05rem !important; font-weight: 400 !important; }
h3, h3 span { font-size: 0.95rem !important; font-weight: 400 !important; }
h4, h4 span, h5, h5 span, h6, h6 span { font-size: 0.88rem !important; font-weight: 400 !important; }

/* ── Sidebar overrides: even smaller, no uppercase ── */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h1 span,
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h2 span,
[data-testid="stSidebar"] h3, [data-testid="stSidebar"] h3 span,
[data-testid="stSidebar"] h4, [data-testid="stSidebar"] h4 span {
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: normal !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    font-size: 0.8rem !important;
}
/* Sidebar text spans — but NOT the collapse button icon */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] span {
    font-size: 0.8rem !important;
}
/* Hide the sidebar collapse button icon-name leak */
[data-testid="collapsedControl"] span,
[data-testid="stSidebarCollapsedControl"] span,
button[kind="headerNoPadding"] span {
    font-size: 0 !important;
}

/* ── Metric cards: compact values and labels ── */
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {
    font-size: 1.15rem !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    font-size: 0.78rem !important;
}

/* Full-width layout — reduce default padding */
.stMainBlockContainer, .block-container {
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* Tighten top padding */
.block-container {
    padding-top: 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Authentication ──────────────────────────────────────────

def check_password():
    """Simple password gate for the app."""
    if "authenticated" in st.session_state and st.session_state["authenticated"]:
        return True

    st.title("MMM Platform")
    st.markdown("Enter the team password to continue.")

    password = st.text_input("Password", type="password", key="password_input")

    if st.button("Log in", type="primary"):
        correct_password = st.secrets.get("auth", {}).get("password", "")
        if password == correct_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()

# ── Load config ──────────────────────────────────────────────

@st.cache_data
def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Inject Windsor API key from Streamlit secrets
    windsor_key = st.secrets.get("windsor", {}).get("api_key", "")
    if windsor_key:
        cfg["windsor"]["api_key"] = windsor_key

    return cfg

config = load_config()

# ── Sidebar: Navigation + Client Selection ──────────────────

st.session_state["config"] = config
render_sidebar()

st.sidebar.markdown("---")
if st.sidebar.button("Log out"):
    st.session_state["authenticated"] = False
    st.rerun()
st.sidebar.caption("Built for DTC e-commerce brands")

selected_client = st.session_state.get("selected_client", "juniper")
selected_display = st.session_state.get("client_config", {}).get("display_name", selected_client)

# ── Main Page ────────────────────────────────────────────────

st.title(f"Marketing Mix Model — {selected_display}")

st.markdown("""
Welcome to the MMM Platform. Use the pages in the sidebar to:

1. **Client Overview** — Revenue decomposition, channel contributions, model fit
2. **Channel Analysis** — Deep-dive per channel: ROAS, saturation, adstock
3. **Budget Optimizer** — What-if scenarios and optimal budget allocation
4. **Event Calendar** — Manage promotional events, product drops, holidays
5. **Spend-aMER** — GP3-optimized spend planning by month with seasonal adjustments

### Getting Started

To run the model for a client:
1. Ensure all channels are connected in Windsor.ai (see sidebar)
2. Go to **Client Overview** and click \"Fetch Data & Run Model\"
3. Review the results across all pages
4. Use **Spend-aMER** for GP3-optimized monthly spend recommendations
""")

# Quick stats if results exist
results_path = Path(f"results/{selected_client}")
if results_path.exists():
    st.success("Model results available — navigate to pages to explore.")
else:
    st.info("No model results yet for this client. Go to Client Overview to run the model.")
